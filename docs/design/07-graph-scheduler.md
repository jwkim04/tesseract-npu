# PV: Tesseract Pivot, the graph scheduler

Area PV specifies Tesseract Pivot, the graph scheduler of the ggml backend. It orders the nodes of a graph
split before the backend lowers them, so that independent matrix products run on the MVM unit while the
control core and the glue pipeline are busy with work that does not depend on them. It is a scheduler over a
dependency graph with a cost model of the device. The other areas are EX (execution model), VU (vector unit),
AT (attention and KV cache), CC (control core), DB (debug) and SW (software and verification). Numbers marked
estimate are design figures. The constants of §4 are design values that rank alternatives; they do not
predict a time.

---

## 1. Purpose and where it runs

- A decode token leaves the MVM unit idle while the MoE router runs (softmax on the VU, top-k on the control
  core), while expert outputs are combined, while the Gated DeltaNet recurrence runs, and between the Q, K and V
  products of a layer. Other matrix products do not depend on that work, but llama.cpp's node order, the
  depth-first post-order of its `ggml_build_forward_expand` calls, puts them later in the graph.
- A MUL_MAT that is lowered as MUL_MAT_BEGIN and MUL_MAT_END no longer holds the DMA queue until its MVM is done
  (SW §7.3, §9.3). A product that is issued earlier then runs under the idle window. The builder decides that
  split from the final record order and the byte ranges of the records, so Pivot changes the node order and
  nothing else.
- Pivot runs on the host, in the backend's `graph_optimize` hook. ggml-backend's scheduler calls the hook on a
  view of each split after it has split the graph and before ggml-alloc allocates it. The backend may permute
  `cgraph->nodes` and nothing else. The hook runs once for each graph the scheduler splits, outside the device
  mutex. It searches the first graph of a structure; a later graph of that structure gets the same order from a
  memo (§6.6). It reads no tensor data and nothing from the device but what the device open fixed (the channel count
  and the dual weight stream).
- Pivot has two parts. The core (`sw/ggml-tesseract/tsr-pivot.{h,cpp}`, in `tsr_lower`) knows nothing of ggml: it
  takes units with predecessors and costs and returns an order (§2 to §5). The adapter
  (`sw/ggml-tesseract/tsr-pivot-graph.{h,cpp}`) turns the nodes of a split into units, applies the order and
  checks it (§6).
- The pass is general. No rule names a model. Gemma 4, Qwen3.5 and Qwen3.8 layers, and a dense model, go through
  the same code (PV-1).

---

## 2. The graph

### 2.1 Units

The core takes a vector of units in their original order. A unit is what must stay together: the front end
merges the nodes that one record lowers, or between which the builder makes no hazard check, into one unit.
The unit number is its place in the original order.

- `preds` lists the units that must come before this one. Every pred is a smaller number than the unit's own,
  so the original order is valid. Data dependencies and order constraints are both preds.
- The class and costs, in core cycles (§2.2). A field that a class does not use is zero.

| Field | Used by | Meaning |
|---|---|---|
| `issue` | all | control-core time of the unit's records. For an MVM unit it is the cost when the unit is not deferred, or is closed by its own next unit: one MUL_MAT record |
| `issue_open` | MVM | control-core time when the unit stays open: a MUL_MAT_BEGIN record |
| `issue_close` | MVM | control-core time of the record that closes it: a MUL_MAT_END |
| `body` | GLUE, BLOCK | time of the unit |
| `ld`, `qact`, `mvm`, `back`, `defer` | MVM | LD of x on the glue pipeline, QACT unit, MVM unit, ST of the result, and whether the ST can wait for the first reader |
| `ids`, `mvm`, `tail`, `back` | MVM_ID | LD of the ids and the join, all tiles on the MVM unit, how long before the MVM end the control core is free, ST |
| `front`, `mvm`, `back` | ATTN | work before the MVM unit starts, time on the MVM unit, ST |

### 2.2 Classes

| Class | What it is | What it holds |
|---|---|---|
| GLUE | LD, VU and ST work: norms, RoPE, activations, copies, adds | the glue pipeline for `body` |
| MVM | a matrix product with a weight | LD of x on the glue pipeline, QACT, the MVM unit; its ST may be deferred |
| MVM_ID | a MUL_MAT_ID with its scale chain | the control core joins on the ids, then the MVM unit for all tiles |
| BLOCK | the MoE router chain, the Gated DeltaNet recurrence | the control core and the glue pipeline for `body`; the MVM unit is free |
| ATTN | FLASH_ATTN_EXT | the MVM unit |

A unit has MVM work when its class is MVM, MVM_ID or ATTN.

### 2.3 Dependencies

Data dependencies and order constraints are the same thing to the core: a pred. The adapter finds them (§6.3).
The core refuses a graph with a pred that is not a smaller number; `ok` is then false, `error` names the unit,
and the order is the identity.

---

## 3. The machine model

`tsr_pivot_eval(units, order)` is the model: the time in core cycles of a valid order. It is coarse on purpose.
It ranks orders and does not predict the time of one. All numbers are 64-bit integers and every step is integer
arithmetic, so the same input gives the same result on every host.

### 3.1 State

- Four clocks: `t_issue` (the control core), `t_glue` (the in-order glue pipeline, which stands for the DMA
  queue and the VU together), `t_qact` (the QACT unit) and `t_mvm` (the MVM unit).
- Per unit: `done` (its result is in DRAM) and `mvm_end`.
- The open units: the MVM units whose ST has not been issued. At most `TSR_PIVOT_MAX_OPEN` = 8 are open, in the
  order they were opened. The limit is the executor's table of open MUL_MAT_BEGIN records (SW §7.3).

### 3.2 Processing a unit

Units are processed in the order given. `next` is the unit after this one in the order (none at the end).

1. The unit stays open when its class is MVM, it has `defer`, and `next` does not list it in its preds.
2. Every open unit among this unit's preds is closed, in the order the open units were opened. Closing a unit
   adds its `issue_close` to `t_issue`, starts its ST at max(`t_glue`, its `mvm_end`, `t_issue`) and takes
   `back`. That sets `t_glue` and the unit's `done`.
3. If the unit stays open and eight units are open, the oldest is closed first (as in step 2). The check comes
   after step 2, so a unit that closes preds does not close the oldest as well.
4. `t_issue` grows by `issue_open` when the unit stays open and by `issue` otherwise. The cost is added before
   the unit starts: a unit cannot start before its own record is issued (PV-7).
5. `ready` is the latest `done` of the preds. The start is `s` = max(`t_glue`, `t_issue`, `ready`).
6. The class rule:

| Class | Rule |
|---|---|
| GLUE | `t_glue` = `s` + `body`; `done` = `t_glue` |
| MVM | LD ends at `s` + `ld` (`t_glue`). QACT ends at max(LD end, `t_qact`) + `qact` (`t_qact`). The MVM runs from max(QACT end, `t_mvm`) for `mvm` (`t_mvm`, `mvm_end`). A unit that stays open joins the open list. Otherwise its ST follows at max(`t_glue`, `mvm_end`) for `back`: `t_glue` and `done` take its end |
| MVM_ID | the join ends at `s` + `ids` (`t_glue`, and `t_issue` is at least that). The MVM runs from max(join end, `t_mvm`) for `mvm`. `t_issue` is at least `mvm_end` − `tail`. `done` = `t_glue` = `mvm_end` + `back` |
| BLOCK | `t_glue` = `t_issue` = `done` = `s` + `body` |
| ATTN | the MVM runs from max(`s` + `front`, `t_mvm`) for `mvm`. `done` = `t_glue` = `mvm_end` + `back` |

At the end every open unit is closed, in the order it was opened. The result is the latest of the four clocks.

`tsr_pivot_eval` takes an optional `after`, the unit that follows the last unit of `order` when `order` is a
prefix of a longer order. Step 1 looks at it for the last unit. The search uses it (§5.4).

### 3.3 What the model keeps

- A deferred product (BEGIN, then END at the first reader): its ST waits for the first reader instead of
  holding the glue pipeline, so independent work runs under its MVM.
- The price of the split. A split product costs `issue_open` + `issue_close` (820 + 460) where a MUL_MAT costs
  `issue` (830): 450 core cycles more of control-core time. A move whose gain is smaller than that price is not
  a gain in the model.
- The idle glue pipeline. After a BLOCK or MVM_ID unit `t_issue` equals `t_glue`, so the next unit pays its
  issue cost with the glue pipeline idle. The same unit placed earlier, while the glue pipeline is busy, hides
  it. A small product can gain from this alone.

### 3.4 What the model leaves out

- Queue depths: the MVM unit's 4, the VU's 16 and the DMA's 8 entries (CC §7.4).
- The VU and the DMA as separate units: both are the glue pipeline.
- DDR contention between the DMA and the weight streams on one channel.
- Hazards other than a read after a write. The builder opens a BEGIN when the next record leaves the y rows
  alone, which includes write hazards; the model opens it when the next unit does not list it as a pred.
- Closings that the executor makes on its own: a refused ring allocation closes every open BEGIN (SW §7.3). A
  multi-tile record's probe for a second buffer does that too.
- The JOIN records the builder adds above two dependencies.
- The cost of an attention that depends on the context length, and the SPM capacity.

---

## 4. The costs

The adapter computes the costs from the structure of the graph and the device (§6.4). All constants are in one
table, `tsr_pivot_costs` in `tsr-pivot-graph.h`, and the control-core costs are in `tsr-pivot.h`. They are
design values. The tests check the arithmetic of each, not the time of a device.

### 4.1 Control-core cost of a record

Measured by `fw_exec_perf` in clk_rv cycles; one clk_rv cycle is two core cycles.

| Constant | Value | Use |
|---|---|---|
| `TSR_PIVOT_ISSUE_MUL_MAT` | 830 | `issue` of a weight MUL_MAT |
| `TSR_PIVOT_ISSUE_BEGIN` | 820 | `issue_open` |
| `TSR_PIVOT_ISSUE_END` | 460 | `issue_close` |
| `TSR_PIVOT_ISSUE_SHORT` | 700 | one short record: `issue` of a GLUE unit per record, of an ATTN unit |
| `TSR_PIVOT_MAX_OPEN` | 8 | the open table (SW §7.3) |

A BLOCK or MVM_ID unit has `issue` 0: its control-core time is in its own costs.

### 4.2 Device work

| Class | Field | Value | Source |
|---|---|---|---|
| MVM | `mvm` | beats per channel × 100 / rate, at least 1. Beats per channel = ⌈N / channels⌉ × row beats of one row of K elements in the weight's device format. Rate 141 with the dual weight stream (`CAPS` bit 16), 100 without | `cyc_mvm` in `sw/sim/tsr_sim.c`; weight beats of `plan.hpp`; the rate of a decode MVM with two weight streams per channel ([rtl-mvm.md](../rtl-mvm.md) §15) |
| MVM | `ld` | 64 + K / 16 | `cyc_ldst` |
| MVM | `qact` | 16 · (K / 32) + 44 for Q8_0 and every format that is converted to it; 17 · (K / 256) + 72 for Q4_K | `cyc_qact` (SW-80) |
| MVM | `back` | 110 + N / 16 | `cyc_st` |
| MVM | `defer` | set unless a SCALE or softcap partner fuses with the product | the builder defers a plain MUL_MAT only (SW §9.3) |
| MVM_ID | `mvm` | N_USED × the time of one expert's tiles at the MVM rule above | `cyc_mvm` |
| MVM_ID | `ids`, `back` | 400, 300 | gaps read from decode profiles of the simulator |
| MVM_ID | `tail` | the smaller of `mvm` and two experts' tile times | gap from the same profiles |
| BLOCK, router | `body` | 6,500 + 96 × n_expert | gap from decode profiles of the simulator |
| BLOCK, Gated DeltaNet | `body` | 3,300 + H_v × (the head time of VU §7.12.8 + 126) | `cyc_gdn`; the head time for S_v 128 is 2,914, so 3,040 with the margin |
| ATTN | `front`, `mvm`, `back` | 500, 6,000, 300 | the real time depends on the context; a coarse mid value |
| GLUE | `body` | per record: 1,300 + 3 × (first dimension / 16) | gap from decode profiles of the simulator |

The head time of the Gated DeltaNet is (S + 1) · S / 8 + ⌈19 S / 16⌉ + 9 S / 2 + 122 + 126 cycles for a head of
S_v = S rows and columns.

N is the number of rows of the weight and K its first dimension. The weight of a MUL_MAT_ID is one expert's
matrix.

---

## 5. The search

`tsr_pivot_run(units, params)` returns the new order, the model's time before and after, the list of accepted
moves and the number of passes. It is deterministic: the same units and parameters give the same result.

### 5.1 Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `window` | 64 | a unit is tried at the positions of this many units before it |
| `max_moved` | 8 | a move of more units, the unit and its ancestors, is not tried |
| `threshold` | 256 | a move is taken when it lowers the model's time by at least this many cycles; a value below 1 is taken as 1 |
| `horizon` | 32 | units after the moved span that a candidate is scored on |

### 5.2 The hoist move

`hoist(u, p)` takes the unit u and every ancestor of u that lies at position p or later, and puts them, in
their relative order, right before position p. The other units of the span keep their relative order behind
them.

The result is always a valid order. Call the span S = positions p to i, M the moved set and R the rest of S.

- A moved unit's preds that lie in S are ancestors of u, so they are in M and stay in front of it. Its preds
  before p stay in front. The unit at p is not an ancestor of u (such a p is not a candidate, §5.3), so it is no
  pred of a moved unit.
- A unit of R has its preds in M (now in front of all of R), in R (relative order kept) or before p.
- A unit after position i has its preds in S, which stay in front, or outside S.

So no unit lands before a pred, and the move needs no validity check.

### 5.3 Passes

The search starts from the original order, which must be valid. A pass visits the positions i = 1 … n − 1 in
order. The unit u at position i is tried when it has MVM work. A unit without MVM work (GLUE, BLOCK) is never
hoisted for itself; it moves only as an ancestor.

For u at position i:

1. Positions p = i − 1 down to max(0, i − `window`) are the candidates. The sweep keeps the moved set: u, and
   each ancestor met on the way (found with a mark per unit, no clearing between candidates).
2. A position p whose unit is an ancestor joins the moved set and is not a candidate: the order it would give is
   the one of the position after it. If the moved set has more than `max_moved` units the sweep ends.
3. For any other p the candidate order is: the moved units in their relative order, then the other units of
   [p, i] in their relative order.
4. Each candidate is scored (§5.4). The identity is scored the same way. The candidate with the lowest score is
   kept; a tie keeps the first one tried, which is the one with the least movement (p descending, strict
   comparison).
5. The best candidate is a move when its score is lower than the identity's by at least `threshold`. It is then
   confirmed on the rest of the graph (§5.4) and is accepted only when the model's time of the whole graph falls
   by at least `threshold`.
6. An accepted move is recorded (the unit, `from` = i, `to` = p, the number of moved units, the gain) and the
   order is changed. The pass goes on with position i + 1.

Passes repeat until one accepts no move.

### 5.4 Scoring

The search keeps `snap[k]`, the model's state before position k of the current order, and the `done` and
`mvm_end` of the units before it. A candidate is scored from a saved state, not from the start of the graph:

- A saved state before position p contains the decision whether the unit at p − 1 stays open, and that decision
  looks at the next unit, which the move changes. So a trial starts one unit earlier, at the state before p − 1
  (before position 0 for p = 0).
- The trial runs the model on the candidate's units from that state: the changed span and the `horizon` units
  that follow it, with the unit after them as `after`. The identity is scored on the same units to the same end.
  The trial writes its own copies of `done` and `mvm_end`, so the saved values of the units before it are left
  as they were.
- The confirmation runs from the state before p − 1 to the end of the graph.
- After an accepted move the saved states from p − 1 on are recomputed, lazily, as far as the next unit needs.

### 5.5 Termination, the fixed point, determinism

- Every accepted move lowers the model's time of the whole graph by at least `threshold`, which is at least 1.
  The time is a non-negative integer, so the search ends. With the clamp a move that gains nothing is never
  taken, so the search cannot undo its own moves forever.
- The result is never slower in the model than the input: `cycles_after` is at most `cycles_before`.
- The last pass accepts no move and decides nothing from the unit numbers, only from the positions and the
  costs. A second run on the result (the units renumbered in the new order) makes the same last pass and moves
  nothing: the result is a fixed point.
- No step reads a clock, a random source or an address. The same input gives the same order on every host.

### 5.6 Complexity

A pass takes about M × `window` trials, M being the number of units with MVM work. A trial runs at most
`window` + `horizon` + a few model steps, and a step costs a scan of the unit's preds and of the open table
(eight entries). The confirmation of an accepted move runs the model to the end of the graph, so the accepted
moves add n steps each. The ancestors of a candidate are found by one sweep, not by a search per candidate.

### 5.7 Validator

`tsr_pivot_valid(units, order)` is true when every pred is in range, `order` is a permutation of the unit
numbers and every unit follows its preds. `why` names the rule that fails. The adapter and the tests use it.

---

## 6. The ggml adapter

`tsr_pivot_optimize(cg, dev, params, ..., memo)` is the hook's body. For the first graph of a structure it analyzes
the graph, runs the core, permutes the nodes, validates them and restores the original order if the validator
refuses (§6.1 to §6.5). A later graph of the same structure takes the order of the first one (§6.6).

### 6.1 Nodes and units

A compute node is a node with the COMPUTE flag that is not a no-op (NONE, VIEW, RESHAPE, PERMUTE, TRANSPOSE).
Emptiness is not looked at. Whether a node has rows depends on the batch: the in-place SCALE that zeroes the
recurrent-state cell of a new sequence has no rows when the sequence goes on. The units, and so the order, must
not depend on the batch (§7.2).

A unit is a run of nodes of the original order, so every producer-to-consumer edge goes from a smaller unit to a
larger one, as the core requires.

- Groups kept together. The adapter calls the builder's structural matchers, declared in `tsr-build.h`, so the
  builder and the adapter cannot disagree on a group. The matchers check structure alone: no device, no
  address, no token count. A node that matches may still not fuse in the builder (placement, a node written in
  between); the unit then holds nodes the builder lowers separately, which costs nothing.

| Group | Matcher | Members |
|---|---|---|
| MUL_MAT with its epilogue | `tsr_fusion_user`, `tsr_fuse_pair_shape`, `tsr_softcap_shape` | the MUL_MAT, whose result has one use, then the ADD (a bias or the residual) or the SCALE, and for a SCALE the softcap chain TANH and the closing SCALE |
| RMS_NORM with its MUL | `tsr_fuse_pair_shape` | the RMS_NORM and the MUL by a weight |
| Router chain | `tsr_moe_route_match` | SOFT_MAX, ARGSORT, GET_ROWS, SUM_ROWS, CLAMP, DIV |
| Expert scale chain | `tsr_moe_scale_match` | MUL_MAT_ID, the REPEAT and GET_ROWS of the scale table, the MUL |
| SSM_CONV with its SILU | `tsr_conv_group_shape` | SSM_CONV and the SILU after it |

- When the members of a group are not contiguous, the whole span from the first to the last compute node is one
  unit.
- No-op nodes join the unit of the next compute node in the original order. The last run of no-ops joins the
  unit before it.
- Groups with order constraints only. The Gated DeltaNet state gather, GATED_DELTA_NET and state CPY, and the
  state CPYs of an SSM_CONV are separate units; their order is kept by the constraints of §6.3 (`tsr_gdn_state_shape`
  and `tsr_conv_group_shape` find the nodes). The gather and the state CPYs cost nothing: the builder folds them
  into the record of the node that uses the state.
- State plumbing. An in-place SCALE or a CPY into a view of a leaf (the recurrent-state caches) costs nothing,
  with rows or without. So does the GET_ROWS behind the state operand of a GATED_DELTA_NET, whether or not the
  builder absorbs it: that depends on the batch (the number of sequences, the snapshot slots, extra states).

### 6.2 Dependencies

- Data: for every source of every node, the unit of the node that produced it (through views). A source that is
  not a node of the split (a leaf, a weight, a node of another split) has no edge.
- Order constraints: see §6.3.
- A graph whose nodes are not in topological order, or whose constraint would point backwards, is left alone.

### 6.3 Order constraints

ggml orders some nodes only through a buffer, not through a source. For every base tensor that a compute node
writes through a view or in place, all compute nodes that read it (through a source, or through a view) or write
it keep their original relative order. That covers the KV cache planes (SET_ROWS writes a view, FLASH_ATTN reads
a view; the builder gives the FLASH_ATTN its FA_NEW word only when the SET_ROWS_KV came first, AT §6.6 rule 7), the
recurrent state caches (SCALE of a cell, GET_ROWS, GATED_DELTA_NET, CPY) and in-place ops. Each chain gives an edge
from each node to the next one in the chain. Two reads of one written buffer keep their order too; the rule is
conservative.

Two more rules:

- The state CPYs of an SSM_CONV group go before the SSM_CONV, as the builder requires (`tsr_conv_group_shape`).
- A split whose first node is a MUL_MAT_ID keeps it first: every other unit gets unit 0 as a pred. llama.cpp's
  expert-copy callback (`sched_copy_experts`) reads the first node of a split and copies only the selected
  experts when it is the MUL_MAT_ID that reads host-resident weights.

### 6.4 Classes and costs from nodes

The class of a unit is the first rule that matches one of its compute nodes:

1. A router chain: BLOCK, `body` from the expert count (the first dimension of the SOFT_MAX).
2. A GATED_DELTA_NET with a value input: BLOCK, `body` from the head size S_v and the head count H_v of the value
   input.
3. A MUL_MAT_ID with an expert weight: MVM_ID; N_USED is the first dimension of the ids.
4. A FLASH_ATTN_EXT: ATTN, `issue` 700.
5. A MUL_MAT whose source 0 is a two-dimensional weight leaf: MVM, with `defer` unless a SCALE or softcap
   partner fuses with it, and `issue`, `issue_open`, `issue_close` as in §4.1.
6. Anything else: GLUE. `body` is the sum over the records of the unit and `issue` is 700 per record. A record is
   a compute node that is not folded into another record: the fused partners of a group and the state
   plumbing are not records. A CONCAT along dimension 0 of a transposed source takes its
   second dimension as the first dimension of the cost, because its first dimension grows with the token count.

The costs come from weight shapes, expert counts, the head size and the device (channels and the dual weight
stream), never from the number of tokens (PV-8).

`defer` is set whatever the number of tokens. The builder does not split a product of more than one T-tile; the
model does not know the tile count, which would make the permutation depend on the batch (§7.2).

### 6.5 Applying and guarding

- The new node order is the units in the core's order, each unit's nodes in their original order, the no-ops
  included. An order equal to the identity changes nothing.
- The adapter permutes `cg->nodes` in place and runs the node-level validator `tsr_pivot_check_order` on the
  result. It checks:
  - the order is a permutation of the nodes;
  - every node follows the producers of its sources;
  - node 0 is still first when it is a MUL_MAT_ID (§6.3);
  - the order constraints of §6.3, recomputed from the original order, and the builder's `before` edges;
  - no compute node outside a fused group lies between its first and last node.
- If the validator refuses, the original order is put back, one warning goes to stderr and the graph is lowered
  as it was. The same happens when the core's unit order is not a permutation.

This is the path of the first graph of a structure. A later graph does not search; §6.6 says what it does.

### 6.6 One order per structure

Two graphs of one structure must get the same permutation (§7.2). The pass makes that hold by construction: it
remembers the order of the first graph of each structure and gives it to every later graph of that structure.

The structure key. `tsr_pivot_structure_key(cg)` is a 64-bit hash of the graph. For each node in order it hashes
the op (and the unary op), the node's type and its use count. For the view base and for every
source it hashes one of three things: absent; a node of this graph, by its position; a tensor outside the graph,
by its op, its type and a number given in the order of first reference, so the key knows which links reach the
same outside tensor. For a MUL_MAT or MUL_MAT_ID whose source 0 is a leaf it also hashes the weight's type and its
four dimensions. The key leaves out the shape of a node, every offset, the op parameters, every address and every
name. These are the things that change with the batch: llama.cpp keeps the topology of a graph, with its ops, its
links and its use counts, the same for every batch, and the weights are the same for every batch.

The memo. `tsr_pivot_memo` maps a key to a node order; an empty order means that the original order stays. The key of
the map mixes the structure key with the device (the channel count and the dual weight stream) and the search
parameters. The memo holds at most 256 entries. The backend keeps one memo in each backend instance. One backend
instance belongs to one llama.cpp context, which has its own scheduler and its own ggml-alloc plan, so the memo
covers exactly the graphs that share a plan. The scheduler calls the hook from one thread, so the memo needs no
lock.

The key also leaves out the COMPUTE flag. llama.cpp selects the branches of a graph by that flag
(`ggml_build_forward_select`, for example the token and the embedding input paths), and ggml-alloc plans a node with
or without the flag, so graphs that differ in the flag alone share a buffer plan. They share a key, and the later
graph is checked against the order of the first: it accepts that order or it is a conflict.

- The first graph of a structure takes the path of §6.5. The memo then holds its node order. When the pass applied
  an order, the memo also holds the key of the permuted graph, with an empty order: a scheduler that reserves a
  graph and then computes the same graph calls the hook twice on it, and the second call changes nothing.
- A later graph of the structure is not searched. The adapter analyzes that graph, runs `tsr_pivot_check_order`
  on the remembered order against that analysis (the checks of §6.5) and permutes the nodes.
- A first graph that kept its order (nothing to gain, or the core or the validator refused) is remembered with an
  empty order, and every later graph of the structure keeps its order too. So does a graph with nothing to
  schedule.
- A full memo learns no new structure. A graph of a new structure is not searched and keeps its original order, so
  all graphs of that structure agree.
- A later graph that does not accept the remembered order is a conflict. The nodes stay as they are and the
  backend stops: the hook calls `tsr_pivot_stop_on_conflict(report)` after `tsr_pivot_optimize`, which does nothing
  without a conflict and otherwise aborts the process (GGML_ABORT) with a message that names the structure, the
  validator's reason and `TSR_PIVOT=0`. The buffers of the structure are planned for the remembered order, so neither
  that order nor the original one is safe for this graph. A conflict needs an analysis that depends on the batch,
  which §7.2 excludes. One more case exists: a later graph that computes nodes the first graph did not compute can
  carry constraints that the remembered order breaks.

---

## 7. What must hold

### 7.1 Validity

A new order is used only when it passes the node-level validator (§6.5). The core builds valid orders by
construction (§5.2) and the validator does not trust it: nothing in ggml checks the order that a backend
returns, so the adapter does.

### 7.2 Independence of the token count

llama.cpp reserves the compute buffer with a worst-case graph. ggml-alloc keeps the allocation of the graph it
last allocated by node index and replays it for a later graph of the same node count and shapes
(`ggml_gallocr_needs_realloc`). A graph of another ubatch size that got another permutation of its nodes would
find its tensors at the wrong places in the replayed allocation.

So the same structure must give the same permutation for every token count. The units, their classes, their
preds and every cost depend on the structure of the graph and the device only. Three consequences:

- Costs never use the token count (PV-8).
- A node is a compute node whether or not it has rows (§6.1).
- `defer` does not depend on the tile count of the product (§6.4).

A graph with many tokens therefore gets the order chosen for the same structure at one token. Its products of
more than one T-tile are not split, so the order gives them no overlap, and the model does not see that. This is
a limit of the design, not of the search.

The pass does not rest on the costs alone. §6.6 makes "the same structure, the same permutation" hold by
construction. The costs that do not depend on the batch keep the order of the first graph a good one for every
batch, and they keep the conflict rule of §6.6 from firing.

### 7.3 The outputs do not change

The records that run are the same on the same operands. Pivot moves nodes; it does not change a node, a source
or a value. The builder derives BEGIN, END and JOIN records, and the hazards between records, from the final
order and the byte ranges, so the lists differ only in the order of independent records and in which products
are split. The device output of a graph is bit-identical with and without the pass (SW §9.3, SW-94).

### 7.4 The compute buffer

Interleaving independent work extends the lifetimes of its results, so the compute buffer that ggml-alloc sizes
for the reordered graph may be larger than for the original order. The buffer is sized from the reordered
reserve graph, so its size follows the order. The reordered graph must still be placed on the device.

### 7.5 The plan cache

The plan cache key `tsr_graph_key` hashes the nodes in their final order with their addresses (SW §9.3). The
permutation is a function of the structure, so the identity of a graph is stable from one call to the next.

---

## 8. The knob, the log and the tests

### 8.1 Knob

`TSR_PIVOT` is read once when a device opens. `0` turns the pass off; any other value, or none, leaves it on.
The pass also does nothing when `TSR_MM_DEFER` is `0`, or when the booted image (narrowed by `TSR_KINDS`) lacks
MUL_MAT_BEGIN or MUL_MAT_END: without the split a reorder gains nothing in the model. The parameters are the
defaults of §5.1. The device fixes two inputs at open: the number of DDR channels and the dual weight stream.

### 8.2 Log

With `TSR_DEBUG=1` the backend prints at open whether the pass is on, and for each graph the pass ran on:

- one line per accepted move: the unit's name (the name of its first compute node, `(+n)` for the other compute
  nodes of the unit), the positions `from` and `to`, the number of moved units and the gain;
- one line with the node count, the structure key (16 hex digits), the unit count, the moves and passes, the
  model's cycles before and after, and the time of the call; it ends with a note when the validator refused the new
  order. It is the line of a graph that was searched, which is the first graph of its structure;
- for a graph that got its order from the memo, one line with the node count, the structure key and "the remembered
  order", followed by "(no change)" when the graph keeps the order it came in (the first graph of the structure kept
  its order, or the pass has already permuted the graph), and the time of the call;
- for a graph of a new structure when the memo is full, one line with the node count and the structure key that
  ends with "not searched, the memo is full";
- for the first graph of a structure that has no compute node, one line with the node count and the structure key that
  ends with "nothing to schedule".

### 8.3 Tests

| Test | What it checks |
|---|---|
| `ggml_tesseract_pivot` (`tests/test-tsr-pivot.cpp`; core, no ggml) | the model's arithmetic for each class and rule, with the sums in comments; the validator's refusals; hand-built graphs with their expected orders (a BLOCK unit with an independent chain, sibling products apart, a later and longer window, nothing to gain, order constraints that forbid the best move, the threshold, `max_moved` and `window`); random DAGs: a valid result, never slower, deterministic, a fixed point, every recorded move replayed by an independent hoist with its gain equal to the fall of the model's time, and every incremental score equal to a full evaluation of the same units |
| `ggml_tesseract_pivot_graph` (`tests/test-tsr-pivot-graph.cpp`; adapter on shape-only graphs in llama.cpp's node order with Gemma 4 26B-A4B and Qwen3.8 27B layer shapes) | units, classes and costs against hand arithmetic; every group that the builder fuses lies inside one unit; no-op joining, the span of a split group, in-place constraints; the moves (the dense gate and up products around the router and the experts, K and V next to Q, the z product before GATED_DELTA_NET); the node validator refuses a consumer before its producer, a FLASH_ATTN before its SET_ROWS, a node inside the router chain, a CPY behind its SSM_CONV, and restores the original order; the same permutation, unit count and costs for 1, 2, 8 and 64 tokens and with or without rows in the state zeroing; the gather cost across batch layouts; the structure key (equal for 1, 2, 8 and 64 tokens, with and without rows in the state zeroing, and across the batch layouts of a Gated DeltaNet state group; different for another node count, another weight shape or type, another op, another unary op, another link, another use count, a view that starts later, another view base, which outside tensors are linked, and another cast type; the same key at other addresses, and for graphs that differ in the COMPUTE flag alone); the memo (a graph of another batch gets the order of the first without a search; a permuted graph stays; a first graph that kept its order gives every later graph its order; an order that is not the search's is given on unchanged; a conflict for a remembered order with consumers before producers, a missing node, or a first MUL_MAT_ID that does not stay first, with the nodes left as they were; a full memo of 256 entries does not grow and does not search; a full memo still serves the structures it knows; a graph with a node that is not computed gets the first graph's order; a conflict stops the process, which a child process shows by ending with SIGABRT and the message; another device or other search parameters search again); the pinned first MUL_MAT_ID; the fixed point |
| `ggml_tesseract_pivot_*` (`m7_tests.py pivot`; tiny dense, Gemma 4 MoE, Qwen3.5 and Qwen3 MoE models on the simulator) | the device output with the pass and with `TSR_PIVOT=0`: bit-identical logits for the prompt, decode tokens and a replay in verification windows; the same launches and the same records but for BEGIN and END; the knob lines; the minimum number of moves; every structure has one search line (a "nothing to schedule" line counts as one), in front of its remembered-order lines, so every later graph of a structure gets the order of the first without a search; the graphs of one node count have one structure; at least one graph got its order from the memo; in the MoE model the dense gate product is a BEGIN before MOE_ROUTE and the ENDs come after the MUL_MAT_ID records; in the Qwen3.5 model the z product is a BEGIN before GATED_DELTA_NET with its END after it |

Every `ggml_tesseract_*` test passes with the pass on, which is the default, and with `TSR_PIVOT=0`.

---

## 9. How another front end uses the core

The core is a function from a graph of units to an order. A front end that has a dependency graph and a way to
apply an order can use it (a runtime without ggml, for example):

1. Number the units in the original order of the graph. Merge what must stay together into one unit: whatever
   one record lowers, and every set of nodes between which the lowering makes no hazard check.
2. Give every unit its preds: the data producers and every order constraint that the graph carries outside its
   edges (buffers written through views, state caches, in-place ops). A pred is always a smaller number.
3. Give every unit its class and costs (§4). Use the device's numbers; do not use the batch size when the
   front end must give the same order for every batch (§7.2).
4. Call `tsr_pivot_run(units, params)`. Check `ok`. Check the order with `tsr_pivot_valid`.
5. Apply `result.order`: the units in that order, each unit's members in their original order.
6. Check the result at the level of the front end's own objects (the adapter's node validator is the example)
   and keep the original order when the check fails.

```cpp
std::vector<tsr_pivot_unit> units = build_units(graph);          // the front end's mapping
tsr_pivot_result r = tsr_pivot_run(units);                       // r.ok, r.order, r.moves, r.cycles_*
```

The core has no ggml type, no device access and no floating point. The caller owns the unit names and the
mapping from units to its objects: a move carries unit numbers only. The core is plain C++17 and builds in
`tsr_lower`, which `sw/rt` also links.

---

## 10. Decisions

| Id | Decision | Reason |
|---|---|---|
| PV-1 | Pivot is a general scheduler over a dependency graph with a cost model of the device. No rule names a model or a layer type | The same code serves Gemma 4, Qwen3.5, Qwen3.8 and dense models, and a new graph shape needs a cost class at most, not a rule |
| PV-2 | The core has no ggml, no device and no floating point; the adapter owns the mapping from nodes to units | Another front end can use the core, the core is tested without llama.cpp, and the order is the same on every host |
| PV-3 | A coarse model with a search, not fixed priorities | Which product belongs under which idle window depends on the costs of both; a model ranks the alternatives and a rule would need a case for each layer shape |
| PV-4 | Only a unit with MVM work is hoisted for itself; GLUE and BLOCK units move only as ancestors | The gain is the use of an idle MVM unit; moving glue alone only shifts the idle time |
| PV-5 | The move is `hoist(u, p)`: u with its ancestors at or after p, put in front of p in their relative order | A closed set moved forward keeps every pred in front, so the result is a valid order without a check and a candidate costs one pass over the span |
| PV-6 | Window of 64 units, at most 8 units a move, a threshold of 256 cycles, a horizon of 32 units; the best candidate is confirmed on the whole graph before it is taken | The window and the move size bound the search; a gain under 256 cycles is within the error of a coarse model; the horizon keeps a trial short and the confirmation makes every accepted move a gain for the whole graph, so the search ends, is never slower in the model and has a fixed point |
| PV-7 | Every unit adds its issue cost before it starts, and a split product is charged `issue_open` + `issue_close` against `issue` | A unit cannot start before its record is issued, and a split that gains less than its 450 cycles of control-core time must not be taken |
| PV-8 | Costs come from structure and the device, never from the token count; `defer` is set for every weight MUL_MAT without a SCALE or softcap partner; a node without rows is a compute node | ggml-alloc replays the reserve graph's allocation by node index, so every batch needs the same permutation (§7.2) |
| PV-9 | Units are intervals of the original order; a group that the builder fuses or lowers without a hazard check is one unit, taken with the builder's own structural matchers; a split group becomes the span between its members | An edge between nodes is then an edge between units from a smaller to a larger number, and the builder and the adapter cannot disagree on a group |
| PV-10 | Order constraints: nodes that touch a base tensor that a compute node writes through a view or in place keep their order; the state CPYs of an SSM_CONV precede it; a first MUL_MAT_ID stays first | ggml orders these nodes through a buffer, not through a source; the builder's FA_NEW rule and its SSM_CONV group need them; llama.cpp's expert-copy callback reads the first node |
| PV-11 | A node-level validator runs after the permutation and the original order is restored when it fails | Nothing in ggml checks a backend's order, and a wrong order is a wrong output |
| PV-12 | A search that starts from the identity, visits positions in order, repeats passes until none moves, breaks ties toward the least movement and uses integer arithmetic only | The result is deterministic on every host, a second run changes nothing, and a tie goes to the least movement |
| PV-13 | The knob `TSR_PIVOT` (default on, read at open); the pass is off for `TSR_MM_DEFER=0` and for an image without MUL_MAT_BEGIN or MUL_MAT_END | Without the split a reorder gains nothing in the model; the knob measures the pass in the same build and takes it out of the way of a failure |
| PV-14 | The hook runs outside the device mutex and reads no tensor data of compute tensors | ggml calls it before allocation, before ggml-alloc has given the compute tensors their addresses |
| PV-15 | The constants are one table with a source for each; they are coarse and only rank the alternatives | A cost that is wrong by a few percent changes no order; a cost that is wrong in kind shows in the tests' hand arithmetic and in the profile report |
| PV-16 | One order per structure, remembered in the backend instance by a structure key; a later graph gets the first graph's order after a check, and a graph that refuses it stops the backend | ggml-alloc replays the plan by node index, so a different permutation of one structure corrupts silently; a property that the pass enforces does not depend on every cost staying independent of the batch |
