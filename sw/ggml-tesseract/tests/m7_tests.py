#!/usr/bin/env python3
"""M7 ctest drivers of sw/ggml-tesseract (docs/design/06-software-verification.md §8, §10.2, §10.4).

graph   one tiny-model graph test (ggml_tesseract_graph_qwen_tiny, _gemma_tiny, _qwen_tiny_q4k):
        1. klcheck generates on the device with the M7 settings (--phase m7: FA on, q8_0 / q8_0 KV
           on the device = offload_kqv, LLAMA_ATTN_ROT_DISABLE=1) under TSR_DEBUG=1 (one line per
           launch), GGML_SCHED_DEBUG=2 (the scheduler's split listing of every graph it splits),
           TSR_STATS=1 and klcheck -v (one marker line before and after each llama_decode);
        2. the log is checked: every single-token decode graph the scheduler split has exactly one
           Tesseract split (SW §8 R1, §10.4 item 1; a CPU split after it is reported, R8); every
           single-token decode made exactly one launch, every prefill decode one per ubatch
           (ceil(n_tokens / ub); prefill graphs with more device splits are reported);
           at least --min-decodes single-token decodes; no launch outside a decode; the launch
           lines agree with TSR_STATS; with --prefill any the prefill launches are only reported
           (ggml_tesseract_graph_qwen_tiny_ub1024: one 700-token ubatch, whose FLASH_ATTN of more than
           680 tokens stays on the CPU, SW-43), and --min-prefill-tokens requires such a ubatch;
        3. KL against the CPU backend with klcheck's criterion: CPU replay, the 1-ulp floors of the
           seeds 1 to --floor-seeds (default 3), the embd identity check (as `klcheck all`), then
           `klcheck compare` with --min-kl / --min-top1 (the criterion is mean KL <= max(min-kl,
           3 x the mean floor KL) and top-1 >= min-top1; docs/klcheck.md "Pass"; why the tiny
           models need --min-kl: tiny.cmake).
        The device run takes no TSR_KINDS from the caller: the tests check the full kind set.
defer   MUL_MAT deferral (SW-92, SW-94; ggml_tesseract_defer_*): klcheck generates the same tokens on the device twice, with the default
        (a plain one-tile MUL_MAT is lowered as MUL_MAT_BEGIN + MUL_MAT_END when the next record leaves its result alone) and
        with TSR_MM_DEFER=0 (every MUL_MAT is one record); the two runs must be bit-identical (`klcheck compare
        --expect-identical`: the same tokens, the same logits), launch for launch (the same launch count), the default run must
        have launched at least --min-pairs MUL_MAT_BEGIN / MUL_MAT_END pairs (TSR_STATS' records per kind) and the TSR_MM_DEFER=0
        run none.
pivot   Tesseract Pivot, the graph scheduler (docs/design/07-graph-scheduler.md; ggml_tesseract_pivot_*): klcheck generates the same tokens on
        the device twice, with the pass on (the default) and with TSR_PIVOT=0; the two runs must be bit-identical (the same tokens, the
        same logits) with the same launches, the same records but for the MUL_MAT_BEGIN / MUL_MAT_END split, a replay of the recorded
        tokens in verification windows (--verify-k, the batches of speculative decoding) must be bit-identical too, the pass must have
        made at least --min-moves moves (its TSR_DEBUG lines), every structure must have been searched once and every later graph of it
        must have got the remembered order, with one structure per node count (the permutation is a function of the graph's structure:
        ggml-alloc replays the reserve graph's allocation by node index) and the knob off none. --expect moe | gdn checks the decode launch of the
        pass-on run (TSR_DEBUG=2's record listing): moe: in every layer the dense ffn_gate product is a MUL_MAT_BEGIN in front of the
        MOE_ROUTE and its MUL_MAT_END comes after the layer's MUL_MAT_ID records, the up product's END too; gdn: in every layer with a
        GATED_DELTA_NET record the z product is a MUL_MAT_BEGIN in front of it with its MUL_MAT_END behind it.
ops     test-backend-ops on the device for one op filter: every supported case must pass, at least
        --min-run cases must be supported (so that a supports_op regression cannot pass as "all
        unsupported"), and the unsupported cases are counted and listed (the full list in --list).

Exit status: 0 pass, 1 fail, 2 usage or tool error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run(cmd: list[str], log: Path | None = None, env: dict | None = None, quiet: bool = False) -> int:
    """Run cmd; its stdout and stderr go to log (if given) and, unless quiet, to our stdout."""
    print("m7_tests: $ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    out = p.stdout.decode("utf-8", "replace")
    if log is not None:
        log.write_text(out)
    if not quiet:
        sys.stdout.write(out)
        sys.stdout.flush()
    return p.returncode


# ---------------------------------------------------------------------------------------------
# graph: the decode log

RE_BEGIN = re.compile(r"^klcheck: decode begin seq (\d+) tokens \[(\d+), (\d+)\) n_tokens (\d+)")
RE_END = re.compile(r"^klcheck: decode end")
RE_SPLIT = re.compile(r"^## SPLIT #(\d+): (\S+) # (\d+) inputs(?:: (.*))?")
RE_INPUT = re.compile(r"\[(.*?) \(\s*[^)]*\)\]")
# ggml-backend.cpp ggml_backend_sched_print_assignments (GGML_SCHED_DEBUG=2):
#   "node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:"
RE_NODE = re.compile(r"^node #\s*(\d+) \((.{10})\): (.{20}) \((.{5})\) \[(.{5}) (.{8})\] use=(\d+),c=(\d)")
RE_STATS = re.compile(r"^\[tesseract\] stats: graph_compute (\d+) calls.*?launches (\d+) .*?([\d.]+) per scheduler pass")
# SW-82: GATED_DELTA_NET records that ran on their state cache cell (no gather, no copy) of all those launched
RE_GDN_CELLS = re.compile(r"^\[tesseract\] stats: GATED_DELTA_NET records on their state cache cell.*?: (\d+) of (\d+)")


class Listing:
    """One scheduler split listing (one graph the scheduler split)."""

    def __init__(self, where: str):
        self.where = where          # "decode N" or "outside a decode (reserve)"
        self.splits: list[tuple[int, str, int]] = []
        self.inputs: list[list[str]] = []                         # the scheduler's input copies of each split
        self.nodes: list[tuple[int, str, str, str, int]] = []   # (i, op, name, backend, compute flag)
        self.node_split: list[int] = []                           # split index of each node

    def device_splits(self) -> list[int]:
        return [k for k, (_, be, _) in enumerate(self.splits) if not be.startswith("CPU")]

    def problems(self) -> list[str]:
        dev = self.device_splits()
        return [] if len(dev) == 1 else [f"{len(dev)} device splits (want exactly 1)"]

    def trailing_cpu(self) -> bool:
        """A CPU split after the device split: allowed (SW §8 R8; no launch), reported."""
        dev = self.device_splits()
        return len(dev) == 1 and dev[0] < len(self.splits) - 1

    def describe(self, max_nodes: int = 40) -> str:
        lines = ["    splits: " + ", ".join(f"#{i} {be} ({n} inputs)" for i, be, n in self.splits)]
        dev = self.device_splits()
        for k in dev[1:]:       # what started each further device split: its inputs (a weight there = a
            i, be, n = self.splits[k]   # weight the device does not hold, ggml-backend.cpp pass 5)
            lines.append(f"    split #{i} ({be}) inputs: {', '.join(self.inputs[k]) or '-'}")
        dev = self.device_splits()
        first_dev = dev[0] if dev else len(self.splits)
        cpu_late = [(n, s) for n, s in zip(self.nodes, self.node_split)
                    if n[4] == 1 and n[3].strip() == "CPU" and s > first_dev]
        cpu_all = [n for n in self.nodes if n[4] == 1 and n[3].strip() == "CPU"]
        lines.append(f"    computed nodes on the CPU: {len(cpu_all)}"
                     + (f", {len(cpu_late)} after the first device split:" if cpu_late else ""))
        for (i, op, name, be, c), s in cpu_late[:max_nodes]:
            lines.append(f"      node {i:4d} split #{s} {op.strip():>10s} {name.strip()}")
        if len(cpu_late) > max_nodes:
            lines.append(f"      ... {len(cpu_late) - max_nodes} more")
        return "\n".join(lines)


class Decode:
    def __init__(self, idx: int, seq: int, a: int, b: int, n: int):
        self.idx, self.seq, self.a, self.b, self.n = idx, seq, a, b, n
        self.launches = 0
        self.listings: list[Listing] = []


def parse_log(text: str, launch_re: re.Pattern) -> dict:
    decodes: list[Decode] = []
    outside_listings: list[Listing] = []
    cur: Decode | None = None
    listing: Listing | None = None
    launches_outside = 0
    launch_lines = 0
    stats = None
    gdn_cells = None
    for raw in text.splitlines():
        line = ANSI.sub("", raw)
        m = RE_BEGIN.match(line)
        if m:
            cur = Decode(len(decodes), *(int(g) for g in m.groups()))
            decodes.append(cur)
            listing = None
            continue
        if RE_END.match(line):
            cur = None
            listing = None
            continue
        if launch_re.search(line):
            launch_lines += 1
            if cur is not None:
                cur.launches += 1
            else:
                launches_outside += 1
            continue
        m = RE_SPLIT.match(line)
        if m:
            k = int(m.group(1))
            if k == 0 or listing is None:
                listing = Listing(f"decode {cur.idx}" if cur is not None else "outside a decode (reserve)")
                (cur.listings if cur is not None else outside_listings).append(listing)
            listing.splits.append((k, m.group(2), int(m.group(3))))
            listing.inputs.append(RE_INPUT.findall(m.group(4) or ""))
            continue
        m = RE_NODE.match(line)
        if m and listing is not None:
            listing.nodes.append((int(m.group(1)), m.group(2), m.group(3), m.group(5), int(m.group(8))))
            listing.node_split.append(len(listing.splits) - 1)
            continue
        m = RE_STATS.match(line)
        if m:
            stats = {"graph_compute": int(m.group(1)), "launches": int(m.group(2)), "per_pass": float(m.group(3))}
        m = RE_GDN_CELLS.match(line)
        if m:
            gdn_cells = (int(m.group(1)), int(m.group(2)))
    return {"decodes": decodes, "outside": outside_listings, "launches_outside": launches_outside,
            "launch_lines": launch_lines, "stats": stats, "gdn_cells": gdn_cells}


def check_graph_log(p: dict, ub: int, min_decodes: int, prefill: str = "ubatch",
                    min_prefill_tokens: int = 0, gdn_cells: str = "") -> tuple[list[str], dict]:
    fails: list[str] = []
    decodes: list[Decode] = p["decodes"]
    tok = [d for d in decodes if d.n == 1]
    pre = [d for d in decodes if d.n > 1]
    bad_tok = [d for d in tok if d.launches != 1]
    bad_pre = [d for d in pre if d.launches != math.ceil(d.n / ub)] if prefill == "ubatch" else []
    listings = [(d, l) for d in decodes for l in d.listings]
    tok_listings = [(d, l) for d, l in listings if d.n == 1]
    # the split rule is the decode graph's (SW §8, §10.4 item 1); a prefill graph is held to one launch
    # per ubatch (above), its extra device splits are reported (e.g. a ubatch without outputs, whose
    # emptied last-layer nodes run nowhere)
    bad_listings = [(d, l, l.problems()) for d, l in tok_listings if l.problems()]
    pre_multi = [(d, l, l.problems()) for d, l in listings if d.n > 1 and l.problems()]
    summary = {
        "decodes": len(decodes), "token_decodes": len(tok), "prefill_decodes": len(pre),
        "token_launches": sorted({d.launches for d in tok}),
        "prefill": [(d.n, math.ceil(d.n / ub), d.launches) for d in pre],
        "listings_in_decodes": len(listings), "listings_in_token_decodes": len(tok_listings),
        "prefill_listings_multi_split": len(pre_multi),
        "launch_lines": p["launch_lines"], "launches_outside": p["launches_outside"], "stats": p["stats"],
    }
    if not decodes:
        fails.append("no decode markers in the log (klcheck -v prints them; an older klcheck?)")
    if p["launch_lines"] == 0:
        fails.append("no launch lines in the log (TSR_DEBUG=1 prints '[tesseract] launch #N'; format changed? see --launch-re)")
    if len(tok) < min_decodes:
        fails.append(f"{len(tok)} single-token decodes, want >= {min_decodes}")
    if min_prefill_tokens and not any(min_prefill_tokens <= d.n <= ub for d in pre):
        fails.append(f"no prefill decode of {min_prefill_tokens}..{ub} tokens (one ubatch); prefills: {sorted(d.n for d in pre)}")
    if bad_tok:
        fails.append(f"{len(bad_tok)}/{len(tok)} single-token decodes did not make exactly one launch "
                     f"(launch counts {sorted({d.launches for d in bad_tok})}; first: decode {bad_tok[0].idx}, "
                     f"seq {bad_tok[0].seq} token {bad_tok[0].a})")
    if bad_pre:
        d = bad_pre[0]
        fails.append(f"{len(bad_pre)}/{len(pre)} prefill decodes did not make one launch per ubatch "
                     f"(first: decode {d.idx}, {d.n} tokens, {math.ceil(d.n / ub)} ubatches of <= {ub}, {d.launches} launches)")
    if p["launches_outside"]:
        fails.append(f"{p['launches_outside']} launches outside a llama_decode")
    if not tok_listings:
        fails.append("no scheduler listing of a single-token decode graph (GGML_SCHED_DEBUG=2 output missing)")
    trailing = [(d, l) for d, l in listings if l.trailing_cpu()]
    if trailing:
        d, l = trailing[0]
        print(f"m7_tests: note: {len(trailing)}/{len(listings)} decode graphs end with a CPU split (allowed, SW §8 R8); "
              f"first: decode {d.idx}\n{l.describe()}")
    if pre_multi:
        d, l, why = pre_multi[0]
        print(f"m7_tests: note: {len(pre_multi)} prefill graphs have more than one device split (their launches are "
              f"checked above); first: decode {d.idx} ({d.n} tokens): {'; '.join(why)}\n{l.describe()}")
    if bad_listings:
        d, l, why = bad_listings[0]
        fails.append(f"{len(bad_listings)}/{len(tok_listings)} single-token decode graphs do not have exactly one Tesseract "
                     f"split; first: decode {d.idx}: {'; '.join(why)}\n{l.describe()}")
    if gdn_cells:
        # SW-82: with the pinned graph every sequence keeps its cache cell, so every GATED_DELTA_NET record of the run reads
        # and writes its cell ("all"); with TSR_GDN_CELL=0 none does ("none")
        gc = p["gdn_cells"]
        if gc is None or gc[1] == 0:
            fails.append("no GATED_DELTA_NET records in the TSR_STATS summary (the model has none, or the line is missing)")
        elif gdn_cells == "all" and gc[0] != gc[1]:
            fails.append(f"{gc[0]} of {gc[1]} GATED_DELTA_NET records ran on their state cache cell, want all")
        elif gdn_cells == "none" and gc[0] != 0:
            fails.append(f"{gc[0]} of {gc[1]} GATED_DELTA_NET records ran on their state cache cell with TSR_GDN_CELL=0")
    st = p["stats"]
    if st is None:
        print("m7_tests: note: no TSR_STATS summary line in the log")
    elif st["launches"] != p["launch_lines"]:
        fails.append(f"TSR_STATS counts {st['launches']} launches, the log {p['launch_lines']} launch lines")
    return fails, summary


def cmd_graph(a: argparse.Namespace) -> int:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.klc"):
        f.unlink()
    kl = a.klcheck
    common = ["-m", a.model, "--phase", "m7", "-ub", str(a.ub), "-t", str(a.threads)]
    env = dict(os.environ)
    env.update({"TSR_DEBUG": "1", "GGML_SCHED_DEBUG": "2", "TSR_STATS": "1"})
    # klcheck run --phase m7 keeps the caller's TSR_KINDS; these tests check the full kind set (a list
    # without SET_ROWS_KV aborts inside llama.cpp with the KV cache on the device)
    env.pop("TSR_KINDS", None)
    dev, cpu, embd = (str(out / f) for f in ("dev.klc", "cpu.klc", "cpu-embd.klc"))
    floors = [str(out / f"floor-seed{i}.klc") for i in range(1, a.floor_seeds + 1)]
    log = out / "dev.log"
    prompts = ["--prompts", a.prompts] if a.prompts else []          # none: klcheck's built-in prompts
    if a.join_prompts:
        # one long prompt: the file's first N prompts joined by spaces (a prefill ubatch of several hundred
        # tokens)
        lines = [l.strip() for l in Path(a.prompts).read_text().splitlines() if l.strip() and not l.startswith("#")]
        joined = out / "joined-prompt.txt"
        joined.write_text(" ".join(lines[:a.join_prompts]) + "\n")
        prompts = ["--prompts", str(joined)]
    rc = run([kl, "run", "--device", a.device] + prompts + ["--n-prompts", str(a.n_prompts),
              "-n", str(a.n_gen), "--prompt-tail", str(a.prompt_tail), "-v", "--out", dev] + common,
             log=log, env=env, quiet=True)
    text = log.read_text(errors="replace")
    if rc != 0:
        # the cause (an abort's "file.cpp:N: message", a die(), an assert), not the debugger's backtrace
        lines = [l for l in text.splitlines() if not re.match(r"^(#\d+ |\[New LWP|\[Thread |Using host|0x[0-9a-f]+ in |warning: \d+\s)", l)]
        cause = [l for l in lines if re.search(r"\.(c|cc|cpp|h|hpp):\d+: |GGML_ASSERT|error|ABORT|tesseract: ", l)]
        print("\n".join(lines[-12:] if not cause else cause[:8]))
        print(f"m7_tests: FAIL: the device run exited with {rc} (full log: {log})")
        return 1
    for line in text.splitlines():                     # klcheck's own summary lines
        if line.startswith("klcheck [") or line.startswith("    output:") or line.startswith("[tesseract] stats:"):
            print(line)

    fails, s = check_graph_log(parse_log(text, re.compile(a.launch_re)), a.ub, a.min_decodes, a.prefill,
                               a.min_prefill_tokens, a.gdn_cells)
    print(f"m7_tests: decodes: {s['token_decodes']} single-token (launches per decode: {s['token_launches']}), "
          f"{s['prefill_decodes']} prefill [(tokens, ubatches, launches)]: {s['prefill']}")
    print(f"m7_tests: scheduler listings inside decodes: {s['listings_in_decodes']} "
          f"({s['listings_in_token_decodes']} single-token); launch lines {s['launch_lines']}, "
          f"outside a decode {s['launches_outside']}; TSR_STATS {s['stats']}")
    structure_ok = not fails
    for f in fails:
        print(f"m7_tests: FAIL: {f}")
    (out / "graph.json").write_text(json.dumps({"summary": s, "failures": fails}, indent=1, default=str))

    # KL against the CPU backend (klcheck all's phases 2-5 on the device tokens)
    rc_cpu = run([kl, "run", "--device", "cpu", "--tokens", dev, "--out", cpu] + common, quiet=True)
    rc_floor = 0
    for i, fl in enumerate(floors, 1):
        rc_floor |= run([kl, "run", "--device", "cpu", "--tokens", dev, "--perturb-ulp", str(i), "--out", fl] + common,
                        quiet=True)
    rc_embd = run([kl, "run", "--device", "cpu", "--tokens", dev, "--input", "embd", "--out", embd] + common, quiet=True)
    if rc_cpu or rc_floor or rc_embd:
        print(f"m7_tests: FAIL: a CPU run failed (cpu {rc_cpu}, floor {rc_floor}, embd {rc_embd})")
        return 1
    rc_id = run([kl, "compare", "--ref", cpu, "--dev", embd, "--expect-identical", "-q"])
    if rc_id != 0:
        print("m7_tests: FAIL: the embd input path is not the token path bit for bit (the floor would not measure the nudge)")
    cmp = [kl, "compare", "--ref", cpu, "--dev", dev, "--json", str(out / "report.json"),
           "--min-kl", str(a.min_kl), "--min-top1", str(a.min_top1)]
    for fl in floors:
        cmp += ["--floor", fl]
    rc_cmp = run(cmp)
    ok = structure_ok and rc_id == 0 and rc_cmp == 0
    print(f"m7_tests: {'PASS' if ok else 'FAIL'}: structure {'ok' if structure_ok else 'FAIL'}, "
          f"embd identity {'ok' if rc_id == 0 else 'FAIL'}, KL {'ok' if rc_cmp == 0 else 'FAIL'} (files in {out})")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------
# defer: the device output with and without TSR_MM_DEFER

RE_KINDS = re.compile(r"^\[tesseract\] stats: records per kind: (.*)$")
RE_LAUNCH_PAIRS = re.compile(r"^\[tesseract\] launch #\d+: .*; MUL_MAT_BEGIN (\d+), MUL_MAT_END (\d+)$")


def kind_counts(text: str) -> dict:
    """TSR_STATS' 'records per kind' line as {kind name: count}."""
    for raw in text.splitlines():
        m = RE_KINDS.match(ANSI.sub("", raw))
        if m:
            return {k: int(v) for k, v in (x.strip().rsplit(" ", 1) for x in m.group(1).split(",") if " " in x.strip())}
    return {}


def cmd_defer(a: argparse.Namespace) -> int:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kl = a.klcheck
    common = ["-m", a.model, "--phase", "m7", "-ub", str(a.ub), "-t", str(a.threads)]
    prompts = ["--prompts", a.prompts] if a.prompts else []
    runs = {}
    for name, knob in (("on", None), ("off", "0")):
        env = dict(os.environ)
        env["TSR_STATS"] = "1"
        env["TSR_DEBUG"] = "1"                    # the launch lines: BEGIN and END counts of every list
        env.pop("TSR_KINDS", None)
        env.pop("TSR_MM_DEFER", None)
        if knob is not None:
            env["TSR_MM_DEFER"] = knob
        klc, log = out / f"{name}.klc", out / f"{name}.log"
        rc = run([kl, "run", "--device", a.device] + prompts + ["--n-prompts", str(a.n_prompts), "-n", str(a.n_gen),
                  "--prompt-tail", str(a.prompt_tail), "--out", str(klc)] + common, log=log, env=env, quiet=True)
        if rc != 0:
            print("\n".join(log.read_text(errors="replace").splitlines()[-12:]))
            print(f"m7_tests: FAIL: the device run (TSR_MM_DEFER {knob or 'default'}) exited with {rc} (log: {log})")
            return 1
        text = log.read_text(errors="replace")
        st = RE_STATS.search(next((ANSI.sub("", l) for l in text.splitlines() if RE_STATS.match(ANSI.sub("", l))), ""))
        lines = [ANSI.sub("", l) for l in text.splitlines()]
        ends = [tuple(int(x) for x in m.groups()) for m in (RE_LAUNCH_PAIRS.search(l) for l in lines) if m]
        runs[name] = {"kinds": kind_counts(text), "launches": int(st.group(2)) if st else -1, "lists": ends}
    fails = []
    on, off = runs["on"], runs["off"]
    pairs = on["kinds"].get("MUL_MAT_BEGIN", 0)
    if pairs < a.min_pairs or pairs != on["kinds"].get("MUL_MAT_END", -1):
        fails.append(f"default run: {pairs} MUL_MAT_BEGIN and {on['kinds'].get('MUL_MAT_END', 0)} MUL_MAT_END records (want equal, at least {a.min_pairs})")
    # a BEGIN and its END are in one list: every launch has as many ENDs as BEGINs, whatever cut its list (SW §9.3)
    unpaired = [i for i, (b, e) in enumerate(on["lists"]) if b != e]
    if unpaired or len(on["lists"]) != on["launches"]:
        fails.append(f"launches with a BEGIN count other than their END count: {unpaired[:5]} (launch lines {len(on['lists'])} of {on['launches']})")
    if off["kinds"].get("MUL_MAT_BEGIN", 0) or off["kinds"].get("MUL_MAT_END", 0):
        fails.append(f"TSR_MM_DEFER=0 run launched BEGIN / END records: {off['kinds']}")
    if on["launches"] <= 0 or (on["launches"] != off["launches"] and not a.split):
        fails.append(f"launches differ: {on['launches']} with the knob on, {off['launches']} with it off")
    if a.split and on["launches"] <= a.n_prompts * (a.n_gen - 1):
        fails.append(f"TSR_OPL_SPLIT cut no list: {on['launches']} launches")
    n_on = sum(v for k, v in on["kinds"].items() if k not in ("MUL_MAT_BEGIN", "MUL_MAT_END", "MUL_MAT"))
    n_off = sum(v for k, v in off["kinds"].items() if k not in ("MUL_MAT_BEGIN", "MUL_MAT_END", "MUL_MAT"))
    mm_on = on["kinds"].get("MUL_MAT", 0) + pairs
    if n_on != n_off or mm_on != off["kinds"].get("MUL_MAT", 0):
        fails.append(f"the matrix products or the other records differ: {mm_on} / {n_on} with the knob, {off['kinds'].get('MUL_MAT', 0)} / {n_off} without")
    rc = run([kl, "compare", "--ref", str(out / "off.klc"), "--dev", str(out / "on.klc"), "--expect-identical", "-q"])
    if rc != 0:
        fails.append("the logits with TSR_MM_DEFER on differ from those with it off")
    for f in fails:
        print(f"m7_tests: FAIL: {f}")
    print(f"m7_tests: {'FAIL' if fails else 'PASS'}: {pairs} BEGIN / END pairs in {on['launches']} launches, logits "
          f"{'identical' if rc == 0 else 'DIFFER'} with and without TSR_MM_DEFER (files in {out})")
    return 1 if fails else 0


# ---------------------------------------------------------------------------------------------
# pivot: the device output with and without Tesseract Pivot

RE_REC = re.compile(r"^\[tesseract\]\s+(\d+) (\S+)\s+deps\s+(-?\d+)\s+(-?\d+)((?: [0-9a-f]{8})*)(?:\s+(.*))?$")
RE_LAUNCH = re.compile(r"^\[tesseract\] launch #(\d+): (\d+) records")
RE_PIVOT_MOVE = re.compile(r"^\[tesseract\] pivot move: ")
RE_PIVOT_GRAPH = re.compile(r"^\[tesseract\] pivot: graph \S+: (\d+) nodes, structure ([0-9a-f]{16}), (\d+) units, (\d+) moves in (\d+) passes, "
                            r"model (\d+) -> (\d+) cycles")
RE_PIVOT_HIT = re.compile(r"^\[tesseract\] pivot: graph \S+: (\d+) nodes, structure ([0-9a-f]{16}): the remembered order")
RE_PIVOT_NONE = re.compile(r"^\[tesseract\] pivot: graph \S+: (\d+) nodes, structure ([0-9a-f]{16}): nothing to schedule")


def launches_of(text: str) -> list[list[tuple[int, str, str]]]:
    """The record listings of a TSR_DEBUG=2 log: per launch the (index, kind, name) of its records."""
    out: list[list[tuple[int, str, str]]] = []
    cur = None
    for raw in text.splitlines():
        line = ANSI.sub("", raw)
        if RE_LAUNCH.match(line):
            cur = []
            out.append(cur)
            continue
        m = RE_REC.match(line)
        if m and cur is not None:
            cur.append((int(m.group(1)), m.group(2), (m.group(6) or "").strip()))
        elif cur is not None and not line.startswith("[tesseract]   pre"):
            cur = None
    return out


def check_plan(recs: list[tuple[int, str, str]], expect: str) -> list[str]:
    """The pivot's mark on a decode launch (a launch of the pass-on run): where the dense products and the z products are split."""
    fails = []

    def named(kind: str | None, name: str) -> list[int]:
        return [i for i, k, n in recs if (kind is None or k == kind) and re.search(r"(^| )" + re.escape(name) + r"$", n)]

    def region(il: int) -> tuple[int, int]:
        # a layer's records lie between the record that writes the previous layer's output (named l_out-N, alone or fused) and its own
        start = named(None, f"l_out-{il - 1}")
        end = named(None, f"l_out-{il}")
        return (start[0] if start else -1, end[0] if end else len(recs))

    layers = sorted({int(m.group(1)) for _, k, n in recs for m in [re.search(r"ffn_gate-(\d+)$", n)] if m and k == "MUL_MAT_BEGIN"}) \
        if expect == "moe" else sorted({int(m.group(1)) for _, k, n in recs for m in [re.search(r"(?:^| )z-(\d+)$", n)] if m})
    if not layers:
        return [f"no {'dense ffn_gate' if expect == 'moe' else 'z'} product is a MUL_MAT_BEGIN in the decode launch"]
    for il in layers:
        lo, hi = region(il)
        if expect == "moe":
            route = [i for i, k, n in recs if k == "MOE_ROUTE" and lo < i < hi]
            mmid = [i for i, k, n in recs if k == "MUL_MAT_ID" and lo < i < hi]
            gate_b, gate_e = named("MUL_MAT_BEGIN", f"ffn_gate-{il}"), named("MUL_MAT_END", f"ffn_gate-{il}")
            up_b, up_e = named("MUL_MAT_BEGIN", f"ffn_up-{il}"), named("MUL_MAT_END", f"ffn_up-{il}")
            if not (route and mmid and gate_b and gate_e and up_b and up_e):
                fails.append(f"layer {il}: records missing (MOE_ROUTE {route}, MUL_MAT_ID {mmid}, gate {gate_b}/{gate_e}, up {up_b}/{up_e})")
            elif not (gate_b[0] < route[0] and gate_e[0] > max(mmid) and up_e[0] > max(mmid)):
                fails.append(f"layer {il}: ffn_gate BEGIN {gate_b[0]} / END {gate_e[0]}, ffn_up BEGIN {up_b[0]} / END {up_e[0]}, "
                             f"MOE_ROUTE {route[0]}, MUL_MAT_ID up to {max(mmid)}: the gate's BEGIN has to be in front of MOE_ROUTE and the ENDs behind the experts")
        else:
            gdn = [i for i, k, n in recs if k == "GATED_DELTA_NET" and lo < i < hi]
            zb, ze = named("MUL_MAT_BEGIN", f"z-{il}"), named("MUL_MAT_END", f"z-{il}")
            if not (gdn and zb and ze):
                fails.append(f"layer {il}: records missing (GATED_DELTA_NET {gdn}, z {zb}/{ze})")
            elif not (zb[0] < min(gdn) and ze[0] > max(gdn)):
                fails.append(f"layer {il}: the z product's BEGIN {zb[0]} / END {ze[0]} does not enclose GATED_DELTA_NET {gdn}")
    return fails


def cmd_pivot(a: argparse.Namespace) -> int:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kl = a.klcheck
    common = ["-m", a.model, "--phase", "m7", "-ub", str(a.ub), "-t", str(a.threads)]
    prompts = ["--prompts", a.prompts] if a.prompts else []
    runs = {}
    for name, knob in (("on", None), ("off", "0")):
        env = dict(os.environ)
        env["TSR_STATS"] = "1"
        env["TSR_DEBUG"] = "2" if (name == "on" and a.expect) else "1"
        env.pop("TSR_KINDS", None)
        env.pop("TSR_MM_DEFER", None)
        env.pop("TSR_PIVOT", None)
        if knob is not None:
            env["TSR_PIVOT"] = knob
        klc, log = out / f"{name}.klc", out / f"{name}.log"
        rc = run([kl, "run", "--device", a.device] + prompts + ["--n-prompts", str(a.n_prompts), "-n", str(a.n_gen),
                  "--prompt-tail", str(a.prompt_tail), "--out", str(klc)] + common, log=log, env=env, quiet=True)
        if rc != 0:
            print("\n".join(log.read_text(errors="replace").splitlines()[-12:]))
            print(f"m7_tests: FAIL: the device run (TSR_PIVOT {knob or 'default'}) exited with {rc} (log: {log})")
            return 1
        text = log.read_text(errors="replace")
        lines = [ANSI.sub("", l) for l in text.splitlines()]
        st = next((RE_STATS.search(l) for l in lines if RE_STATS.match(l)), None)
        # one order per structure: ggml-alloc replays the allocation of the reserve graph by node index, so every graph of one structure
        # must get the same permutation, whatever its tokens and its batch. The pass searches the first graph of a structure and gives
        # every later one that order from its memo (klcheck has one context, so one memo): each structure has one search line (or a
        # "nothing to schedule" line), in front of its "remembered order" lines, and the graphs of one node count have one structure
        searched: set[str] = set()
        keys_of: dict[int, set] = {}
        bad: list[str] = []
        hits = 0
        for l in lines:
            m, h = RE_PIVOT_GRAPH.match(l) or RE_PIVOT_NONE.match(l), RE_PIVOT_HIT.match(l)
            if m:
                keys_of.setdefault(int(m.group(1)), set()).add(m.group(2))
                if m.group(2) in searched:
                    bad.append(f"structure {m.group(2)} ({m.group(1)} nodes) was searched twice")
                searched.add(m.group(2))
            elif h:
                keys_of.setdefault(int(h.group(1)), set()).add(h.group(2))
                hits += 1
                if h.group(2) not in searched:
                    bad.append(f"structure {h.group(2)} ({h.group(1)} nodes) got a remembered order with no search before it")
        bad += [f"graphs with {n} nodes have {len(v)} structures" for n, v in sorted(keys_of.items()) if len(v) > 1]
        runs[name] = {"kinds": kind_counts(text), "launches": int(st.group(2)) if st else -1, "text": text, "same_perm": "; ".join(bad),
                      "moves": sum(1 for l in lines if RE_PIVOT_MOVE.match(l)), "pivot_lines": len(searched) + hits, "hits": hits, "searched": len(searched),
                      "knob": next((l.rsplit(": ", 1)[-1] for l in lines if "Tesseract Pivot (graph_optimize):" in l), "?")}
    fails = []
    on, off = runs["on"], runs["off"]
    if on["knob"] != "on" or off["knob"] != "off":
        fails.append(f"the knob: the default run says '{on['knob']}', TSR_PIVOT=0 says '{off['knob']}'")
    if on["same_perm"]:
        fails.append(on["same_perm"])
    if on["moves"] < a.min_moves:
        fails.append(f"the pass made {on['moves']} moves in the default run (want at least {a.min_moves})")
    if on["hits"] < 1:
        fails.append("no graph of the default run got its order from the memo (the prompt and the decodes have one structure)")
    if off["moves"] or off["pivot_lines"]:
        fails.append(f"TSR_PIVOT=0 run: {off['moves']} moves, {off['pivot_lines']} pass lines")
    if on["launches"] <= 0 or on["launches"] != off["launches"]:
        fails.append(f"launches differ: {on['launches']} with the pass, {off['launches']} without")
    skip = ("MUL_MAT_BEGIN", "MUL_MAT_END", "MUL_MAT")
    n_on = sum(v for k, v in on["kinds"].items() if k not in skip)
    n_off = sum(v for k, v in off["kinds"].items() if k not in skip)
    mm_on = on["kinds"].get("MUL_MAT", 0) + on["kinds"].get("MUL_MAT_BEGIN", 0)
    mm_off = off["kinds"].get("MUL_MAT", 0) + off["kinds"].get("MUL_MAT_BEGIN", 0)
    if n_on != n_off or mm_on != mm_off or on["kinds"].get("MUL_MAT_BEGIN", 0) != on["kinds"].get("MUL_MAT_END", -1):
        fails.append(f"the records differ: {mm_on} matrix products / {n_on} other records with the pass, {mm_off} / {n_off} without")
    rc = run([kl, "compare", "--ref", str(out / "off.klc"), "--dev", str(out / "on.klc"), "--expect-identical", "-q"])
    if rc != 0:
        fails.append("the logits with the pass differ from those without it")
    # the batches of speculative decoding: the recorded tokens in verification windows of K + 1 tokens, with and without the pass
    vident = None
    if a.verify_k:
        for name, knob in (("on", None), ("off", "0")):
            env = dict(os.environ)
            env.pop("TSR_KINDS", None)
            env.pop("TSR_PIVOT", None)
            if knob is not None:
                env["TSR_PIVOT"] = knob
            rc = run([kl, "run", "--device", a.device, "--tokens", str(out / "off.klc"), "--verify-k", str(a.verify_k), "--out",
                      str(out / f"verify_{name}.klc")] + common, log=out / f"verify_{name}.log", env=env, quiet=True)
            if rc != 0:
                fails.append(f"the verification replay ({name}) exited with {rc} (log: {out / f'verify_{name}.log'})")
                break
        else:
            vident = run([kl, "compare", "--ref", str(out / "verify_off.klc"), "--dev", str(out / "verify_on.klc"), "--expect-identical", "-q"]) == 0
            if not vident:
                fails.append("the verification windows' logits with the pass differ from those without it")
    if a.expect:
        launches = launches_of(on["text"])
        # the last launch with a single token's records: a decode (the prompt's launch has the same records, but a different T)
        if not launches:
            fails.append("no record listing in the default run's log (TSR_DEBUG=2)")
        else:
            fails += check_plan(launches[-1], a.expect)
    for f in fails:
        print(f"m7_tests: FAIL: {f}")
    print(f"m7_tests: {'FAIL' if fails else 'PASS'}: {on['moves']} moves in {on['searched']} structures, {on['hits']} graphs with a remembered order, "
          f"BEGIN / END {on['kinds'].get('MUL_MAT_BEGIN', 0)} with the pass "
          f"and {off['kinds'].get('MUL_MAT_BEGIN', 0)} without, logits {'identical' if rc == 0 else 'DIFFER'}"
          f"{'' if vident is None else (', verification windows identical' if vident else ', verification windows DIFFER')} (files in {out})")
    return 1 if fails else 0


# ---------------------------------------------------------------------------------------------
# ops: test-backend-ops per op

RE_CASE = re.compile(r"^\s+([A-Z][A-Z0-9_]*)\((.*)\): (.*)$")


def cmd_ops(a: argparse.Namespace) -> int:
    cmd = [a.test_backend_ops, "test", "-b", a.backend, "-o", a.op]
    if a.params:
        cmd += ["-p", a.params]
    print("m7_tests: $ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = ANSI.sub("", p.stdout.decode("utf-8", "replace"))
    ok, fail, unsup, skipped, other = [], [], [], [], []
    for line in text.splitlines():
        m = RE_CASE.match(line)
        if not m:
            continue
        case, res = f"{m.group(1)}({m.group(2)})", m.group(3).strip()
        if res.startswith("not supported"):
            unsup.append(case)
        elif res.startswith("skipping"):
            skipped.append(case)
        elif res.endswith("OK"):
            ok.append(case)
        elif "FAIL" in res:
            fail.append(f"{case}: {res}")
        else:
            other.append(f"{case}: {res}")
    if a.list:
        Path(a.list).write_text("".join(f"{c}\n" for c in unsup))
    for f in fail + other:
        print(f"  FAIL {f}")
    shown = unsup[: a.max_list]
    if shown:
        print(f"m7_tests: {a.op}: not supported on {a.backend} ({len(unsup)} cases{', first ' + str(len(shown)) if len(unsup) > len(shown) else ''}):")
        for c in shown:
            print(f"  not supported: {c}")
    for line in text.splitlines():
        if "tests passed" in line or line.startswith("Backend ") or "Failed" in line:
            print(line)
    run_n = len(ok) + len(fail) + len(other)
    print(f"m7_tests: {a.op} on {a.backend}: {run_n} supported cases run ({len(ok)} OK, {len(fail) + len(other)} failed), "
          f"{len(unsup)} not supported, {len(skipped)} skipped (large); test-backend-ops exit {p.returncode}")
    fails = []
    if p.returncode != 0:
        # an abort inside a case: its message ("file.cpp:N: ...", GGML_ASSERT) and the cases around it
        lines = text.splitlines()
        hits = [k for k, l in enumerate(lines) if re.search(r"\.(c|cc|cpp|h|hpp):\d+: |GGML_ASSERT|Segmentation|Aborted", l)]
        for k in hits[:3]:
            for l in lines[max(0, k - 2): k + 1]:
                print(f"  | {l[:300]}")
        fails.append(f"test-backend-ops exited with {p.returncode}" + (" (an abort, above)" if hits else ""))
    if fail or other:
        fails.append(f"{len(fail) + len(other)} supported cases failed")
    if run_n < a.min_run:
        fails.append(f"{run_n} supported cases, want >= {a.min_run} (supports_op rejects the op?)")
    if run_n + len(unsup) + len(skipped) == 0:
        fails.append("no test case of the op ran (filter matches nothing?)")
    for f in fails:
        print(f"m7_tests: FAIL: {f}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("graph", help="tiny-model graph test")
    g.add_argument("--klcheck", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--prompts", default="", help="klcheck --prompts file (default: klcheck's built-in prompts)")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--device", default="sim")
    g.add_argument("--n-prompts", type=int, default=4)
    g.add_argument("-n", "--n-gen", type=int, default=17, help="tokens per prompt (n-1 single-token decodes)")
    g.add_argument("--prompt-tail", type=int, default=8)
    g.add_argument("-ub", "--ub", type=int, default=16)
    g.add_argument("-t", "--threads", type=int, default=4)
    g.add_argument("--min-decodes", type=int, default=16)
    g.add_argument("--min-kl", type=float, default=0.0)
    g.add_argument("--floor-seeds", type=int, default=3, help="floor runs with the seeds 1 to N (klcheck all's default)")
    g.add_argument("--min-top1", type=float, default=0.8)
    g.add_argument("--launch-re", default=r"^\[tesseract\] launch #\d+")
    g.add_argument("--gdn-cells", choices=("all", "none"), default="",
                   help="SW-82: require every (all) or no (none) GATED_DELTA_NET record to run on its state cache cell")
    g.add_argument("--join-prompts", type=int, default=0, metavar="N",
                   help="generate from one prompt: the --prompts file's first N prompts joined (with --n-prompts 1)")
    g.add_argument("--prefill", choices=("ubatch", "any"), default="ubatch",
                   help="prefill launches: one per ubatch (default), or any count (reported only: a ubatch whose "
                        "FLASH_ATTN stays on the CPU, T > 680)")
    g.add_argument("--min-prefill-tokens", type=int, default=0,
                   help="require a prefill decode of at least this many tokens in one ubatch")
    d = sub.add_parser("defer", help="device output with and without TSR_MM_DEFER")
    d.add_argument("--klcheck", required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--prompts", default="")
    d.add_argument("--out-dir", required=True)
    d.add_argument("--device", default="sim")
    d.add_argument("--n-prompts", type=int, default=2)
    d.add_argument("-n", "--n-gen", type=int, default=9)
    d.add_argument("--prompt-tail", type=int, default=8)
    d.add_argument("-ub", "--ub", type=int, default=16)
    d.add_argument("-t", "--threads", type=int, default=4)
    d.add_argument("--min-pairs", type=int, default=1, help="MUL_MAT_BEGIN / MUL_MAT_END pairs the default run must launch")
    d.add_argument("--split", action="store_true",
                   help="the environment has TSR_OPL_SPLIT: the lists are cut after n records, an END comes before each cut, so the two "
                        "runs cut at other nodes and their launch counts may differ; the cuts must still happen")
    pv = sub.add_parser("pivot", help="device output with and without Tesseract Pivot")
    pv.add_argument("--klcheck", required=True)
    pv.add_argument("--model", required=True)
    pv.add_argument("--prompts", default="")
    pv.add_argument("--out-dir", required=True)
    pv.add_argument("--device", default="sim")
    pv.add_argument("--n-prompts", type=int, default=2)
    pv.add_argument("-n", "--n-gen", type=int, default=9)
    pv.add_argument("--prompt-tail", type=int, default=8)
    pv.add_argument("-ub", "--ub", type=int, default=16)
    pv.add_argument("-t", "--threads", type=int, default=4)
    pv.add_argument("--min-moves", type=int, default=1, help="moves the pass must have made in the default run")
    pv.add_argument("--verify-k", type=int, default=0, help="also replay the tokens in verification windows of K + 1 tokens, with and without the pass")
    pv.add_argument("--expect", choices=("moe", "gdn"), default="", help="check the decode launch's records (see the module text)")
    o = sub.add_parser("ops", help="test-backend-ops for one op on the device")
    o.add_argument("--test-backend-ops", required=True)
    o.add_argument("--backend", default="Tesseract0")
    o.add_argument("--op", required=True, help="test-backend-ops -o value")
    o.add_argument("--params", default="", help="test-backend-ops -p value")
    o.add_argument("--min-run", type=int, default=1)
    o.add_argument("--max-list", type=int, default=12)
    o.add_argument("--list", default="", help="write every unsupported case to this file")
    a = ap.parse_args()
    try:
        return cmd_graph(a) if a.cmd == "graph" else cmd_defer(a) if a.cmd == "defer" else cmd_pivot(a) if a.cmd == "pivot" else cmd_ops(a)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"m7_tests: error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
