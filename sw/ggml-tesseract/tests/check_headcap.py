#!/usr/bin/env python3
"""check_headcap.py <mode> <log>: the records of the head launches of test-tsr-headcap on TSR_DEBUG=2 (SW-88, SW-97).

The log holds nine cases (N = 131072 / 262144 rows, 2 / 4 row blocks; the case line of test-tsr-headcap.cpp gives T and the allocation:
0 separate tensors, 1 the chain in place on y, 2 x sharing bytes with the cap's first node, 3 no cap). Per case the records of the lists that the
first graph_compute launched are checked (the later runs use the cached plan and the same lists).

  new       default lowering: per block a MUL_MAT_BEGIN and its MUL_MAT_END, the soft cap (SCALE, UNARY, SCALE) per block slice, in the order
            BEGIN 0, BEGIN 1, END 0, cap 0, BEGIN 2, END 1, cap 1, ..., BEGIN P-1, END P-2, cap P-2, END P-1, cap P-1.
  cut       the same with the lists cut inside the group (TSR_OPL_SPLIT): every list holds as many BEGINs as ENDs, the group has P products
            and P caps, in the order of the blocks, and the caps' slices are right.
  nodefer   TSR_MM_DEFER=0: P MUL_MAT records, then the whole cap over N elements; no BEGIN, no END.
  whole     TSR_HEAD_SPLIT=0: one MUL_MAT of N rows, then the whole cap.

The cases with the cap whole (T above T_MAX: no BEGIN at all; allocation 2: the cap overwrites x, so the blocks are not deferred; allocation 3:
no cap) are checked as `nodefer`, the cases cut in the group by TSR_OPL_SPLIT only for the pairing of BEGIN and END.

A block's record carries Y = the whole row's + p N / P / 16 lines and N / P rows of output; a cap slice p of record q moves the dst and the
src handles by p N / P / 16 lines from slice 0's, counts N / P elements of T rows, and keeps the row strides (N / 16 lines).
"""
import re
import sys

MM, SCALE, UNARY = 'MUL_MAT', 'SCALE', 'UNARY'
MM_N, MM_T, MM_Y, MM_YS = 6, 8, 2, 3     # tsr::opl::mm
EW_DST, EW_DS, EW_SRC, EW_SS, EW_N, EW_ROWS = 0, 1, 2, 3, 4, 5     # tsr::opl::scl and unary


def fail(msg):
    print('check_headcap: FAIL ' + msg)
    sys.exit(1)


def parse(path):
    """-> [(N, K, T, [list: [(kind, args)]])]: the launches of each case before its first cached plan."""
    cases, cur, lst, done = [], None, None, False
    rec = re.compile(r'^\[tesseract\]\s+(\d+) (\S+)\s+deps\s+(-?\d+)\s+(-?\d+)((?: [0-9a-f]{8})*)')
    for line in open(path, errors='replace'):
        m = re.match(r'headcap case: N=(\d+) K=(\d+) T=(\d+) alloc=(\d+)', line)
        if m:
            cur = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), [])
            cases.append(cur)
            lst, done = None, False
            continue
        if cur is None or done:
            continue
        if 'cached plan' in line:
            done = True
        elif re.match(r'^\[tesseract\] launch #', line):
            lst = []
            cur[4].append(lst)
        else:
            m = rec.match(line)
            if m and lst is not None:
                lst.append((m.group(2), [int(w, 16) for w in m.group(5).split()]))
    return cases


def check_plans(d):
    """TSR_DUMP_PLAN: every record (a block, a slice of the cap) is filed under the graph node it lowers, its name (SW-97)."""
    import glob
    import json
    n = 0
    for f in sorted(glob.glob(d + '/plan_*.json')):
        plan = json.load(open(f))
        for l in plan['lists']:
            for r in l['recs']:
                if r['name'] == '(JOIN)':
                    continue
                n += 1
                if r['node'] < 0 or plan['nodes'][r['node']]['name'] != r['name']:
                    fail('%s: record %d (%s) is filed under node %d' % (f, r['i'], r['name'], r['node']))
    if n == 0:
        fail('no plan records in ' + d)
    print('check_headcap: plan ok')


def check_dump(path):
    """TSR_DUMP_RE: each case dumps the head and the three nodes of its cap (the head only, when it has no cap)."""
    cases, cur = [], None
    for line in open(path, errors='replace'):
        m = re.match(r'headcap case: N=\d+ K=\d+ T=\d+ alloc=(\d+)', line)
        if m:
            cur = (int(m.group(1)), set())
            cases.append(cur)
        m = re.match(r'DUMP (\S+) ', line)
        if m and cur is not None:
            cur[1].add(m.group(1))
    if len(cases) != 9:
        fail('dump log: %d cases' % len(cases))
    for alloc, names in cases:
        want = {'result_output_pre'} if alloc == 3 else {'result_output_pre', 'cap_scale_in', 'cap_tanh', 'result_output'}
        if names != want:
            fail('dump log: a case (alloc %d) dumped %s, expected %s' % (alloc, sorted(names), sorted(want)))
    print('check_headcap: dump ok')


def main():
    mode, path = sys.argv[1], sys.argv[2]
    if mode == 'dump':
        check_dump(path)
        return
    if mode == 'plan':
        check_plans(path)
        return
    cases = parse(path)
    want = [(131072, 1, 0), (131072, 4, 0), (262144, 1, 0), (262144, 3, 0), (262144, 4, 0), (131072, 8, 0), (262144, 2, 1),
            (262144, 3, 2), (262144, 3, 3)]
    if [(c[0], c[2], c[3]) for c in cases] != want:
        fail('cases %s' % [(c[0], c[2], c[3]) for c in cases])
    for N, K, T, alloc, lists in cases:
        flat = [r for l in lists for r in l if r[0] != 'JOIN']     # a JOIN only merges the dependencies of a record
        kinds = [r[0] for r in flat]
        P = 1 if mode == 'whole' else (2 if N == 131072 else 4)
        rows = N // P
        where = 'N=%d T=%d alloc=%d %s' % (N, T, alloc, mode)
        if T > 4:                                       # a block needs two T-tiles: nothing is deferred
            if any(k in ('MUL_MAT_BEGIN', 'MUL_MAT_END') for k in kinds):
                fail('%s: a BEGIN or END for a head of more than one T-tile per block' % where)
            continue
        if alloc == 3:                                  # no cap: P plain products, no BEGIN, no END
            if kinds != [MM] * P:
                fail('%s: records %s, expected %d MUL_MAT' % (where, kinds, P))
            continue
        if mode == 'cut' and alloc == 2:
            continue
        if mode in ('nodefer', 'whole') or alloc == 2:
            exp = [MM] * P + [SCALE, UNARY, SCALE]
            if kinds != exp:
                fail('%s: records %s, expected %s' % (where, kinds, exp))
            caps = [flat[P], flat[P + 1], flat[P + 2]]
            for r in caps:
                if r[1][EW_N] != N or r[1][EW_ROWS] != T:
                    fail('%s: whole cap record: %d elements, %d rows' % (where, r[1][EW_N], r[1][EW_ROWS]))
            for p in range(P):
                a = flat[p][1]
                if a[MM_N] != rows or a[MM_T] != T or a[MM_Y] != flat[0][1][MM_Y] + p * (rows // 16):
                    fail('%s: block %d: N %d T %d Y %08x' % (where, p, a[MM_N], a[MM_T], a[MM_Y]))
            if len(lists) != 1:
                fail('%s: %d lists' % (where, len(lists)))
            continue
        # BEGIN / END pairing, per list
        for l in lists:
            k = [r[0] for r in l]
            if k.count('MUL_MAT_BEGIN') != k.count('MUL_MAT_END'):
                fail('%s: a list with %d BEGIN and %d END' % (where, k.count('MUL_MAT_BEGIN'), k.count('MUL_MAT_END')))
        blocks = [r for r in flat if r[0] in (MM, 'MUL_MAT_BEGIN')]
        if len(blocks) != P or any(r[0] == MM for r in blocks) and mode == 'new':
            fail('%s: %d product records (%s)' % (where, len(blocks), [r[0] for r in blocks]))
        for p, r in enumerate(blocks):
            a = r[1]
            if a[MM_N] != rows or a[MM_T] != T or a[MM_Y] != blocks[0][1][MM_Y] + p * (rows // 16) or a[MM_YS] != N // 16:
                fail('%s: block %d: N %d T %d Y %08x stride %d' % (where, p, a[MM_N], a[MM_T], a[MM_Y], a[MM_YS]))
        caps = [r for r in flat if r[0] in (SCALE, UNARY)]
        if len(caps) != 3 * P:
            fail('%s: %d cap records, expected %d' % (where, len(caps), 3 * P))
        for p in range(P):
            for q in range(3):
                r = caps[3 * p + q]
                b = caps[q]
                if r[0] != (UNARY if q == 1 else SCALE):
                    fail('%s: cap %d record %d is a %s' % (where, p, q, r[0]))
                a, a0, lines = r[1], b[1], p * rows // 16
                if a[EW_N] != rows or a[EW_ROWS] != T:
                    fail('%s: cap %d record %d: %d elements, %d rows, expected %d, %d' % (where, p, q, a[EW_N], a[EW_ROWS], rows, T))
                if a[EW_DST] != a0[EW_DST] + lines or a[EW_SRC] != a0[EW_SRC] + lines:
                    fail('%s: cap %d record %d: handles %08x %08x, expected %08x %08x' % (
                        where, p, q, a[EW_DST], a[EW_SRC], a0[EW_DST] + lines, a0[EW_SRC] + lines))
                if a[EW_DS] != N // 16 or a[EW_SS] != N // 16:
                    fail('%s: cap %d record %d: strides %d %d, expected %d' % (where, p, q, a[EW_DS], a[EW_SS], N // 16))
        # cap 0 reads the head's y slice 0
        if caps[0][1][EW_SRC] != blocks[0][1][MM_Y]:
            fail('%s: cap 0 does not read the first block\'s y' % where)
        if mode == 'new':
            exp = []
            for p in range(P):
                exp.append('MUL_MAT_BEGIN')
                if p >= 1:
                    exp += ['MUL_MAT_END', SCALE, UNARY, SCALE]
            exp += ['MUL_MAT_END', SCALE, UNARY, SCALE]
            if len(lists) != 1 or kinds != exp:
                fail('%s: records %s, expected %s in one list' % (where, kinds, exp))
        elif len(lists) < 2:
            fail('%s: the lists were not cut' % where)
    print('check_headcap: %s ok' % mode)


main()
