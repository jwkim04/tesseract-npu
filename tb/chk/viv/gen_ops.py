"""tb/chk/viv/gen_ops.py OUT.memh — the stimulus and expected answers of the tsr_chk netlist regression
(net_chk.sh): every directed op of tb/chk/test_chk.py directed_ops() for the U200 configuration (4 channels,
T_MAX 8, the tsr_pkg defaults), one line per op: the 256-bit instruction (word 7 on the left), the
simulator's error code and ERR_INFO, all in hex. The simulator is the oracle (libtsr_sim.so, as tb/chk)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parents[1], HERE.parent, HERE.parents[1] / "issue"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import isa            # noqa: E402
import issue_sim as S             # noqa: E402
import test_chk as T              # noqa: E402


def main(path):
    cfg, qd = S.config(num_ch=4, t_max=8), S.U200_QD
    sim = S.Sim(cfg, qd)
    lines = []
    for w in T.directed_ops(cfg):
        unit, code, info, _ = sim.check(w)
        lines.append("%064x %x %08x" % (isa.insn_int(w), code, info))
    sim.close()
    Path(path).write_text("\n".join(lines) + "\n")
    print("gen_ops: %d ops -> %s (codes %s)" % (len(lines), path, sorted({int(x.split()[1], 16) for x in lines})))


if __name__ == "__main__":
    main(sys.argv[1])
