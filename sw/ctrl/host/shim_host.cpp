// shim_host.cpp — the host shim of tsr_fw_host (SW §6.8): runs one firmware kernel, compiled for
// the host (TSR_FW_HOST), on the bound simulator the way the runtime calls it, without the runtime's
// R-steps (no pop, no answer, no QCLEAR): tsr/backend_host.hpp's host_run has the contract.
#include "tsr/kernel.hpp"

namespace tsr {

host_result host_run(tsr_kernel_fn kernel, const tsr_args & args_in, const kernel_info * ki, uint32_t kernel_index,
                     bool stop) {
    tsr_args args = args_in;
    // Argument block (EX §4.3, SW §6.3 R5): one LD into the top 16 SPM lines, one join.
    if (ki != nullptr && (ki->flags & TSR_KFLAG_ARGBLK) != 0u) {
        const spm blk{spm_lines() - TSR_ARG_LINES};
        join(issue(LD(blk, from_handle(args.a[0]), ki->argblk_lines)));
        args.blk = reinterpret_cast<const volatile uint8_t *>(spm_ptr<uint8_t>(blk));
    }
    // The running launch as R5 leaves it (tsr::launch()).
    host_launch = launch_info{};
    host_launch.seq = args.seq;
    host_launch.kernel = kernel_index;
    host_launch.flags = args.flags;
    host_launch.pidx0 = pidx_now();
    for (uint32_t u = 0; u < NUM_UNITS; ++u) host_launch.base[u] = issued_seq(u);
    host_launch.active = 1u;
    host_stop = stop ? 1u : 0u;

    host_result r{};
    r.pidx0 = host_launch.pidx0;
    r.ret = kernel(&args);
    // R6: every op of the launch has completed, including aborted ones (I1).
    const ticket f = fence();
    join(f);
    r.pidx_end = pidx_now();
    // R7's rule: the smallest ERR_PIDX (relative to pidx0) over the units in error state.
    bool found = false;
    uint32_t best = 0;
    for (uint32_t u = 0; u < NUM_UNITS; ++u) {
        const uint32_t code = hw::cw_read(CW_ERR_CODE(u)) & 0xFFu;
        if (code == 0u) continue;
        const uint32_t p = hw::cw_read(CW_ERR_PIDX(u));
        if (found && p - r.pidx0 >= best) continue;
        found = true;
        best = p - r.pidx0;
        r.err_code = code;
        r.err_ticket = make_ticket(u, hw::cw_read(CW_ERR_SEQ(u))).v;
        r.err_info = hw::cw_read(CW_ERR_INFO(u));
        r.err_w0 = hw::cw_read(CW_ERR_W0(u));
        r.err_pidx = p;
    }
    if (!found) r.err_ticket = TICKET_NONE;
    r.stop_seen = host_launch.stop_seen;
    host_launch.active = 0u;
    host_stop = 0u;
    return r;
}

}  // namespace tsr
