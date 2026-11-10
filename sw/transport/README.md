# Transport (`sw/transport`)

`tsr_transport` is the only target-dependent part of the host software. It connects a host process to one Tesseract
device: it binds the channel memory, loads a firmware image into the control core, boots it, launches kernels and
collects their answers. The ggml backend, `tsr_rt`, `klcheck`, `tsr-numcheck`, `tsr-selftest` and the trace tools all
use it, and every device kind runs the same register protocol. Normative:
[EX](../../docs/design/01-execution-model.md) §10 (the API), §2 to §4 (launch protocol), §8 (errors and recovery),
[SW](../../docs/design/06-software-verification.md) §9.1 and §9.2, [DB](../../docs/design/05-debug.md) §7 and §9. The
protocol layer in detail, the tests and the card side are in [`docs/xrt-host.md`](../../docs/xrt-host.md).

```text
tsr_transport.h   C API: open, close, config, write, read, csr, image, boot, launch, wait, cancel, ...
  tsr_transport.c    dispatcher: opens the kind named by TSR_DEVICE
  tsr_kdev.cpp       protocol layer, one implementation for every kind (open, bind, boot with VERIFY,
                     launch, wait, recovery ladder, TRAP handling, log)
  tsr_image.cpp, tsr_logdec.cpp, tsr_mon.{h,cpp}    firmware images, log decoder, the live monitor record
    kdev (tsr_kdev.h)   raw access to one kernel: registers, channel memory, a time base
      sim      tsr_kdev_sim.cpp        the simulator in-process behind the core CSR port, no adapter (the default)
      mock     tsr_kdev_mock.cpp       the sim kdev inside a model of the ap_ctrl_none adapter, with fault injection
      vkernel  tsr_kdev_vkernel.cpp    client of a Verilated kernel server (this tree has no server)
      xrt      tsr_transport_xrt.cpp   the Alveo card through xrt::ip and one device-only buffer per channel
```

A new kind derives from `tsr::kdev` (`name`, `num_ch`, `ch_bytes`, `ch_addr`, `reg_read`, `reg_write`, `mem_write`,
`mem_read`, `now_ns` and `idle` when time is simulated, and `detect_capacity` and `buf_bytes` when the buffer can be larger
than the memory behind it, as on `xrt`: [`docs/xrt-host.md`](../../docs/xrt-host.md) §4.4, and `core_clock_hz` when the platform reports the
core clock) and registers with
`tsr::register_kdev("mykind", factory)`, then `TSR_DEVICE=mykind` selects it. On `sim` the host triggers time (a read of the completion ring runs
the model until something happens), so results are deterministic. `tsr_transport.h` is EX §10.1 plus the
conveniences of SW §9.1. Every call returns 0 or a negative `TSR_TE_*` code (`INVAL`, `IO`, `TIMEOUT`, `LOST`,
`DEVICE`, `IMAGE`, `STATE`) and prints a message on stderr. A minimal use:

```c
tsr_transport * t = tsr_transport_open(NULL);                    // $TSR_DEVICE or sim
tsr_image * img;  tsr_image_open_default(&img);                  // $TSR_IMAGE or the build's tsr-fw.elf
tsr_transport_boot(t, img);                                      // verifies, loads, releases the core, waits for HELLO
tsr_launch l = { .type = TSR_DESC_TYPE_LAUNCH, .kernel = (uint16_t) tsr_image_kernel(img, "graph"),
                 .arg = { opl_handle, opl_bytes, 0 } };
tsr_record rec;
tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &rec);            // launch and wait; a timeout runs the recovery ladder
tsr_result r;  tsr_record_decode(&rec, &r);                      // r.status, r.ret, r.err_*
```

The rest of the API: `_write` and `_read` (channel DRAM at any byte offset and size), `_csr_read` and `_csr_write`,
`_launch`, `_wait` and `_poll` (queue depth 8), `_cancel`, `_stop` and `_soft_reset`, `_set_event_cb`,
`_log_drain`, `_get_stats` and `_perf_summary`, the version calls (`tsr_transport_versions`, `tsr_image_versions`,
`tsr_basis_version`; [`docs/versioning.md`](../../docs/versioning.md)) and the debugger hooks (`_dmi_*`, `_set_debugger`,
`_set_debug_service`). On `sim`, `mock` and `vkernel` every timeout counts device time, so results and recoveries do
not depend on host load, and on `xrt` it counts host time. One mutex per transport serializes the calls and
`tsr_transport_wait` releases it between polls.

## Environment

Read at open. A malformed value makes the open fail with a message. The variables of the `xrt` kind are in
[`docs/xrt-host.md`](../../docs/xrt-host.md) §4.3, and the simulator's own (`TSR_SIM_SCHED`, `_RACE`, `_LOG`, `_RV_*`,
`_VU_TIMING`, `TSR_TRACE`) in [`sw/sim/README.md`](../sim/README.md).

| Variable | Default | Effect |
| --- | --- | --- |
| `TSR_DEVICE`, `TSR_IMAGE` | `sim`, the build's `fw/tsr-fw.elf` | the kind (`sim`, `mock`, `vkernel`, `xrt` or a registered one) and the firmware image |
| `TSR_LAUNCH_TIMEOUT_MS`, `TSR_ABORT_MS`, `TSR_BOOT_MS`, `TSR_RESET_MS` | 60000 (0 disables), 10000, 2000, 15000 | wait timeout per launch, wait for the answers of canceled launches, wait for HELLO, wait for `SOFT_RESET` |
| `TSR_LOG`, `TSR_LOG_LEVEL` | off, 3 | decode the device log to `stderr` or a file. `LOG_CFG.LEVEL`, 0 (fatal) to 5 (trace). |
| `TSR_ATTACH` | 0 | `1`: open without `SOFT_RESET`, bind or capacity probe (nothing is written), for diagnosis of a running device |
| `TSR_STATS` | 0 | `1`: transport statistics and the PERF summary on stderr at close |
| `TSR_MON`, `TSR_MON_DIR`, `TSR_MON_PERF_MS`, `TSR_MON_KEEP`, `TSR_MON_MODEL` | on | the live monitor record (SW-87): `0` turns it off, its directory (`/dev/shm`, also the one of the card's capacity record `tsr-cap-<bdf>`, [`docs/xrt-host.md`](../../docs/xrt-host.md) §4.4), the least ms between two PERF reads (100), keep the file at exit, the model name |
| `TSR_DUMP_DIR` | off | directory for hang dumps (`hang-<pid>-<n>/state.json`) |
| `TSR_GDB`, `TSR_DEBUGGER`, `TSR_GDB_WAIT` | off | the embedded debug service when the program links it, and `boot` or `launch` to halt there (a trap then leaves the device trapped instead of restarting it) ([`docs/prof-dbg.md`](../../docs/prof-dbg.md) §1) |
| `TSR_TILE_SERIAL`, `TSR_TILE_HIST` | unset | `0` or `1`: set or clear serial mode and the dispatch history in `DBG_CFG` at open |
| `TSR_SIM_TMAX`, `TSR_SIM_CH_GIB`, `TSR_SIM_THREADS`, `TSR_SIM_DRAM_DIR` | 1, 16, `min(CPUs, 8)`, off | `sim` and `mock`: `T_MAX` of the simulated core (1 to 8), channel size in GiB, MVM worker threads, file-backed DRAM |
| `TSR_SIM_CARD` | unset | `k4full` sets the configuration of the release image: `T_MAX` 4, none of the local-intermediate `CAPS` features and no GDN op, the VU's 8 / 8 / 4 lanes behind two lane-pipe stages, `clk_rv` at half the core clock. The other variables of this table override it (SW-80). The simulator still reports `TOPK` and `DQROW`, which the image lacks. |
| `TSR_SIM_GDN` | 1 (with `TSR_SIM_CARD`: 0) | 0 clears `CAPS` bit 8 (the fused GDN op of the vector unit, VU §7.12), 1 sets it. `GATED_DELTA_NET` then runs its composed or its fused path (SW-78). `TSR_SIM_CARD=k4full TSR_SIM_GDN=1` matches the release image. |
| `TSR_SIM_VU`, `TSR_SIM_ATT_LOCAL`, `TSR_SIM_MVM_LOCAL_SCALE`, `TSR_SIM_MVM_LOCAL_SOFTCAP` | 1 | 0 models a core without the vector unit and attention (`CAPS` 0, so every kind but MUL_MAT stays on the CPU), or clears that `CAPS` feature, for matched comparisons |
| `TSR_MOCK_NUM_CH`, `TSR_MOCK_TMAX`, `TSR_MOCK_CH_GIB` | 4, 1, 16 | `mock`: channels, `T_MAX`, channel size |
| `TSR_MOCK_DMA_ALIGN`, `TSR_MOCK_DMA_CHUNK_KIB`, `TSR_MOCK_DMA_BOUNCE` | 1, 65536, 0 | `mock`: DMA alignment (power of two up to 4096), largest raw transfer in KiB, bounce every transfer |
| `TSR_MOCK_INJECT`, `TSR_MOCK_TCM_FAULT`, `TSR_MOCK_HANG` | off | `mock` fault injection: adapter errors at open or at the N-th launch push, a TCM word that does not hold what was written (caught by `VERIFY`), a unit that hangs at the N-th push |
| `TSR_VKERNEL_SERVER`, `TSR_VKERNEL_SOCKET` | unset | `vkernel`: the command line of a server to spawn (the client appends `--fd N`), or the path of a Unix socket one listens on |

## Test

`ctest --test-dir build/dev -R 'transport_|tools_selftest|tools_log'` runs `transport_kdev`, `transport_protocol`,
`transport_mon` (the monitor record and its seqlock, the capacity record of a card), `transport_xrt_noop` (only with `-DTSR_WITH_XRT=ON`),
`transport_ggml_e2e_mock*`, `tools_selftest_mock*` and `tools_log_selftest`
([`docs/xrt-host.md`](../../docs/xrt-host.md) §6 describes each). `transport_protocol` is skipped (77) when the
firmware images are missing. `test_kdev.cpp` shows registration and the faults a kdev has to survive.
