# fw_example_ising_tool: runs the tool tsr-ising on the simulator and checks its exit status, its dump and the video script.
#   cmake -DTOOL=<tsr-ising> -DVIDEO=<sk_video.py> -DOUT=<scratch dir> -P ising_tool.cmake
#
#   1. n = 64, t = 3, two column tensors, a snapshot every 4 steps, --check and --dump, under a random schedule with the
#      race detector fatal: exit 0, "agrees with the bit-exact model", the best replica line, and a dump of exactly the
#      size and the header words that ising_dump.cpp documents
#   2. the same without the energy words: the dump's curve comes from the snapshots
#   3. Ctrl-C (SIGINT) during a long run: exit 0, a stopped run
#   4. the video script on the dump of 1 (a square n = 64 needs 8 x 8 spins): exit 0, the GIF and the PNG written, when an
#      interpreter with numpy, matplotlib and Pillow is found ($TSR_PYTHON, or python3); the script recomputes the energies
#      of the snapshots from its own copy of the couplings and stops with an error when they differ from the run's
#   5. bad command lines: exit 2

file(MAKE_DIRECTORY ${OUT})
set(DUMP ${OUT}/ising_tool.dump)
file(REMOVE ${DUMP})

function(run_tool expect_rc)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env TSR_SIM_SCHED=random:2 TSR_SIM_RACE=fatal ${TOOL} ${ARGN}
                    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if (NOT rc EQUAL expect_rc)
        message(FATAL_ERROR "tsr-ising ${ARGN}: exit status ${rc}, expected ${expect_rc}\n${out}\n${err}")
    endif()
    set(TOOL_OUT "${out}" PARENT_SCOPE)
endfunction()

# The first `count` words of the dump, little-endian, against the expected words (hex, one argument per word).
function(check_header_words count)
    math(EXPR nbytes "${count} * 4")
    file(READ ${DUMP} head HEX LIMIT ${nbytes})
    string(JOIN "" want ${ARGN})
    if (NOT head STREQUAL want)
        message(FATAL_ERROR "the dump header differs:\n got  ${head}\n want ${want}")
    endif()
endfunction()

# 1
run_tool(0 --n 64 --t 3 --steps 40 --snap-every 4 --kblk 32 --seed 7 --check --dump ${DUMP})
if (NOT TOOL_OUT MATCHES "the device agrees with the bit-exact model")
    message(FATAL_ERROR "--check did not report agreement:\n${TOOL_OUT}")
endif()
if (NOT TOOL_OUT MATCHES "best: replica [0-9]+ +energy -?[0-9]+ +e -?[0-9.]+")
    message(FATAL_ERROR "no best replica line:\n${TOOL_OUT}")
endif()
# 128 header + 3 x 16 results + 40 x 3 x 8 curve + 3 x 64 x 4 final x + 10 x 3 x 64 x 4 snapshots
file(SIZE ${DUMP} dump_size)
if (NOT dump_size EQUAL 9584)
    message(FATAL_ERROR "the dump has ${dump_size} bytes, expected 9584")
endif()
# words 0..11: magic, version 2, kind 2, n 64, 64, t 3, steps 40, snap 4, flags 5 (lower is better, curve from the energy
# words), snapshots 10, reserved 0, n 64
check_header_words(12 49534450 02000000 02000000 40000000 40000000 03000000 28000000 04000000 05000000 0a000000 00000000 40000000)
file(COPY_FILE ${DUMP} ${OUT}/sk.dump)

# 2
run_tool(0 --n 64 --t 2 --steps 30 --snap-every 3 --no-energy --seed 4 --dump ${DUMP})
check_header_words(9 49534450 02000000 02000000 40000000 40000000 02000000 1e000000 03000000 01000000)   # flags 1
file(READ ${DUMP} head HEX LIMIT 88)
string(SUBSTRING "${head}" 136 24 curve_words)   # words 17, 18, 19: curve rows 10, first 3, every 3
if (NOT curve_words STREQUAL "0a0000000300000003000000")
    message(FATAL_ERROR "the curve words of the header: ${curve_words}")
endif()

# 3: a soft stop by SIGINT
find_program(TIMEOUT_EXE timeout)
if (TIMEOUT_EXE)
    file(REMOVE ${DUMP})
    set(ENV{TSR_SIM_SCHED} fixed)   # (not through cmake -E env: timeout's signal would reach that process too)
    execute_process(COMMAND ${TIMEOUT_EXE} --preserve-status -s INT 3 ${TOOL} --n 64 --t 2 --steps 16000000 --chunk 500 --snap-every 100 --dump ${DUMP}
                    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if (NOT rc EQUAL 0 OR NOT out MATCHES "stopped: [0-9]+ of 16000000 steps done" OR NOT EXISTS ${DUMP})
        message(FATAL_ERROR "the run was not stopped cleanly by SIGINT (status ${rc}):\n${out}\n${err}")
    endif()
else()
    message(STATUS "fw_example_ising_tool: no timeout program, the SIGINT case is skipped")
endif()

# 4: the video script on the dump of 1
set(VIDEO_PY "")
foreach(py $ENV{TSR_PYTHON} python3)
    if (py AND NOT VIDEO_PY)
        execute_process(COMMAND ${py} -c "import numpy, matplotlib, PIL" RESULT_VARIABLE r OUTPUT_QUIET ERROR_QUIET)
        if (r EQUAL 0)
            set(VIDEO_PY ${py})
        endif()
    endif()
endforeach()
if (VIDEO_PY)
    file(REMOVE ${OUT}/video.gif ${OUT}/video_last.png)
    execute_process(COMMAND ${VIDEO_PY} ${VIDEO} ${OUT}/sk.dump --out ${OUT}/video --gif-frames 4
                    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if (NOT rc EQUAL 0)
        message(FATAL_ERROR "sk_video.py: status ${rc}\n${out}\n${err}")
    endif()
    foreach(f video.gif video_last.png)
        if (NOT EXISTS ${OUT}/${f})
            message(FATAL_ERROR "sk_video.py did not write ${f}:\n${out}")
        endif()
    endforeach()
    file(READ ${OUT}/video_last.png magic HEX LIMIT 8)
    file(READ ${OUT}/video.gif gifmagic HEX LIMIT 6)
    if (NOT magic STREQUAL "89504e470d0a1a0a" OR NOT gifmagic STREQUAL "474946383961")
        message(FATAL_ERROR "the pictures of sk_video.py are not a PNG and a GIF")
    endif()
else()
    message(STATUS "fw_example_ising_tool: no interpreter with numpy, matplotlib and Pillow (set TSR_PYTHON), the video script is not run")
endif()

# 5
run_tool(2 --bogus)
run_tool(2 --n 1)
run_tool(2 --t 9)
run_tool(2 --steps x)
run_tool(2 --problem sk)
message(STATUS "fw_example_ising_tool: ok")
