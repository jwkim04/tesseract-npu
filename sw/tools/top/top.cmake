# sw/tools/top/top.cmake — tsr-top, the live monitor of the card (docs/prof-dbg.md §7, SW-87).
# Included by sw/tools/CMakeLists.txt.
#
#   tsr-top         bin/tsr-top, a launcher for tsr_top.py (Python standard library only: curses, json,
#                   struct). Reads the record that the process owning the card publishes
#                   (sw/transport/tsr_mon.h) and shows it like nvtop does
#   tsr-mon-probe   opens an idle card briefly in attach mode and prints its identity, clocks and lifetime
#                   counters as JSON (tsr-top --probe)
#
# Tests (ctest -R tools_top; prefix tools_top_):
#   tools_top_layout   the field table of tsr_top.py equals the one tsr_mon.h prints (test-tsr-mon layout)
#   tools_top_synth    a record written by the test: the reader, the rates and the U split against
#                      hand-computed numbers, the text frame at several sizes, stale, torn and foreign records,
#                      a record that changes while tsr-top samples it
#   tools_top_live     a tiny model on the simulator with the publisher on: the record while it runs and the
#                      one it leaves (TSR_MON_KEEP), tokens, model, the JSON snapshot, TSR_MON=0, the probe
#   tools_top_identity klcheck on the tiny model, M6 and M7, with TSR_MON=0 and with the publisher on and a PERF
#                      read after every launch: every logits row bit-identical, and the publishing run did publish
#   tools_top_curses   the curses screen in a pseudo-terminal (tmux; skipped, 77, without it): the frame,
#                      F2, F6, +/-, resize, q

if (TSR_PYTHON3)
    set(TSR_TOP_PYTHON ${TSR_PYTHON3})
else()
    find_program(TSR_TOP_PYTHON NAMES python3)
endif()

add_executable(tsr-mon-probe ${CMAKE_CURRENT_LIST_DIR}/tsr_mon_probe.cpp)
set_target_properties(tsr-mon-probe PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_options(tsr-mon-probe PRIVATE ${TSR_FP_FLAGS} -Wall -Wextra -Wshadow)
target_link_libraries(tsr-mon-probe PRIVATE tsr_transport)

if (TSR_TOP_PYTHON)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/tsr-top.in ${CMAKE_BINARY_DIR}/bin/tsr-top @ONLY
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/tsr-top.in)
    set(_tsr_top_test ${CMAKE_CURRENT_LIST_DIR}/tests/test_top.py)
    add_test(NAME tools_top_layout COMMAND ${TSR_TOP_PYTHON} ${_tsr_top_test} layout $<TARGET_FILE:test-tsr-mon>)
    add_test(NAME tools_top_synth COMMAND ${TSR_TOP_PYTHON} ${_tsr_top_test} synth)
    find_program(TSR_TMUX tmux HINTS $ENV{HOME}/.local/bin)
    add_test(NAME tools_top_curses COMMAND ${TSR_TOP_PYTHON} ${_tsr_top_test} curses ${TSR_TMUX})
    set_tests_properties(tools_top_layout tools_top_synth tools_top_curses PROPERTIES TIMEOUT 300)
    set_tests_properties(tools_top_curses PROPERTIES SKIP_RETURN_CODE 77)
    set_property(GLOBAL PROPERTY TSR_TOP_TEST_SCRIPT ${_tsr_top_test})
    set_property(GLOBAL PROPERTY TSR_TOP_TEST_PYTHON ${TSR_TOP_PYTHON})
    # The live test runs llama-completion, which the top-level directory adds after this one.
    function(tsr_add_top_live)
        get_property(_t GLOBAL PROPERTY TSR_TOP_TEST_SCRIPT)
        get_property(_py GLOBAL PROPERTY TSR_TOP_TEST_PYTHON)
        if (NOT TARGET llama-completion)
            return()
        endif()
        add_test(NAME tools_top_live
                 COMMAND ${_py} ${_t} live $<TARGET_FILE:llama-completion>
                         ${CMAKE_BINARY_DIR}/models/tiny/qwen2-tiny-q4_k.gguf $<TARGET_FILE:tsr-mon-probe>)
        set_tests_properties(tools_top_live PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
        if (TARGET klcheck)
            add_test(NAME tools_top_identity
                     COMMAND ${_py} ${_t} identity $<TARGET_FILE:klcheck> ${CMAKE_BINARY_DIR}/models/tiny/qwen2-tiny-q4_k.gguf
                             ${CMAKE_BINARY_DIR}/models/tiny/tiny-prompts.txt)
            set_tests_properties(tools_top_identity PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77)
        endif()
    endfunction()
    cmake_language(DEFER DIRECTORY ${TESSERACT_ROOT} CALL tsr_add_top_live)
endif()
