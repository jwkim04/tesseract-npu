# sw/torch — the PyTorch side of Tesseract: the C++ extension module _tesseract_rt (binds tsr_rt,
# sw/rt/tsr_rt.h) and the Python package tesseract_torch (torch.compile backend "tesseract").
#
# Enabled from the top-level CMakeLists with -DTSR_WITH_TORCH=ON (needs TSR_WITH_LLAMA for tsr_rt):
#
#   cmake -S . -B build/torch -DCMAKE_BUILD_TYPE=Release -DTSR_WITH_TORCH=ON
#   cmake --build build/torch -j4 --target tesseract_torch_ext
#   PYTHONPATH=sw/torch:build/torch/python build/tools/torchvenv/bin/python -c "import tesseract_torch"
#
# Torch is located through TSR_TORCH_PYTHON (default: build/tools/torchvenv/bin/python if it
# exists, else python3): the interpreter's torch.utils.cpp_extension gives the include and library
# directories, as for a CppExtension. Torch's own CMake config (find_package(Torch)) is not used: the
# torch wheel here is a CUDA build whose config requires a matching CUDA toolkit, and a CPU
# extension only needs c10 / torch / torch_cpu / torch_python. The module is built for that
# interpreter's Python (EXT_SUFFIX) and must be imported by it (or an identical one, e.g. its venv).

if (NOT TARGET tsr_rt)
    message(FATAL_ERROR "TSR_WITH_TORCH needs the tsr_rt runtime (sw/rt), i.e. TSR_WITH_LLAMA=ON")
endif()

set(TSR_TORCH_PYTHON "" CACHE FILEPATH "Python interpreter with torch for the extension (default: build/tools/torchvenv/bin/python, else python3)")
set(_tsr_py "${TSR_TORCH_PYTHON}")
if (NOT _tsr_py)
    if (EXISTS ${TESSERACT_ROOT}/build/tools/torchvenv/bin/python)
        set(_tsr_py ${TESSERACT_ROOT}/build/tools/torchvenv/bin/python)
    else()
        find_program(_tsr_py_found NAMES python3 REQUIRED)
        set(_tsr_py ${_tsr_py_found})
    endif()
endif()

set(_tsr_torch_query [=[
import sysconfig, torch, torch.utils.cpp_extension as c
print("VERSION=" + torch.__version__)
print("INCLUDES=" + ";".join(c.include_paths()))
print("LIBDIR=" + c.library_paths()[0])
print("CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))
print("PY_INCLUDE=" + sysconfig.get_paths()["include"])
print("EXT_SUFFIX=" + sysconfig.get_config_var("EXT_SUFFIX"))
]=])
execute_process(COMMAND ${_tsr_py} -c "${_tsr_torch_query}"
                OUTPUT_VARIABLE _tsr_torch_out ERROR_VARIABLE _tsr_torch_err RESULT_VARIABLE _tsr_torch_rc
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT _tsr_torch_rc EQUAL 0)
    message(FATAL_ERROR "TSR_WITH_TORCH: '${_tsr_py}' cannot import torch:\n${_tsr_torch_err}")
endif()
string(REPLACE "\n" ";" _tsr_torch_lines "${_tsr_torch_out}")
foreach (_l IN LISTS _tsr_torch_lines)
    if (_l MATCHES "^([A-Z0-9_]+)=(.*)$")
        set(_TSR_TORCH_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
    endif()
endforeach()
# INCLUDES is ';'-separated, which the line split above cut into several list items: rebuild it.
string(REGEX MATCH "INCLUDES=[^\n]*" _tsr_inc_line "${_tsr_torch_out}")
string(REPLACE "INCLUDES=" "" _TSR_TORCH_INCLUDES "${_tsr_inc_line}")
message(STATUS "tesseract_torch: torch ${_TSR_TORCH_VERSION} from ${_tsr_py} (${_TSR_TORCH_LIBDIR}), module suffix ${_TSR_TORCH_EXT_SUFFIX}")

set(_tsr_torch_libs)
foreach (_lib c10 torch torch_cpu torch_python)
    set(_p ${_TSR_TORCH_LIBDIR}/lib${_lib}.so)
    if (NOT EXISTS ${_p})
        message(FATAL_ERROR "TSR_WITH_TORCH: ${_p} not found")
    endif()
    list(APPEND _tsr_torch_libs ${_p})
endforeach()

set(TSR_TORCH_PY_DIR ${CMAKE_BINARY_DIR}/python CACHE INTERNAL "")

add_library(tesseract_torch_ext MODULE csrc/tsr_torch.cpp)
set_target_properties(tesseract_torch_ext PROPERTIES
    PREFIX "" OUTPUT_NAME _tesseract_rt SUFFIX "${_TSR_TORCH_EXT_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY ${TSR_TORCH_PY_DIR}
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(tesseract_torch_ext PRIVATE
    TORCH_EXTENSION_NAME=_tesseract_rt TORCH_API_INCLUDE_EXTENSION_H
    _GLIBCXX_USE_CXX11_ABI=${_TSR_TORCH_CXX11_ABI})
target_include_directories(tesseract_torch_ext SYSTEM PRIVATE ${_TSR_TORCH_INCLUDES} ${_TSR_TORCH_PY_INCLUDE})
target_compile_options(tesseract_torch_ext PRIVATE -Wall -Wextra -Wno-unused-parameter)
# tsr_rt is static and PIC (it brings tsr_lower, tsr_transport, tsr_sim, tsr_ref and libggml-base.so);
# torch's libraries by path, found at run time through the RPATH (and already loaded by `import torch`).
target_link_libraries(tesseract_torch_ext PRIVATE tsr_rt ${_tsr_torch_libs})
set_property(TARGET tesseract_torch_ext APPEND PROPERTY BUILD_RPATH ${_TSR_TORCH_LIBDIR})

# pytest suite (sw/torch/tests) as one ctest, in the interpreter the module was built for
add_test(NAME torch_pytest
         COMMAND ${_tsr_py} -m pytest -q -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set_tests_properties(torch_pytest PROPERTIES ENVIRONMENT
    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:${TSR_TORCH_PY_DIR};TSR_DEVICE=sim;TSR_SIM_THREADS=4")
add_test(NAME torch_pytest_tmax8
         COMMAND ${_tsr_py} -m pytest -q -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set_tests_properties(torch_pytest_tmax8 PROPERTIES ENVIRONMENT
    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:${TSR_TORCH_PY_DIR};TSR_DEVICE=sim;TSR_SIM_TMAX=8;TSR_SIM_THREADS=4")
