# TFLite Micro as an IDF component, from the project-generated tree
# ../../../tflm_tree_64b3de8_generic: commit 64b3de8 (2026-10-17), the commit
# the host `tflite-micro` wheel is built from, so the board's integer arithmetic
# is the host oracle's (same reason as the H723's tflm_tree_64b3de8). GENERIC
# target, reference kernels: esp-nn has no int16 path and TFLM ships no Xtensa
# optimised kernel set for this op set, so the 16x8 model runs the portable C++
# here -- which is itself the S3 finding (the tier that beat the M7 on int8 has
# no optimised kernel for the scheme quality forced). Globbed deliberately:
# the tree is a generated output with no aggregates (see benchmark_h723).
set(TFLM_TREE ${COMPONENT_DIR}/../../../tflm_tree_64b3de8_generic)
if(NOT EXISTS ${TFLM_TREE}/tensorflow/lite/micro/micro_interpreter.h)
    message(FATAL_ERROR "tflm component: ${TFLM_TREE} missing -- regenerate at 64b3de8 "
            "(create_tflm_tree.py with no makefile options)")
endif()
file(GLOB_RECURSE TFLM_SRCS
     ${TFLM_TREE}/tensorflow/*.cc ${TFLM_TREE}/tensorflow/*.c
     ${TFLM_TREE}/signal/*.cc     ${TFLM_TREE}/signal/*.c
     ${TFLM_TREE}/third_party/*.cc ${TFLM_TREE}/third_party/*.c)
list(FILTER TFLM_SRCS EXCLUDE REGEX "test_helpers|mock_|fake_|_test\\.cc$")
# the generic tree carries only flatbuffers and kissfft under third_party
set(TFLM_INC ${TFLM_TREE} ${TFLM_TREE}/third_party/flatbuffers/include
             ${TFLM_TREE}/third_party/kissfft ${TFLM_TREE}/signal/micro/kernels)
foreach(d gemmlowp ruy)
    if(EXISTS ${TFLM_TREE}/third_party/${d})
        list(APPEND TFLM_INC ${TFLM_TREE}/third_party/${d})
    endif()
endforeach()
idf_component_register(
    SRCS ${TFLM_SRCS}
    INCLUDE_DIRS ${TFLM_INC})
target_compile_definitions(${COMPONENT_LIB} PUBLIC TF_LITE_STATIC_MEMORY
                           TF_LITE_STRIP_ERROR_STRINGS TF_LITE_DISABLE_X86_NEON)
target_compile_options(${COMPONENT_LIB} PRIVATE -w -fno-unwind-tables -fno-asynchronous-unwind-tables
                       $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti -fno-exceptions -fno-threadsafe-statics>)
