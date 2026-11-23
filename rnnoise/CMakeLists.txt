# RNNoise (task 9): Xiph's reference C, vendored at two pinned releases, plus
# the ABF1 processor wrapper and the WAV tool that drives it through the file
# binding. Same build discipline as classical/: Release = -O2 -DNDEBUG set
# AFTER project() (trap 1), -ffast-math declined.
#
#   xiph-v0.1.1/  git tag v0.1.1 (babe7fb, 2024-03-22): the as-shipped release.
#   xiph-v0.1/    git tag v0.1   (2cb7786, 2017-09-19): the paper-era tree.
#   Identical weights (rnn_data.c, 87,503 int8 weights); v0.1.1 adds the
#   2017-10-10 gain-decay limiter (g >= 0.6 * last g) and maintenance fixes.
cmake_minimum_required(VERSION 3.16)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
project(rnnoise_bench C)
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../harness ${CMAKE_BINARY_DIR}/harness)

find_library(M_LIB m)

function(rnnoise_lib name dir)
  file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/src/*.c)
  add_library(${name} STATIC ${SRC})
  target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/include
                                     PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/src)
  # Vendored code, compiled as shipped: no -Werror, no warning flags of ours.
  if(M_LIB)
    target_link_libraries(${name} PUBLIC ${M_LIB})
  endif()
endfunction()

rnnoise_lib(rnnoise_v011 xiph-v0.1.1)
rnnoise_lib(rnnoise_v01  xiph-v0.1)

# The wrapper is compiled once per release because the v0.1 API differs.
add_library(rnnoise_proc_v011 STATIC rnnoise_proc.c)
target_link_libraries(rnnoise_proc_v011 PUBLIC rnnoise_v011 abf_file)
target_include_directories(rnnoise_proc_v011 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(rnnoise_proc_v011 PRIVATE -Wall -Wextra -Wconversion)

add_library(rnnoise_proc_v01 STATIC rnnoise_proc.c)
target_link_libraries(rnnoise_proc_v01 PUBLIC rnnoise_v01 abf_file)
target_include_directories(rnnoise_proc_v01 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(rnnoise_proc_v01 PRIVATE RNNOISE_V01)
target_compile_options(rnnoise_proc_v01 PRIVATE -Wall -Wextra -Wconversion)

# The "+N" variant (2026-10-31): the three GRUs on CMSIS-NN's s8 vector-matrix
# kernel, weights re-laid at build time, activations int8 per call. Contract
# and numerics in nn8/rnn_nn8.h. The host twin of the F411's nn8 image: the
# integer path is the same C on both, so this is where its quality is measured
# (CLAUDE.md, Locked Decisions: quality host-side, platforms contribute cost).
# nn8_tool.py check runs at every configure so the two derived files cannot
# drift from the vendored rnn.c / rnn_data.c.
set(CMSIS_NN ${CMAKE_CURRENT_SOURCE_DIR}/../tflm_tree_64b3de8/third_party/cmsis_nn)   # V.6.2.0, as the H723 links it
if(NOT EXISTS ${CMSIS_NN}/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.c)
  message(FATAL_ERROR "nn8: ${CMSIS_NN} is missing -- regenerate tflm_tree_64b3de8")
endif()
execute_process(COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/nn8/nn8_tool.py check
                RESULT_VARIABLE _nn8_rc OUTPUT_VARIABLE _nn8_out ERROR_VARIABLE _nn8_out)
if(NOT _nn8_rc EQUAL 0)
  message(FATAL_ERROR "nn8_tool.py check failed:\n${_nn8_out}")
endif()
message(STATUS "nn8: ${_nn8_out}")
file(GLOB SRC_NN8 ${CMAKE_CURRENT_SOURCE_DIR}/xiph-v0.1.1/src/*.c)
list(REMOVE_ITEM SRC_NN8 ${CMAKE_CURRENT_SOURCE_DIR}/xiph-v0.1.1/src/rnn.c)
add_library(rnnoise_v011_nn8 STATIC ${SRC_NN8}
            nn8/rnn_nn8.c nn8/rnn_nn8_gru.c nn8/rnn_nn8_weights.c
            ${CMSIS_NN}/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.c)
target_include_directories(rnnoise_v011_nn8 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/xiph-v0.1.1/include
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/xiph-v0.1.1/src ${CMAKE_CURRENT_SOURCE_DIR}/nn8 ${CMSIS_NN}/Include)
target_compile_definitions(rnnoise_v011_nn8 PRIVATE RNNOISE_NN8_COUNT_SAT)
if(M_LIB)
  target_link_libraries(rnnoise_v011_nn8 PUBLIC ${M_LIB})
endif()
add_library(rnnoise_proc_v011_nn8 STATIC rnnoise_proc.c)
target_link_libraries(rnnoise_proc_v011_nn8 PUBLIC rnnoise_v011_nn8 abf_file)
target_include_directories(rnnoise_proc_v011_nn8 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(rnnoise_proc_v011_nn8 PRIVATE RNNOISE_NN8)
target_compile_options(rnnoise_proc_v011_nn8 PRIVATE -Wall -Wextra -Wconversion)
add_executable(rnnoise_wav_nn8 rnnoise_wav.c)
target_link_libraries(rnnoise_wav_nn8 PRIVATE rnnoise_proc_v011_nn8 abf_file)
target_include_directories(rnnoise_wav_nn8 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/nn8 ${CMAKE_CURRENT_SOURCE_DIR}/xiph-v0.1.1/src)
target_compile_definitions(rnnoise_wav_nn8 PRIVATE RNNOISE_NN8 RNNOISE_NN8_COUNT_SAT)
target_compile_options(rnnoise_wav_nn8 PRIVATE -Wall -Wextra)

add_executable(rnnoise_wav rnnoise_wav.c)
target_link_libraries(rnnoise_wav PRIVATE rnnoise_proc_v011 abf_file)
target_compile_options(rnnoise_wav PRIVATE -Wall -Wextra)

add_executable(rnnoise_wav_v01 rnnoise_wav.c)
target_link_libraries(rnnoise_wav_v01 PRIVATE rnnoise_proc_v01 abf_file)
target_compile_definitions(rnnoise_wav_v01 PRIVATE RNNOISE_V01)
target_compile_options(rnnoise_wav_v01 PRIVATE -Wall -Wextra)

message(STATUS "rnnoise: CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}  C flags (release): ${CMAKE_C_FLAGS_RELEASE}")

# The RPi5 tier's run tool for the RNNoise row (task 9 -> 16, 2026-10-26): the
# classical arm's cdsp_run.c compiled unchanged against rnnoise_select instead
# of cdsp_select -- the same substitution the F411/H723 images make through
# abf_cdc_set_selector(rnnoise_select). RNNOISE_SELECT_NO_CDSP as on the F411
# image, so the executable carries bypass, OLA and RNNoise and NAKs the
# classical ids; the Pi's classical rows come from cdsp_run, not from here.
# Conditions (core pin, SCHED_FIFO, mlockall) are witnessed by cdsp_run.c as
# before; the measurement discipline is inherited, not re-implemented.
add_executable(rnnoise_run ${CMAKE_CURRENT_SOURCE_DIR}/../classical/cdsp_run.c rnnoise_select.c)
target_compile_definitions(rnnoise_run PRIVATE cdsp_select=rnnoise_select RNNOISE_SELECT_NO_CDSP)
target_include_directories(rnnoise_run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../classical)
target_link_libraries(rnnoise_run PRIVATE rnnoise_proc_v011 abf_file)
target_compile_options(rnnoise_run PRIVATE -Wall -Wextra)

# The stamped run tool (2026-11-02): rnnoise_run with per-stage CLOCK_MONOTONIC
# sums inside rnnoise_process_frame(), the F411/H723/S3 stamped build ported to
# this tier. stamps/rnnoise_denoise_stamped.c and stamps/rnnoise_stamps.c are
# byte-identical copies of benchmark_f411's (pinned here by SHA-256 so the guard
# also runs on the Pi, which carries no F411 tree), and the stamped copy must be
# the vendored denoise.c plus RN_STAMP lines and nothing else (diff -I RN_STAMP).
# Only stamps/rnnoise_stamps.h differs (the clock). Diagnostic tool, never the
# measurement tool: rnnoise_run above is untouched.
set(RN_STAMPED ${CMAKE_CURRENT_SOURCE_DIR}/stamps/rnnoise_denoise_stamped.c)
set(RN_STAMPS_C ${CMAKE_CURRENT_SOURCE_DIR}/stamps/rnnoise_stamps.c)
file(SHA256 ${RN_STAMPED} _h_st)
file(SHA256 ${RN_STAMPS_C} _h_sc)
if(NOT _h_st STREQUAL "41bf8bfe1af1a046cbb2629572663a4201ebbab5429aa0a6683d4de03a6f0f9d"
   OR NOT _h_sc STREQUAL "a45d061f1b87306bc97e06bf4cd5fd9f8eafe4dc2d212d1b8c822a9445c4ed56")
  message(FATAL_ERROR "stamps/: the stamped copies are not byte-identical to benchmark_f411's")
endif()
execute_process(COMMAND diff -I RN_STAMP ${CMAKE_CURRENT_SOURCE_DIR}/xiph-v0.1.1/src/denoise.c ${RN_STAMPED}
                RESULT_VARIABLE _rn_stamp_diff OUTPUT_VARIABLE _rn_stamp_out ERROR_VARIABLE _rn_stamp_out)
if(NOT _rn_stamp_diff EQUAL 0)
  message(FATAL_ERROR "stamps/rnnoise_denoise_stamped.c diverges from the vendored denoise.c beyond RN_STAMP lines:\n${_rn_stamp_out}")
endif()
file(GLOB RN_STAMP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/xiph-v0.1.1/src/*.c)
list(REMOVE_ITEM RN_STAMP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/xiph-v0.1.1/src/denoise.c)
add_library(rnnoise_v011_stamp STATIC ${RN_STAMP_SRC} ${RN_STAMPED} ${RN_STAMPS_C})
target_include_directories(rnnoise_v011_stamp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/xiph-v0.1.1/include
                                              PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/xiph-v0.1.1/src
                                                      ${CMAKE_CURRENT_SOURCE_DIR}/stamps)
target_compile_options(rnnoise_v011_stamp PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/stamps/rnnoise_stamps.h)
if(M_LIB)
  target_link_libraries(rnnoise_v011_stamp PUBLIC ${M_LIB})
endif()
add_library(rnnoise_proc_v011_stamp STATIC rnnoise_proc.c)
target_link_libraries(rnnoise_proc_v011_stamp PUBLIC rnnoise_v011_stamp abf_file)
target_include_directories(rnnoise_proc_v011_stamp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(rnnoise_run_stamp ${CMAKE_CURRENT_SOURCE_DIR}/../classical/cdsp_run.c rnnoise_select.c)
target_compile_definitions(rnnoise_run_stamp PRIVATE cdsp_select=rnnoise_select RNNOISE_SELECT_NO_CDSP BENCH_RNNOISE_STAMPS)
target_include_directories(rnnoise_run_stamp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../classical)
target_link_libraries(rnnoise_run_stamp PRIVATE rnnoise_proc_v011_stamp abf_file)
target_compile_options(rnnoise_run_stamp PRIVATE -Wall -Wextra)
