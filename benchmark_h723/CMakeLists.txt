cmake_minimum_required(VERSION 3.22)

#
# This file is generated only once,
# and is not re-generated if converter is called multiple times.
#
# User is free to modify the file as much as necessary
#

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)


# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# --- Benchmark build flags -------------------------------------------------
# Identical in intent to benchmark_f411/CMakeLists.txt, and identical in text
# where it can be: a flag difference between the two boards would show up as a
# platform difference in the results.
#
# CubeMX defaults Release to "-Os -g0". -Os optimises for size and costs
# throughput, and the probe must be built as the deployed firmware is. -g3
# changes no generated code, so Release stays debuggable.
# -DNDEBUG matters: the CubeMX toolchain file overrides CMAKE_C_FLAGS_RELEASE
# without it, so assert() would be live inside timed regions. The ASM line is
# pinned for the same reason -- CMake otherwise defaults the assembler to
# "-O3 -DNDEBUG" while C gets neither.
# Set the project name
set(CMAKE_PROJECT_NAME benchmark_h723)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME})
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable CMake support for ASM and C languages
enable_language(C CXX ASM)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# NOTE (2026-09-11): these three MUST stay after project()/enable_language().
# They were previously set above project(), where they read correctly and did
# nothing: project() re-initialises the per-language flag variables, so CubeMX's
# "-Os -g0" was restored and every kernel in the first H723 run was measured at
# -Os with assert() live. The F411 file has them after project(), which is why
# it was never affected. The guard below makes a recurrence fail the build
# rather than the results.
set(CMAKE_C_FLAGS_RELEASE     "-O2 -g3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE   "-O2 -g3 -DNDEBUG")
set(CMAKE_ASM_FLAGS_RELEASE   "-O2 -g3 -DNDEBUG")

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# --- RAM placement: DTCM vs AXI SRAM ---------------------------------------
# A MEASURED VARIABLE of this tier, not a build convenience. DTCM is
# zero-wait-state, tightly coupled and uncached, 128 KB; AXI SRAM is 320 KB
# behind the bus matrix and cached. Switch with -DBENCH_RAM=AXI and rebuild;
# the harness records where its data actually landed in g_plat.bss_addr.
# The stack and heap never move -- _estack stays pinned to the top of DTCMRAM,
# so stack placement can never contaminate the comparison.
set(BENCH_RAM "DTCM" CACHE STRING "RAM region for benchmark .data/.bss: DTCM or AXI")
set_property(CACHE BENCH_RAM PROPERTY STRINGS DTCM AXI)
if(BENCH_RAM STREQUAL "DTCM")
    set(BENCH_RAM_REGION DTCMRAM)
elseif(BENCH_RAM STREQUAL "AXI")
    set(BENCH_RAM_REGION RAM_D1)
else()
    message(FATAL_ERROR "BENCH_RAM must be DTCM or AXI, got '${BENCH_RAM}'")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bench_ram.ld.in
               ${CMAKE_CURRENT_BINARY_DIR}/bench_ram.ld @ONLY)
# so the linker script's `INCLUDE bench_ram.ld` resolves to the generated copy
# Two further placement variables (2026-11-01), generated into bench_place.ld
# the same way -- see bench_place.ld.in and rnnoise_place.c (the boot copy,
# called first in main(), with memcmp-verified witnesses read over SWD):
#   BENCH_RNNOISE_ITCM=ON      RNNoise's hot code (transforms, pitch, LPC, GRU +
#                              kernel, denoise, processor) + libm + libgcc in ITCM
#   BENCH_RNNOISE_WEIGHTS=AXI  every RNNoise weight array in AXI SRAM (the S3's
#                              +W counterpart: rnn_data.c and rnn_nn8_weights.c)
# Neither carries a HELLO letter on this tier (23-char limit): the CRC is the
# identity, and the ELF's symbol table says what moved.
option(BENCH_RNNOISE_ITCM "RNNoise: copy the hot kernels' code to ITCM at boot" OFF)
set(BENCH_RNNOISE_WEIGHTS "FLASH" CACHE STRING "RNNoise weight arrays: FLASH (as linked) or AXI (copied to RAM_D1 at boot)")
set_property(CACHE BENCH_RNNOISE_WEIGHTS PROPERTY STRINGS FLASH AXI)
if(NOT BENCH_RNNOISE_WEIGHTS MATCHES "^(FLASH|AXI)$")
    message(FATAL_ERROR "BENCH_RNNOISE_WEIGHTS must be FLASH or AXI, got '${BENCH_RNNOISE_WEIGHTS}'")
endif()
if(BENCH_RNNOISE_ITCM)
    set(BENCH_ITCM_PATTERNS "*kiss_fft.c.obj(.text .text*) *pitch.c.obj(.text .text*) *celt_lpc.c.obj(.text .text*) *rnn.c.obj(.text .text*) *rnn_nn8.c.obj(.text .text*) *rnn_nn8_gru.c.obj(.text .text*) *arm_nn_vec_mat_mult_t_svdf_s8.c.obj(.text .text*) *denoise.c.obj(.text .text*) *rnnoise_denoise_stamped.c.obj(.text .text*) *rnnoise_proc.c.obj(.text .text*) *libm.a:*(.text .text*) *libgcc.a:*(.text .text*)")
else()
    set(BENCH_ITCM_PATTERNS "/* empty: BENCH_RNNOISE_ITCM=OFF */")
endif()
if(BENCH_RNNOISE_WEIGHTS STREQUAL "AXI")
    set(BENCH_AXI_RODATA_PATTERNS "*rnn_data.c.obj(.rodata .rodata*) *rnn_nn8_weights.c.obj(.rodata .rodata*)")
else()
    set(BENCH_AXI_RODATA_PATTERNS "/* empty: BENCH_RNNOISE_WEIGHTS=FLASH */")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bench_place.ld.in
               ${CMAKE_CURRENT_BINARY_DIR}/bench_place.ld @ONLY)
message(STATUS "Placement: ITCM code ${BENCH_RNNOISE_ITCM}, RNNoise weights ${BENCH_RNNOISE_WEIGHTS}")
# Diagnostic only (2026-11-01, the I-cache-off anomaly): slide .text by N bytes with
# no code change, to move loops across 32-byte flash lines. 0 in every measurement image.
set(BENCH_TEXT_SHIFT 0 CACHE STRING "diagnostic: shift .text by N bytes (0 = none)")
if(NOT BENCH_TEXT_SHIFT EQUAL 0)
    target_link_options(${CMAKE_PROJECT_NAME} PRIVATE -Wl,--defsym=BENCH_TEXT_SHIFT=${BENCH_TEXT_SHIFT})
    message(STATUS "DIAGNOSTIC: .text shifted by ${BENCH_TEXT_SHIFT} bytes -- not a measurement image")
endif()
target_link_options(${CMAKE_PROJECT_NAME} PRIVATE -L${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "Benchmark RAM placement: ${BENCH_RAM} (${BENCH_RAM_REGION})")

# --- L1 caches -------------------------------------------------------------
# Also measured variables; default on, which is how an M7 audio product ships.
option(BENCH_ICACHE "Enable the Cortex-M7 instruction cache" ON)
option(BENCH_DCACHE "Enable the Cortex-M7 data cache" ON)

# --- CMSIS-DSP (vendored subset) -------------------------------------------
# Version 1.10.0 out of STM32Cube FW_H7 V1.13.0 -- the SAME library version as
# the F411 tier vendors from FW_F4 V1.28.3, verified byte-identical on
# arm_mat_vec_mult_q7.c. Had they differed, part of the F411-vs-H723 delta
# would have been a library-version artifact.
#
# Named individually, never globbed: CMSIS-DSP 1.10 ships both per-function
# files and per-folder aggregates that #include them all, and a GLOB links both
# and fails on duplicate symbols.
set(CMSIS_DSP ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/Third_Party/CMSIS-DSP)

# --- TinyUSB (vendored) ----------------------------------------------------
# Release 0.21.0, commit dae3f9a366bfcddbf9dcf1b48d7500286a849539, copied from
# the F411 tree so the commit is identical by construction. One stack across
# all three MCU platforms is a locked decision: three vendor stacks would make
# the "flash/SRAM after USB" line item three different footprints reported as
# a platform difference.
set(TINYUSB ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/Third_Party/tinyusb)

# The USB stack is switchable so the footprint delta is a scripted two-build
# comparison rather than a remembered number:
#   cmake -B build/NoUSB -DBENCH_WITH_TINYUSB=OFF ... && cmake --build build/NoUSB
option(BENCH_WITH_TINYUSB "Link the TinyUSB device stack" ON)

# The throughput probe (bench_kernels.c and its 64 KB weight pool) is what
# task 3 measured with; a DEPLOYED streaming image does not carry it, and on
# this tier it would not FIT beside the classical context in DTCM. OFF is the
# measurement image for the streaming harness. bench_dwt.c and bench_plat.c
# stay either way: the tick source reads the counter the first starts, and
# the second owns cache/MPU init and the register readout.
option(BENCH_WITH_PROBE "Run the kernel throughput probe at boot" ON)

# --- DTLN on TFLite Micro (task 14 -> 18, 2026-10-21) -----------------------
# The pruned 16x8 DTLN pair (dtln/port/dtln_model_p020_i16.c, SHA-256 inside)
# behind an ABF1 processor, selected by CONFIG.algo_id = ABF_ALGO_DTLN. The
# runtime is the TFLite Micro tree generated at the SAME commit the host
# `tflite-micro` wheel was built from (64b3de8, 2026-10-17), so the board's
# integer arithmetic is the host oracle's by construction; the footprint
# measurement in ../tflm_footprint used an older pin (d0dabe51) and is a
# separate number. Switchable so the flash/RAM delta is a scripted two-build
# comparison: -DBENCH_WITH_DTLN=OFF gives the classical-only image.
option(BENCH_WITH_DTLN "Carry DTLN 16x8 on TFLite Micro" ON)
set(TFLM_TREE ${CMAKE_CURRENT_SOURCE_DIR}/../tflm_tree_64b3de8)

# --- ABF1 streaming harness (task 7) ----------------------------------------
# abf/ holds COPIES of the normative sources in ../harness, like bench/bench.h;
# identical text to the F411 tree's block. When the normative tree is present a
# drifted copy stops the configure.
set(ABF_NORMATIVE ${CMAKE_CURRENT_SOURCE_DIR}/../harness)
set(ABF_FILES abf_frame.h abf_frame.c abf_cdc.h abf_cdc.c abf_cdc_tinyusb.c
              abf_plat_dwt.c abf_proc.h abf_proc.c)
if(EXISTS ${ABF_NORMATIVE}/abf_frame.h)
    foreach(f ${ABF_FILES})
        file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/abf/${f} _ours)
        file(SHA256 ${ABF_NORMATIVE}/${f} _norm)
        if(NOT _ours STREQUAL _norm)
            message(FATAL_ERROR "abf/${f} differs from ${ABF_NORMATIVE}/${f}: "
                    "the ABF1 sources must be byte-identical on every tier. "
                    "cp ../harness/${f} abf/${f} (or the other way, deliberately).")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                     ${CMAKE_CURRENT_SOURCE_DIR}/abf/${f} ${ABF_NORMATIVE}/${f})
    endforeach()
    message(STATUS "abf/: ${ABF_FILES} verified identical to ../harness")
endif()
string(TIMESTAMP ABF_BUILD_STAMP "%Y%m%dT%H%M" UTC)
set(ABF_BUILD_ID "${CMAKE_BUILD_TYPE} ${BENCH_RAM} ${ABF_BUILD_STAMP}")

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user sources here
    bench/bench_dwt.c
    bench/bench_plat.c

    ${CMSIS_DSP}/Source/TransformFunctions/arm_cfft_f32.c
    ${CMSIS_DSP}/Source/TransformFunctions/arm_cfft_init_f32.c
    ${CMSIS_DSP}/Source/TransformFunctions/arm_cfft_radix8_f32.c
    ${CMSIS_DSP}/Source/TransformFunctions/arm_bitreversal2.c
    ${CMSIS_DSP}/Source/TransformFunctions/arm_rfft_fast_f32.c
    ${CMSIS_DSP}/Source/TransformFunctions/arm_rfft_fast_init_f32.c
    ${CMSIS_DSP}/Source/CommonTables/arm_common_tables.c
    ${CMSIS_DSP}/Source/CommonTables/arm_const_structs.c
    ${CMSIS_DSP}/Source/BasicMathFunctions/arm_dot_prod_f32.c
    ${CMSIS_DSP}/Source/BasicMathFunctions/arm_mult_f32.c
    ${CMSIS_DSP}/Source/ComplexMathFunctions/arm_cmplx_mag_f32.c
    ${CMSIS_DSP}/Source/MatrixFunctions/arm_mat_vec_mult_f32.c
    ${CMSIS_DSP}/Source/MatrixFunctions/arm_mat_vec_mult_q7.c
    ${CMSIS_DSP}/Source/FastMathFunctions/arm_vexp_f32.c
    ${CMSIS_DSP}/Source/FastMathFunctions/arm_vlog_f32.c
)

if(BENCH_WITH_PROBE)
target_sources(${CMAKE_PROJECT_NAME} PRIVATE bench/bench_kernels.c bench/bench_itm.c)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE BENCH_WITH_PROBE)
endif()

if(BENCH_WITH_TINYUSB)
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    usb/usb_glue.c
    usb/usb_descriptors.c
    abf/abf_frame.c
    abf/abf_cdc.c
    abf/abf_cdc_tinyusb.c
    abf/abf_plat_dwt.c
    abf/abf_proc.c
    ${TINYUSB}/src/tusb.c
    ${TINYUSB}/src/common/tusb_fifo.c
    ${TINYUSB}/src/device/usbd.c   # usbd_control.c was merged into usbd.c in 0.21.0
    ${TINYUSB}/src/class/cdc/cdc_device.c
    ${TINYUSB}/src/portable/synopsys/dwc2/dcd_dwc2.c
    ${TINYUSB}/src/portable/synopsys/dwc2/dwc2_common.c
)
# Task 8's classical arm from the normative tree UNMODIFIED (task 15 compares
# this port against the oracle built from the same file); the FFT seam is shared
# with the F411 since both are CMSIS-DSP targets.
# As on the F411 (2026-10-26, same semantics): the RNNoise measurement image
# does NOT carry the classical arm -- RNNoise + classical in DTCM overflows by
# 1,736 B at the measured reservations (RESULTS.md, 2026-10-22) -- unless
# BENCH_RNNOISE_WITH_CDSP=ON asks for the combined image. BENCH_WITH_CDSP=OFF
# without RNNoise is that image's build-twice twin (binding + bypass + OLA).
option(BENCH_RNNOISE_WITH_CDSP "RNNoise image: also link the classical arm (does not fit in DTCM)" OFF)
option(BENCH_WITH_CDSP "Link task 8's classical arm" ON)
if(BENCH_WITH_CDSP AND (NOT BENCH_WITH_RNNOISE OR BENCH_RNNOISE_WITH_CDSP))
    set(BENCH_CDSP_LINKED ON)
else()
    set(BENCH_CDSP_LINKED OFF)
endif()
if(BENCH_CDSP_LINKED)
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../classical/cdsp.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../classical/cdsp_select.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../classical/cdsp_fft_cmsis.c)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE BENCH_WITH_CDSP)
else()
    if(BENCH_WITH_DTLN)
        # dtln_select falls through to cdsp_select for bypass / OLA / classical.
        message(FATAL_ERROR "BENCH_WITH_DTLN needs the classical arm linked (BENCH_WITH_CDSP=ON, "
                "and BENCH_RNNOISE_WITH_CDSP=ON if RNNoise is also on)")
    endif()
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE RNNOISE_SELECT_NO_CDSP)
    message(STATUS "classical arm: NOT linked (RNNoise image, or its build-twice twin)")
endif()
target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_place.c)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE usb abf ${CMAKE_CURRENT_SOURCE_DIR} ${TINYUSB}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../classical)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    BENCH_WITH_TINYUSB
    ABF_PROC_MAX_FRAME=512u     # as on the F411: the locked frame, nothing bigger
    CDSP_FFT_MAX=512u           # classical context ~23 KB; 8 kHz is NAKed here too
    ABF_BUILD_ID="${ABF_BUILD_ID}"
)
option(BENCH_CDSP_STAMPS "classical arm: per-stage DWT stage sums" OFF)
if(BENCH_CDSP_STAMPS)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE CDSP_STAMP_DWT)
endif()

# --- RNNoise, as shipped (task 9 -> 18, 2026-10-22) ---------------------------
# Xiph v0.1.1 from ../rnnoise/xiph-v0.1.1 compiled UNMODIFIED (the host oracle
# is built from the same files, task 15), behind the ABF1 processor the host
# oracle already uses (../rnnoise/rnnoise_proc.c), selected by CONFIG.algo_id =
# ABF_ALGO_RNNOISE. Geometry is the weights' own: 48 kHz, frame 960 / hop 480.
# The transform is Xiph's mixed-radix kiss_fft as shipped (480 is not a power
# of two, so neither CMSIS rfft size applies; this is what the budget costed).
# Weights are int8 in .rodata (rnn_data.c, 87,503 B). rnn_reader.c (model
# from a FILE*, fscanf) is deliberately not compiled.
#
# OFF by default so every recorded image re-emits unchanged. The RNNoise
# measurement image is Release + BENCH_WITH_PROBE=OFF + BENCH_WITH_RNNOISE=ON.
option(BENCH_WITH_RNNOISE "Carry Xiph RNNoise v0.1.1 beside the classical arm" OFF)
if(BENCH_WITH_RNNOISE)
    if(NOT BENCH_WITH_TINYUSB)
        message(FATAL_ERROR "BENCH_WITH_RNNOISE needs the ABF1 binding (BENCH_WITH_TINYUSB=ON)")
    endif()
    if(BENCH_WITH_PROBE AND BENCH_RAM STREQUAL "DTCM")
        # The probe's 64 KB pool plus RNNoise's ~29 KB heap and its stack do not
        # fit beside the binding in DTCM; malloc would fail at the first CONFIG
        # and the run would be NAKed, not refused at build time. Break the build.
        message(FATAL_ERROR "BENCH_WITH_RNNOISE and BENCH_WITH_PROBE together exceed DTCM: "
                "configure with -DBENCH_WITH_PROBE=OFF")
    endif()
    set(RNNOISE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../rnnoise/xiph-v0.1.1/src)
    set(RNNOISE_FILES ${RNNOISE_SRC}/denoise.c ${RNNOISE_SRC}/kiss_fft.c ${RNNOISE_SRC}/pitch.c
                      ${RNNOISE_SRC}/celt_lpc.c ${RNNOISE_SRC}/rnn.c ${RNNOISE_SRC}/rnn_data.c)
    # -DBENCH_RNNOISE_STAMPS=ON: per-stage DWT sums inside rnnoise_process_frame(),
    # the F411's stamped build (benchmark_f411/rnnoise_stamps.h, 2026-10-30) ported
    # 2026-10-31. The three files are byte-identical copies of the F411's, asserted
    # here every configure, and the stamped copy must be the vendored denoise.c plus
    # RN_STAMP lines and nothing else (diff -I RN_STAMP), or the sums decompose a
    # different program than the one measured. Diagnostic image, never the
    # measurement image. Sums live in .bss and are read by the debugger after a run
    # (read_rnnoise_stamps.sh, target/stm32h7x.cfg); the debugger reads through the
    # AHB, not the D-cache, so the sums must sit in DTCM -- AXI is refused.
    option(BENCH_RNNOISE_STAMPS "RNNoise image: per-stage DWT stage sums (diagnostic)" OFF)
    # -DBENCH_RNNOISE_VARIANT=ship|splibm|spconst|nn8|nn8fs: the F411's labelled
    # variants (benchmark_f411/CMakeLists.txt, 2026-10-30/31), ported 2026-11-01.
    #   splibm  (+F)  single-precision libm seam, rnnoise_sp_libm.h force-included
    #                 (byte-identical to the F411's and the S3's, asserted here);
    #   spconst (+FS) the seam plus -fsingle-precision-constant;
    #   nn8     (+N)  the three GRUs on CMSIS-NN arm_nn_vec_mat_mult_t_svdf_s8 with
    #                 per-call int8 activations (../rnnoise/nn8/rnn_nn8.h);
    #   nn8fs   (+M)  nn8 on top of the spconst seam.
    # All quality-affecting, all labelled rows, never the measurement image; the
    # host twins are rnnoise_wav (+F/+FS) and rnnoise_wav_nn8 (+N/+M). Unlike the
    # F411 the build id carries NO variant letter: "Release DTCM yyyymmddThhmm" is
    # already past the 23-character HELLO limit, so the image CRC is the identity
    # (as for the stamped image, RESULTS.md 2026-10-31). Orthogonal to the stamps.
    set(BENCH_RNNOISE_VARIANT "ship" CACHE STRING "RNNoise image variant: ship | splibm | spconst | nn8 | nn8fs")
    if(NOT BENCH_RNNOISE_VARIANT MATCHES "^(ship|splibm|spconst|nn8|nn8fs)$")
        message(FATAL_ERROR "BENCH_RNNOISE_VARIANT=${BENCH_RNNOISE_VARIANT}: must be ship, splibm, spconst, nn8 or nn8fs")
    endif()
    set(RNNOISE_EXTRA_OPTS "")
    if(BENCH_RNNOISE_VARIANT MATCHES "^(splibm|spconst|nn8fs)$")
        foreach(_other ../benchmark_f411/rnnoise_sp_libm.h ../benchmark_s3/main/rnnoise_sp_libm.h)
            file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_sp_libm.h _h_here)
            file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/${_other} _h_other)
            if(NOT _h_here STREQUAL _h_other)
                message(FATAL_ERROR "rnnoise_sp_libm.h is not byte-identical to ${_other}")
            endif()
        endforeach()
        set(RNNOISE_EXTRA_OPTS "-include;${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_sp_libm.h")
        if(NOT BENCH_RNNOISE_VARIANT STREQUAL "splibm")
            set(RNNOISE_EXTRA_OPTS "${RNNOISE_EXTRA_OPTS};-fsingle-precision-constant")
        endif()
        message(STATUS "RNNoise variant ${BENCH_RNNOISE_VARIANT}: single-precision libm seam, byte-identical to the F411's and the S3's (diagnostic, not the measurement image)")
    endif()
    if(BENCH_RNNOISE_VARIANT MATCHES "^nn8")
        # rnn.c is replaced by its derived twin (the forwarder) plus the CMSIS-NN
        # GRU, the re-laid weights and the one kernel it calls. nn8_tool.py check
        # runs every configure and refuses the build if either derived file differs
        # from a fresh generation or the activation helpers are not rnn.c's verbatim.
        set(RNNOISE_NN8 ${CMAKE_CURRENT_SOURCE_DIR}/../rnnoise/nn8)
        set(CMSIS_NN ${CMAKE_CURRENT_SOURCE_DIR}/../tflm_tree_64b3de8/third_party/cmsis_nn)   # V.6.2.0, the DTLN image's copy
        if(NOT EXISTS ${CMSIS_NN}/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.c)
            message(FATAL_ERROR "nn8: ${CMSIS_NN} is missing -- regenerate tflm_tree_64b3de8")
        endif()
        execute_process(COMMAND python3 ${RNNOISE_NN8}/nn8_tool.py check
                        RESULT_VARIABLE _nn8_rc OUTPUT_VARIABLE _nn8_out ERROR_VARIABLE _nn8_out)
        if(NOT _nn8_rc EQUAL 0)
            message(FATAL_ERROR "nn8_tool.py check failed:\n${_nn8_out}")
        endif()
        list(REMOVE_ITEM RNNOISE_FILES ${RNNOISE_SRC}/rnn.c)
        list(APPEND RNNOISE_FILES ${RNNOISE_NN8}/rnn_nn8.c ${RNNOISE_NN8}/rnn_nn8_gru.c
                                  ${RNNOISE_NN8}/rnn_nn8_weights.c
                                  ${CMSIS_NN}/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.c)
        target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${RNNOISE_NN8} ${CMSIS_NN}/Include)
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE RNNOISE_NN8)
        # The nn8 files sit outside Xiph's src/ and include "tansig_table.h" etc. by bare name.
        set(RNNOISE_EXTRA_OPTS "${RNNOISE_EXTRA_OPTS};-I${RNNOISE_SRC}")
        message(STATUS "RNNoise variant ${BENCH_RNNOISE_VARIANT}: GRUs on CMSIS-NN svdf_s8 (${_nn8_out})")
    endif()
    if(BENCH_RNNOISE_STAMPS)
        if(NOT BENCH_RAM STREQUAL "DTCM")
            message(FATAL_ERROR "BENCH_RNNOISE_STAMPS needs BENCH_RAM=DTCM: the debugger reads the sums through the AHB and would see stale D-cache lines in AXI RAM")
        endif()
        set(RNNOISE_STAMPED ${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_denoise_stamped.c)
        foreach(_f rnnoise_stamps.h rnnoise_stamps.c rnnoise_denoise_stamped.c)
            file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/${_f} _h_here)
            file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/../benchmark_f411/${_f} _h_f411)
            if(NOT _h_here STREQUAL _h_f411)
                message(FATAL_ERROR "${_f} is not byte-identical to benchmark_f411/${_f}")
            endif()
        endforeach()
        execute_process(COMMAND diff -I RN_STAMP ${RNNOISE_SRC}/denoise.c ${RNNOISE_STAMPED}
                        RESULT_VARIABLE _rn_stamp_diff OUTPUT_VARIABLE _rn_stamp_out ERROR_VARIABLE _rn_stamp_out)
        if(NOT _rn_stamp_diff EQUAL 0)
            message(FATAL_ERROR "rnnoise_denoise_stamped.c diverges from the vendored denoise.c beyond RN_STAMP lines:\n${_rn_stamp_out}")
        endif()
        list(REMOVE_ITEM RNNOISE_FILES ${RNNOISE_SRC}/denoise.c)
        list(APPEND RNNOISE_FILES ${RNNOISE_STAMPED})
        target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_stamps.c)
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE BENCH_RNNOISE_STAMPS)
        # -I: the stamped copy sits outside Xiph's src/ and includes "kiss_fft.h" etc. by bare name.
        # Appended, so a variant's -include (rnnoise_sp_libm.h) still applies to the stamped file.
        set(RNNOISE_EXTRA_OPTS "${RNNOISE_EXTRA_OPTS};-I${RNNOISE_SRC};-include;${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_stamps.h")
        message(STATUS "RNNoise stamps: ON (three files byte-identical to the F411's; stamped copy = vendored + RN_STAMP lines only; diagnostic, not the measurement image)")
    endif()
    set(RNNOISE_PROC ${CMAKE_CURRENT_SOURCE_DIR}/../rnnoise/rnnoise_proc.c)
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${RNNOISE_FILES} ${RNNOISE_PROC}
        ${CMAKE_CURRENT_SOURCE_DIR}/../rnnoise/rnnoise_select.c)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../rnnoise
        ${CMAKE_CURRENT_SOURCE_DIR}/../rnnoise/xiph-v0.1.1/include)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE BENCH_WITH_RNNOISE RNNOISE_PROC_MCU)
    # Contraction: the classical rows on this tier are the default (contracting)
    # build, with the bit-exact -ffp-contract=off twin recorded at +6-14 % per
    # hop (classical/RESULTS.md). RNNoise follows the same convention; OFF
    # builds the twin. Vendored files are compiled as shipped: -w, no warning
    # flags of ours.
    option(BENCH_RNNOISE_FP_CONTRACT "RNNoise: GCC's default FMA contraction (OFF = bit-exact twin of the x86 oracle)" ON)
    if(BENCH_RNNOISE_FP_CONTRACT)
        set_source_files_properties(${RNNOISE_FILES} PROPERTIES COMPILE_OPTIONS "-w;${RNNOISE_EXTRA_OPTS}")
    else()
        set_source_files_properties(${RNNOISE_FILES} PROPERTIES COMPILE_OPTIONS "-w;-ffp-contract=off;${RNNOISE_EXTRA_OPTS}")
        set_source_files_properties(${RNNOISE_PROC}  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    endif()
    # RAM placement is the linker's: RNNoise takes its state from newlib's heap
    # (rnnoise_create -> malloc; the 960-point kiss_fft twiddles likewise),
    # which lives between .bss and the MSP stack. These reservations make the
    # LINK fail if .data + .bss + heap + stack exceed RAM, rather than malloc
    # failing at the first CONFIG. Need, measured on the cross-compiler:
    # 28,820 B heap (DenoiseState 18,492 + GRU states 672 + kiss_fft 9,656);
    # stack from the -fstack-usage files, see RESULTS.md. Hex, as the .ld has it.
    # (F411 figures; the M7 build's frames are re-read from its .su files.)
    set(BENCH_RNNOISE_HEAP_BYTES  0x7400 CACHE STRING "RNNoise image: _Min_Heap_Size (hex)")
    set(BENCH_RNNOISE_STACK_BYTES 0xD800 CACHE STRING "RNNoise image: _Min_Stack_Size (hex)")
    target_link_options(${CMAKE_PROJECT_NAME} PRIVATE
        -Wl,--defsym=_Min_Heap_Size=${BENCH_RNNOISE_HEAP_BYTES}
        -Wl,--defsym=_Min_Stack_Size=${BENCH_RNNOISE_STACK_BYTES})
    message(STATUS "RNNoise v0.1.1: ON (heap ${BENCH_RNNOISE_HEAP_BYTES}, stack ${BENCH_RNNOISE_STACK_BYTES}, "
                   "fp-contract ${BENCH_RNNOISE_FP_CONTRACT})")
endif()

if(BENCH_WITH_DTLN)
    if(NOT EXISTS ${TFLM_TREE}/tensorflow/lite/micro/micro_interpreter.h)
        message(FATAL_ERROR "BENCH_WITH_DTLN: ${TFLM_TREE} is missing -- regenerate it "
                "(tflm_footprint/RESULTS.md 'Reproducing', at commit 64b3de8)")
    endif()
    # The generated tree is globbed deliberately: it is an output of
    # create_tflm_tree.py with no per-folder aggregates (unlike CMSIS-DSP), and
    # tflm_footprint/build.sh compiled it the same way; --gc-sections keeps only
    # the registered kernels, which tflm_footprint verified by symbol.
    file(GLOB_RECURSE TFLM_SRCS CONFIGURE_DEPENDS
         ${TFLM_TREE}/tensorflow/*.cc ${TFLM_TREE}/tensorflow/*.c
         ${TFLM_TREE}/signal/*.cc     ${TFLM_TREE}/signal/*.c
         ${TFLM_TREE}/third_party/*.cc ${TFLM_TREE}/third_party/*.c)
    list(FILTER TFLM_SRCS EXCLUDE REGEX "test_helpers|mock_|fake_|_test\\.cc$")
    add_library(tflm STATIC ${TFLM_SRCS})
    target_include_directories(tflm PUBLIC
        ${TFLM_TREE} ${TFLM_TREE}/third_party/flatbuffers/include
        ${TFLM_TREE}/third_party/gemmlowp ${TFLM_TREE}/third_party/ruy
        ${TFLM_TREE}/third_party/kissfft ${TFLM_TREE}/third_party/cmsis_nn
        ${TFLM_TREE}/third_party/cmsis_nn/Include
        ${TFLM_TREE}/third_party/cmsis/CMSIS/Core/Include ${TFLM_TREE}/signal/micro/kernels)
    # TFLM's required configuration, as in tflm_footprint/build.sh:
    # STRIP_ERROR_STRINGS is the smallest honest build and makes DebugLog a
    # no-op (no semihosting, harness rule 10).
    option(BENCH_DTLN_DEBUG "diagnostic: keep TFLM error strings, capture them in g_dtln_log" OFF)
    if(BENCH_DTLN_DEBUG)
        # PROJECT_GENERATION: micro_time.cc returns 0 instead of touching DWT/DCB (no device header)
        target_compile_definitions(tflm PUBLIC TF_LITE_STATIC_MEMORY TF_LITE_DISABLE_X86_NEON CMSIS_NN PROJECT_GENERATION)
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE DTLN_DEBUG_LOG)
    else()
        target_compile_definitions(tflm PUBLIC TF_LITE_STATIC_MEMORY TF_LITE_STRIP_ERROR_STRINGS
                                   TF_LITE_DISABLE_X86_NEON CMSIS_NN)
    endif()
    target_compile_options(tflm PRIVATE -w -fno-unwind-tables -fno-asynchronous-unwind-tables
                           $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti -fno-exceptions -fno-threadsafe-statics>)
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../dtln/port/dtln_proc.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../dtln/port/dtln_select.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../dtln/port/dtln_model_p020_i16pt.c)
    # The boundary arithmetic must round as the host's numpy did: no FMA
    # contraction in this file (task 15's one-LSB mechanism, proven on the Pi).
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../dtln/port/dtln_proc.cc
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../dtln/port)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE BENCH_WITH_DTLN)
    option(BENCH_DTLN_STAMPS "DTLN: per-stage DWT stage sums in g_dtln (not the measurement image)" OFF)
    if(BENCH_DTLN_STAMPS)
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE DTLN_STAMP_DWT)
    endif()
    # Arena provisioning is a measured variable: g_dtln reports the bytes the
    # allocator actually used, and the DTCM image is sized from that reading.
    set(BENCH_DTLN_ARENA1_BYTES 0 CACHE STRING "stage-1 arena bytes (0 = dtln_proc.cc default)")
    set(BENCH_DTLN_ARENA2_BYTES 0 CACHE STRING "stage-2 arena bytes (0 = dtln_proc.cc default)")
    if(BENCH_DTLN_ARENA1_BYTES GREATER 0)
        set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../dtln/port/dtln_proc.cc APPEND PROPERTY
                     COMPILE_DEFINITIONS DTLN_ARENA1_BYTES=${BENCH_DTLN_ARENA1_BYTES}u)
    endif()
    if(BENCH_DTLN_ARENA2_BYTES GREATER 0)
        set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../dtln/port/dtln_proc.cc APPEND PROPERTY
                     COMPILE_DEFINITIONS DTLN_ARENA2_BYTES=${BENCH_DTLN_ARENA2_BYTES}u)
    endif()
    target_link_libraries(${CMAKE_PROJECT_NAME} tflm)
    message(STATUS "DTLN 16x8 on TFLite Micro: ON (${TFLM_TREE})")
endif()
endif()

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined include paths
    bench
    ${CMSIS_DSP}/Include
    ${CMSIS_DSP}/PrivateInclude
)

# Add project symbols (macros)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined symbols
    #
    # ARM_MATH_LOOPUNROLL is NOT auto-defined anywhere in CMSIS-DSP 1.10 -- it
    # is documented in arm_math.h and set only by the library's own CMake, so a
    # hand-rolled build silently gets the non-unrolled f32 kernels and reports
    # the portable fallback as "what CMSIS-DSP is worth". bench_kernels.c
    # #errors if it is missing. ARM_MATH_DSP (the SMLAD/SXTB16 paths) is
    # derived automatically from __ARM_FEATURE_DSP, so it is asserted, not set.
    ARM_MATH_LOOPUNROLL

    # Selective FFT tables. Without these the twiddle and bit-reversal tables
    # for every supported transform size are linked -- ~110 KB on the F411 --
    # because arm_rfft_fast_init_f32 dispatches through a switch that
    # references them all, so --gc-sections cannot drop them.
    # ARM_FFT_ALLOW_TABLES is an OUTER guard (arm_common_tables.h:40): without
    # it ARM_DSP_CONFIG_TABLES suppresses every declaration regardless of which
    # ARM_TABLE_* are named, and the build fails on an undeclared twiddleCoef_*.
    ARM_DSP_CONFIG_TABLES
    ARM_FFT_ALLOW_TABLES
    ARM_TABLE_TWIDDLECOEF_F32_256
    ARM_TABLE_TWIDDLECOEF_F32_512
    ARM_TABLE_BITREVIDX_FLT_256
    ARM_TABLE_BITREVIDX_FLT_512
    ARM_TABLE_TWIDDLECOEF_RFFT_F32_512

    BENCH_ICACHE=$<BOOL:${BENCH_ICACHE}>
    BENCH_DCACHE=$<BOOL:${BENCH_DCACHE}>
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Add linked libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
    stm32cubemx

    # Add user defined libraries
    m           # expf/logf/sqrtf/tanhf in the classical-DSP kernels
)

# --- Post-build artifacts --------------------------------------------------
# .bin is what the DFU path uploads; .hex suits STM32CubeProgrammer.
# arm-none-eabi-size prints the flash/SRAM figures the budget table needs.
add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${CMAKE_PROJECT_NAME}>
            $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>/${CMAKE_PROJECT_NAME}.bin
    COMMAND ${CMAKE_OBJCOPY} -O ihex   $<TARGET_FILE:${CMAKE_PROJECT_NAME}>
            $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>/${CMAKE_PROJECT_NAME}.hex
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${CMAKE_PROJECT_NAME}>
    COMMENT "Generating .bin/.hex and reporting size"
)

# --- Build-flag assertion: LAST in the file, deliberately ----------------------
# Placed here rather than beside the set() above, because the defect this guards
# against is a LATER re-initialisation, not a wrong initial value. The original
# bug was set(CMAKE_C_FLAGS_RELEASE ...) sitting above project(); project() then
# reset it and the probe silently built at -Os. A check next to the set() would
# have passed and caught nothing. Anything that re-initialises the flags --
# project(), enable_language(), or add_subdirectory(cmake/stm32cubemx) pulling in
# the CubeMX toolchain file -- has run by this point.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    foreach(_lang C ASM)
        if(NOT CMAKE_${_lang}_FLAGS_RELEASE MATCHES "-O2")
            message(FATAL_ERROR
                "CMAKE_${_lang}_FLAGS_RELEASE is '${CMAKE_${_lang}_FLAGS_RELEASE}', expected "
                "-O2 -g3 -DNDEBUG. Something re-initialised it after the set() above. "
                "Benchmark numbers from an -Os build are NOT comparable to the F411 -- "
                "see 'Defect 1' in CLAUDE.md.")
        endif()
        if(NOT CMAKE_${_lang}_FLAGS_RELEASE MATCHES "NDEBUG")
            message(FATAL_ERROR
                "CMAKE_${_lang}_FLAGS_RELEASE lacks -DNDEBUG: assert() would be live "
                "inside timed regions.")
        endif()
    endforeach()
    # The flags that actually reach the probe's translation unit are what matter;
    # the variable only predicts them. Report it so a build log carries evidence.
    message(STATUS "Benchmark Release C flags: ${CMAKE_C_FLAGS_RELEASE}")
endif()
