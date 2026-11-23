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

# Set the project name
set(CMAKE_PROJECT_NAME benchmark_f411)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME})
message("Build type: " ${CMAKE_BUILD_TYPE})

# --- Benchmark build flags -------------------------------------------------
# CubeMX defaults Release to "-Os -g0". -Os optimises for size and costs
# throughput, and the throughput probe must be built with the same flags as the
# deployed firmware. -g3 changes no generated code (debug info lives in the ELF,
# not the .bin), so keeping it lets the Release build be debugged and profiled.
# These flags are a recorded output of the throughput probe -- if they change,
# every cycles/frame figure measured with the old ones is void.
# -DNDEBUG matters: the CubeMX toolchain file overrides CMAKE_C_FLAGS_RELEASE
# without it, so assert() would stay live inside timed regions. CMake's own
# default for ASM Release (-O3 -DNDEBUG) is left un-overridden by CubeMX too,
# which is why the assembler line is pinned here as well -- so that every
# translation unit in a Release build is compiled with the same, stated flags.
set(CMAKE_C_FLAGS_RELEASE     "-O2 -g3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE   "-O2 -g3 -DNDEBUG")
set(CMAKE_ASM_FLAGS_RELEASE   "-O2 -g3 -DNDEBUG")

# Enable CMake support for ASM and C languages
enable_language(C ASM)

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# --- CMSIS-DSP (vendored subset) -------------------------------------------
# Version 1.10.0 out of STM32Cube FW_F4 V1.28.3; see
# Middlewares/Third_Party/CMSIS-DSP/PROVENANCE.md for why it is vendored and
# why only these translation units are listed.
#
# The files are named individually and NOT globbed. CMSIS-DSP 1.10 ships both
# per-function sources (arm_dot_prod_f32.c) and per-folder aggregates that
# #include them all (BasicMathFunctions.c); a GLOB picks up both and the link
# fails on duplicate symbols.
set(CMSIS_DSP ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/Third_Party/CMSIS-DSP)

# --- TinyUSB (vendored) ----------------------------------------------------
# Release 0.21.0, commit dae3f9a366bfcddbf9dcf1b48d7500286a849539, upstream
# hathach/tinyusb. Vendored for the same reason as CMSIS-DSP: the memory
# figures this project reports are only reproducible against a pinned stack.
#
# ONE stack across all three MCU platforms is a locked decision -- with three
# vendor stacks the "flash/SRAM after USB" line item would be three different
# footprints reported as a platform difference.
#
# Only the device-side translation units are compiled: TinyUSB's src/ tree also
# carries the host stack, TypeC, and every device class, none of which this
# target links.
set(TINYUSB ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/Third_Party/tinyusb)

# The USB stack is switchable so the "flash/SRAM remaining after the USB stack"
# line item in the budget table is a SCRIPTED two-build delta rather than a
# figure someone remembered to write down:
#   cmake -B build/NoUSB -DBENCH_WITH_TINYUSB=OFF ... && cmake --build build/NoUSB
# Both builds are otherwise identical, which is the only way the difference is
# attributable to the stack.
option(BENCH_WITH_TINYUSB "Link the TinyUSB device stack" ON)

# The throughput probe (bench_kernels.c and its 64 KB weight pool) is what
# task 3 measured with; a DEPLOYED streaming image does not carry it. OFF
# gives the memory figure the budget's classical row should quote: the
# algorithm, the binding and the USB stack, nothing that exists to measure
# them. bench_dwt.c stays either way -- the ABF1 tick source reads the counter
# it starts.
option(BENCH_WITH_PROBE "Run the kernel throughput probe at boot" ON)

# RNNoise (task 9 -> 19, 2026-10-22; the block that builds it is below the
# binding). Declared here because it decides whether the classical arm is
# linked: measured on this part, RNNoise's as-shipped stack (51 KB) and heap
# (29 KB) do not fit beside the classical context (23,708 B) in 128 KB -- the
# link overflows by 1,504 B at the measured reservations. So the RNNoise image
# carries RNNoise + bypass + OLA by default and the classical arm stays in its
# own, already-measured image. WITH_CDSP=ON builds the combined image, which
# links only with reservations cut to the bare measured minimum (see RESULTS.md).
option(BENCH_WITH_RNNOISE "Carry Xiph RNNoise v0.1.1 (Release, BENCH_WITH_PROBE=OFF)" OFF)
option(BENCH_RNNOISE_WITH_CDSP "RNNoise image: also link the classical arm (does not fit at safe reservations)" OFF)
# Diagnostic variant (2026-10-30), the S3's "+F" twin -- never the measurement image.
#   ship   : as shipped (the measurement image; build id unsuffixed)
#   splibm : +F  log10/sqrt/floor -> log10f/sqrtf/floorf via rnnoise_sp_libm.h
#            (byte-identical to benchmark_s3/main/rnnoise_sp_libm.h), force-included
#            ahead of each vendored source with the vendored text unchanged. The
#            M4F has no double FPU, so each shipped call is a software double; the
#            variant measures what that costs per hop (no stage stamps on this
#            image) and is a quality-affecting change task 15 adjudicates against
#            the oracle. Keep its outputs in their own results directory.
#   spconst: +FS (build id suffix "+S") splibm plus -fsingle-precision-constant on the vendored files.
#            The +F link map still carries __aeabi_dadd/dmul/ddiv/dcmp from
#            denoise.c, rnn.c and pitch.c: Xiph's float code is written against
#            double literals (.5*x, 1e-15, floor(.5+25*x)), and C promotes each
#            such expression to a software double on this core. The three libm
#            calls are therefore not the whole of the software-double cost, and
#            this variant removes the rest at the compiler, vendored text still
#            unchanged. Quality-affecting in the same class as +F.
set(BENCH_RNNOISE_VARIANT "ship" CACHE STRING "RNNoise image variant: ship | splibm | spconst | nn8 | nn8fs")
# nn8 (+N, 2026-10-31): the three GRUs on CMSIS-NN arm_nn_vec_mat_mult_t_svdf_s8
# with int8 activations -- ../rnnoise/nn8/rnn_nn8.h. nn8fs (+M): the same on top
# of the spconst seam. Quality-affecting; the host twin is rnnoise_wav_nn8.
# -DBENCH_RNNOISE_STAMPS=ON: per-stage DWT sums inside rnnoise_process_frame(),
# the RNNoise twin of BENCH_CDSP_STAMPS (2026-10-30). Compiles
# rnnoise_denoise_stamped.c -- Xiph's denoise.c plus RN_STAMP lines and NOTHING
# else, which the configure step proves by diffing the two files -- with
# rnnoise_stamps.h force-included. A diagnostic build, never the measurement
# image; orthogonal to BENCH_RNNOISE_VARIANT. Read the sums with
# read_rnnoise_stamps.sh after a run.
option(BENCH_RNNOISE_STAMPS "RNNoise image: per-stage DWT stage sums (diagnostic)" OFF)
if(NOT BENCH_RNNOISE_VARIANT MATCHES "^(ship|splibm|spconst|nn8|nn8fs)$")
    message(FATAL_ERROR "BENCH_RNNOISE_VARIANT=${BENCH_RNNOISE_VARIANT}: must be ship, splibm, spconst, nn8 or nn8fs")
endif()
# BENCH_WITH_CDSP=OFF is the build-twice TWIN of the RNNoise image: binding +
# bypass + OLA and nothing else, so the RNNoise delta is attributable.
option(BENCH_WITH_CDSP "Link task 8's classical arm" ON)
if(BENCH_WITH_CDSP AND (NOT BENCH_WITH_RNNOISE OR BENCH_RNNOISE_WITH_CDSP))
    set(BENCH_CDSP_LINKED ON)
else()
    set(BENCH_CDSP_LINKED OFF)
endif()

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user sources here
    bench/bench_dwt.c

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

# --- ABF1 streaming harness (task 7) ----------------------------------------
# abf/ holds COPIES of the normative sources in ../harness, the same way
# bench/bench.h is a per-tree copy: the three MCU tiers and the Pi must carry
# byte-identical framing or a protocol bug becomes a platform difference. The
# rule used to be "cksum them before trusting a cross-platform number"; here it
# is a build guard instead, in the spirit of the ARM_MATH_LOOPUNROLL #error --
# when the normative tree is present, a drifted copy stops the configure.
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

# HELLO.build_id. A configure-time stamp, recorded in the cache: with +-0.7 %
# re-link sensitivity measured on this part, "which build produced this" is a
# measurement input. The runtime image CRC in the same HELLO is the hard
# identity; this is the human-readable one.
string(TIMESTAMP ABF_BUILD_STAMP "%Y%m%dT%H%M" UTC)
set(ABF_BUILD_ID "${CMAKE_BUILD_TYPE} ${ABF_BUILD_STAMP}")
if(BENCH_WITH_RNNOISE AND BENCH_RNNOISE_VARIANT STREQUAL "splibm")
    set(ABF_BUILD_ID "${ABF_BUILD_ID}+F")   # the host names the variant from HELLO
elseif(BENCH_WITH_RNNOISE AND BENCH_RNNOISE_VARIANT STREQUAL "spconst")
    # "+S", not "+FS": the id is 23 characters at most (ABF_BUILD_ID_LEN 24 with
    # the NUL) and "Release YYYYMMDDTHHMM" already uses 21, so "+FS" was silently
    # truncated to "+F" on 2026-10-30 and the +FS image announced itself as the +F
    # one. The CRC told them apart; the label did not. Guarded below.
    set(ABF_BUILD_ID "${ABF_BUILD_ID}+S")
elseif(BENCH_WITH_RNNOISE AND BENCH_RNNOISE_VARIANT STREQUAL "nn8")
    set(ABF_BUILD_ID "${ABF_BUILD_ID}+N")   # CMSIS-NN GRUs
elseif(BENCH_WITH_RNNOISE AND BENCH_RNNOISE_VARIANT STREQUAL "nn8fs")
    set(ABF_BUILD_ID "${ABF_BUILD_ID}+M")   # +N on top of +FS (one letter: 23-char limit)
endif()
if(BENCH_WITH_RNNOISE AND BENCH_RNNOISE_STAMPS)
    # Stamped builds: +T = ship, +U = +F, +V = +FS, +X = +N, +Y = +M (one letter
    # each; the id has room for exactly two characters and the variant letter
    # already uses them when combined). The image CRC remains the hard identity.
    if(BENCH_RNNOISE_VARIANT STREQUAL "ship")
        set(ABF_BUILD_ID "${ABF_BUILD_ID}+T")
    elseif(BENCH_RNNOISE_VARIANT STREQUAL "splibm")
        string(REPLACE "+F" "+U" ABF_BUILD_ID "${ABF_BUILD_ID}")
    elseif(BENCH_RNNOISE_VARIANT STREQUAL "spconst")
        string(REPLACE "+S" "+V" ABF_BUILD_ID "${ABF_BUILD_ID}")
    elseif(BENCH_RNNOISE_VARIANT STREQUAL "nn8")
        string(REPLACE "+N" "+X" ABF_BUILD_ID "${ABF_BUILD_ID}")
    else()
        string(REPLACE "+M" "+Y" ABF_BUILD_ID "${ABF_BUILD_ID}")
    endif()
endif()
string(LENGTH "${ABF_BUILD_ID}" _abf_id_len)
if(_abf_id_len GREATER 23)
    message(FATAL_ERROR "ABF_BUILD_ID '${ABF_BUILD_ID}' is ${_abf_id_len} characters; HELLO carries 23. "
            "A truncated id mislabels the image -- shorten the suffix.")
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
if(BENCH_CDSP_LINKED)
    # Task 8's classical arm, compiled from the normative tree UNMODIFIED --
    # not copied, because task 15 compares this port against the host oracle
    # built from the same file, and a copy would reintroduce the drift the
    # abf/ guard exists to catch. Only the FFT seam is this tier's own.
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../classical/cdsp.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../classical/cdsp_select.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../classical/cdsp_fft_cmsis.c)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE BENCH_WITH_CDSP)
else()
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE RNNOISE_SELECT_NO_CDSP)
    message(STATUS "classical arm: NOT linked (RNNoise image, or its build-twice twin)")
endif()
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE usb abf ${TINYUSB}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../classical)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    BENCH_WITH_TINYUSB
    # The reference processors' frame cap. 4096 (the host default, for the
    # MUSDB tier) would cost 56 KB of .bss here; the locked configuration,
    # RNNoise's 480 and DTLN's 512 all fit in 512. A larger CONFIG is NAKed.
    ABF_PROC_MAX_FRAME=512u
    # The classical context's frame cap, same reasoning: ~23 KB at 512 against
    # ~46 KB at the host's 1024. Only the 512-point real FFT tables are linked
    # (ARM_TABLE_* above), so an 8 kHz / 256-point CONFIG is NAKed on this tier.
    CDSP_FFT_MAX=512u
    ABF_BUILD_ID="${ABF_BUILD_ID}"
)
# -DBENCH_CDSP_STAMPS=ON: per-stage DWT sums inside cdsp_process(), read off
# the board with the debugger (g_cdsp_stage[], g_cdsp_stage_n). A diagnostic
# build, not the measurement image.
option(BENCH_CDSP_STAMPS "classical arm: per-stage DWT stage sums" OFF)
if(BENCH_CDSP_STAMPS)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE CDSP_STAMP_DWT)
endif()

# --- RNNoise, as shipped (task 9 -> 19, 2026-10-22) ---------------------------
# Xiph v0.1.1 from ../rnnoise/xiph-v0.1.1 compiled UNMODIFIED (the host oracle
# is built from the same files, task 15), behind the ABF1 processor the host
# oracle already uses (../rnnoise/rnnoise_proc.c), selected by CONFIG.algo_id =
# ABF_ALGO_RNNOISE. Geometry is the weights' own: 48 kHz, frame 960 / hop 480.
# The transform is Xiph's mixed-radix kiss_fft as shipped (480 is not a power
# of two, so neither CMSIS rfft size applies; this is what the budget costed).
# Weights are int8 in .rodata (rnn_data.c, 87,503 B). rnn_reader.c (model
# from a FILE*, fscanf) is deliberately not compiled.
#
# OFF by default (declared above) so every recorded image re-emits unchanged.
# The RNNoise measurement image is Release + BENCH_WITH_PROBE=OFF +
# BENCH_WITH_RNNOISE=ON, classical arm not linked.
if(BENCH_WITH_RNNOISE)
    if(NOT BENCH_WITH_TINYUSB)
        message(FATAL_ERROR "BENCH_WITH_RNNOISE needs the ABF1 binding (BENCH_WITH_TINYUSB=ON)")
    endif()
    if(BENCH_WITH_PROBE)
        # The probe's 64 KB pool plus RNNoise's ~29 KB heap and its stack do not
        # fit beside the binding in 128 KB SRAM; malloc would fail at the first CONFIG
        # and the run would be NAKed, not refused at build time. Break the build.
        message(FATAL_ERROR "BENCH_WITH_RNNOISE and BENCH_WITH_PROBE together exceed 128 KB SRAM: "
                "configure with -DBENCH_WITH_PROBE=OFF")
    endif()
    set(RNNOISE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../rnnoise/xiph-v0.1.1/src)
    set(RNNOISE_FILES ${RNNOISE_SRC}/denoise.c ${RNNOISE_SRC}/kiss_fft.c ${RNNOISE_SRC}/pitch.c
                      ${RNNOISE_SRC}/celt_lpc.c ${RNNOISE_SRC}/rnn.c ${RNNOISE_SRC}/rnn_data.c)
    if(BENCH_RNNOISE_STAMPS)
        set(RNNOISE_STAMPED ${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_denoise_stamped.c)
        # The stamped copy must be the vendored file plus RN_STAMP lines and
        # nothing else, or the stage sums decompose a different program than
        # the one measured. Checked here, every configure, before anything
        # compiles: any divergence other than an RN_STAMP line breaks the build.
        execute_process(COMMAND diff -I RN_STAMP ${RNNOISE_SRC}/denoise.c ${RNNOISE_STAMPED}
                        RESULT_VARIABLE _rn_stamp_diff OUTPUT_VARIABLE _rn_stamp_out ERROR_VARIABLE _rn_stamp_out)
        if(NOT _rn_stamp_diff EQUAL 0)
            message(FATAL_ERROR "rnnoise_denoise_stamped.c diverges from the vendored denoise.c beyond RN_STAMP lines:\n${_rn_stamp_out}")
        endif()
        list(REMOVE_ITEM RNNOISE_FILES ${RNNOISE_SRC}/denoise.c)
        list(APPEND RNNOISE_FILES ${RNNOISE_STAMPED})
        target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_stamps.c)
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE BENCH_RNNOISE_STAMPS)
        message(STATUS "RNNoise stamps: ON (rnnoise_denoise_stamped.c verified vendored + RN_STAMP lines only; diagnostic, not the measurement image)")
    endif()
    if(BENCH_RNNOISE_VARIANT MATCHES "^nn8")
        # The +N variant: rnn.c is replaced by its derived twin (the forwarder)
        # plus the CMSIS-NN GRU, the re-laid weights and the one kernel it
        # calls. nn8_tool.py check runs here, every configure, and refuses the
        # build if either derived file differs from a fresh generation or the
        # activation helpers are not rnn.c's verbatim.
        set(RNNOISE_NN8 ${CMAKE_CURRENT_SOURCE_DIR}/../rnnoise/nn8)
        set(CMSIS_NN ${CMAKE_CURRENT_SOURCE_DIR}/../tflm_tree_64b3de8/third_party/cmsis_nn)   # V.6.2.0
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
        message(STATUS "RNNoise variant +N: GRUs on CMSIS-NN svdf_s8 (${_nn8_out})")
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
    set(RNNOISE_EXTRA_OPTS "")
    if(BENCH_RNNOISE_VARIANT STREQUAL "splibm")
        set(RNNOISE_EXTRA_OPTS "-include;${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_sp_libm.h")
        message(STATUS "RNNoise variant +F: single-precision libm (diagnostic, not the measurement image)")
    elseif(BENCH_RNNOISE_VARIANT STREQUAL "spconst")
        set(RNNOISE_EXTRA_OPTS "-include;${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_sp_libm.h;-fsingle-precision-constant")
        message(STATUS "RNNoise variant +FS: single-precision libm AND constants (diagnostic, not the measurement image)")
    elseif(BENCH_RNNOISE_VARIANT STREQUAL "nn8fs")
        set(RNNOISE_EXTRA_OPTS "-include;${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_sp_libm.h;-fsingle-precision-constant")
        message(STATUS "RNNoise variant +M: +N on top of the +FS seam")
    endif()
    if(BENCH_RNNOISE_VARIANT MATCHES "^nn8")
        # The nn8 files sit outside Xiph's src/ and include "tansig_table.h" etc. by bare name.
        set(RNNOISE_EXTRA_OPTS "${RNNOISE_EXTRA_OPTS};-I${RNNOISE_SRC}")
    endif()
    if(BENCH_RNNOISE_STAMPS)
        # Appended, so a variant's -include (rnnoise_sp_libm.h) still applies to
        # the stamped file; the header itself defines nothing the variant touches.
        # -I: the stamped copy sits outside Xiph's src/ and includes "kiss_fft.h" etc. by bare name.
        set(RNNOISE_EXTRA_OPTS "${RNNOISE_EXTRA_OPTS};-I${RNNOISE_SRC};-include;${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_stamps.h")
    endif()
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
    # 28,820 B heap (DenoiseState 18,492 + GRU states 672 + kiss_fft 9,656) +
    # newlib chunk headers; stack 50,944 B on the deepest -fstack-usage chain
    # (rnnoise_process_frame 35,400 -> forward_transform 15,376 -> rnn_fft_impl
    # 168) plus the binding's loop and one USB exception frame. Reserved with
    # ~0.8 KB / ~4 KB to spare. Hex, as the .ld has it.
    set(BENCH_RNNOISE_HEAP_BYTES  0x7400 CACHE STRING "RNNoise image: _Min_Heap_Size (hex)")
    set(BENCH_RNNOISE_STACK_BYTES 0xD800 CACHE STRING "RNNoise image: _Min_Stack_Size (hex)")
    target_link_options(${CMAKE_PROJECT_NAME} PRIVATE
        -Wl,--defsym=_Min_Heap_Size=${BENCH_RNNOISE_HEAP_BYTES}
        -Wl,--defsym=_Min_Stack_Size=${BENCH_RNNOISE_STACK_BYTES})
    message(STATUS "RNNoise v0.1.1: ON (heap ${BENCH_RNNOISE_HEAP_BYTES}, stack ${BENCH_RNNOISE_STACK_BYTES}, "
                   "fp-contract ${BENCH_RNNOISE_FP_CONTRACT})")
endif()
endif()

if(BENCH_WITH_PROBE)
target_sources(${CMAKE_PROJECT_NAME} PRIVATE bench/bench_kernels.c bench/bench_itm.c)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE BENCH_WITH_PROBE)
option(BENCH_PRUNE "probe: Arm B (c) pruning-cost kernels (bench_prune.c), appended last" OFF)
if(BENCH_PRUNE)
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE bench/bench_prune.c)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE BENCH_PRUNE)
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
    # is documented in arm_math.h and set by the library's own CMake. Building
    # the sources by hand without it silently yields the non-unrolled f32
    # kernels, i.e. a measurement of the portable fallback reported as "what
    # CMSIS-DSP is worth". bench_kernels.c #errors if it is missing.
    #
    # ARM_MATH_DSP (the SMLAD/SXTB16 paths) is derived automatically from
    # __ARM_FEATURE_DSP, which GCC sets for -mcpu=cortex-m4, so it is asserted
    # rather than set here.
    ARM_MATH_LOOPUNROLL

    # --- selective FFT tables ---------------------------------------------
    # Without these, linking CMSIS-DSP costs 147,772 B of flash on this
    # project, of which ~105 KB is twiddle and bit-reversal tables for
    # transform sizes never used (4096, 2048, 1024). --gc-sections cannot drop
    # them: arm_rfft_fast_init_f32 dispatches through a switch that references
    # every supported length, so every table is reachable from the init call.
    #
    # ARM_DSP_CONFIG_TABLES switches arm_common_tables.c / arm_const_structs.c
    # from "compile everything" to "compile what is named", which is the
    # vendor's own documented mechanism. Named here: the 256- and 512-point
    # complex transforms of the locked 16 kHz configuration, plus the 512-point
    # real transform (which internally runs a 256-point complex FFT, hence the
    # _F32_256 pair rather than _F32_512).
    #
    # This is a REPORTED figure, not a tidy-up: the untrimmed footprint is a
    # third of the F411's flash and would dominate its budget-table row for a
    # reason that has nothing to do with the algorithm.
    # ARM_FFT_ALLOW_TABLES is an OUTER guard (arm_common_tables.h:40) around the
    # whole per-table block. Without it ARM_DSP_CONFIG_TABLES suppresses every
    # FFT table declaration regardless of which ARM_TABLE_* are named, and the
    # build fails on an undeclared twiddleCoef_*.
    ARM_DSP_CONFIG_TABLES
    ARM_FFT_ALLOW_TABLES
    ARM_TABLE_TWIDDLECOEF_F32_256
    ARM_TABLE_TWIDDLECOEF_F32_512
    ARM_TABLE_BITREVIDX_FLT_256
    ARM_TABLE_BITREVIDX_FLT_512
    ARM_TABLE_TWIDDLECOEF_RFFT_F32_512
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
# .bin is what the DFU / USB-bootloader flash path uploads; .hex is convenient
# for STM32CubeProgrammer. arm-none-eabi-size prints the flash/SRAM figures that
# feed the constraint budget table (the linker also emits --print-memory-usage).
add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    # The .bin is also what HELLO.image_crc32 is checked against, so FLASH THE
    # .bin (write_image <bin> 0x08000000), not the .elf: the image in flash is
    # then the .bin by construction, alignment gaps included. (--gap-fill is
    # not the answer -- with this linker script it makes objcopy span the RAM
    # sections too and emit a 384 MB file.)
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${CMAKE_PROJECT_NAME}>
            $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>/${CMAKE_PROJECT_NAME}.bin
    COMMAND ${CMAKE_OBJCOPY} -O ihex   $<TARGET_FILE:${CMAKE_PROJECT_NAME}>
            $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>/${CMAKE_PROJECT_NAME}.hex
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${CMAKE_PROJECT_NAME}>
    COMMENT "Generating .bin/.hex and reporting size"
)
