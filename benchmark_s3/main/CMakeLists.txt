# --- Working-set placement: internal SRAM vs octal PSRAM ---------------------
# The counterpart of the H723's BENCH_RAM=DTCM|AXI. Default is internal, which
# is the only configuration in which the numbers mean anything; the PSRAM row is
# the comparison, not the default. g_plat.pool_addr witnesses where the pool
# actually landed, so the knob is never trusted on its own.
if(BENCH_MEM EQUAL 1)
    set(BENCH_LDFRAGMENTS "linker_psram.lf")
    message(STATUS "benchmark_s3: working set -> EXTERNAL PSRAM")
else()
    set(BENCH_LDFRAGMENTS "")
    message(STATUS "benchmark_s3: working set -> internal SRAM")
endif()

# --- TinyUSB CDC ------------------------------------------------------------
# OFF/ON is what makes the flash/SRAM line item a SCRIPTED two-build delta
# rather than a remembered figure -- the same requirement the STM32 toolchain
# decision imposed, and the reason benchmark_f411 has BENCH_WITH_TINYUSB.
if(NOT DEFINED BENCH_WITH_TINYUSB)
    set(BENCH_WITH_TINYUSB 1)
endif()
if(BENCH_WITH_TINYUSB)
    set(BENCH_USB_SRCS "usb_glue.c")
    set(BENCH_USB_REQ  espressif__esp_tinyusb)
    message(STATUS "benchmark_s3: TinyUSB CDC LINKED")
else()
    set(BENCH_USB_SRCS "")
    set(BENCH_USB_REQ  "")
    message(STATUS "benchmark_s3: TinyUSB CDC NOT linked")
endif()

# --- The probe is optional: the streaming measurement image carries none ----
if(NOT DEFINED BENCH_WITH_PROBE)
    set(BENCH_WITH_PROBE 1)
endif()
if(BENCH_WITH_PROBE)
    set(BENCH_PROBE_SRCS "bench/bench_kernels.c" "bench/bench_report.c")
else()
    set(BENCH_PROBE_SRCS "")
endif()

# --- ABF1 streaming harness (task 7) + task 8's classical arm ----------------
# abf/ holds COPIES of the normative sources in ../../harness, like the STM32
# trees; the same configure-time SHA-256 guard stops a drifted copy. The tick
# source is abf_plat_ccount.c here (CCOUNT, per-core) instead of abf_plat_dwt.c.
# cdsp.c and cdsp_select.c compile from ../../classical UNMODIFIED; the seam
# file cdsp_fft_espdsp.c is this tier's own.
set(ABF_NORMATIVE ${COMPONENT_DIR}/../../harness)
set(ABF_FILES abf_frame.h abf_frame.c abf_cdc.h abf_cdc.c abf_cdc_tinyusb.c
              abf_plat_ccount.c abf_proc.h abf_proc.c)
if(NOT CMAKE_BUILD_EARLY_EXPANSION AND EXISTS ${ABF_NORMATIVE}/abf_frame.h)
    foreach(f ${ABF_FILES})
        file(SHA256 ${COMPONENT_DIR}/abf/${f} _ours)
        file(SHA256 ${ABF_NORMATIVE}/${f} _norm)
        if(NOT _ours STREQUAL _norm)
            message(FATAL_ERROR "abf/${f} differs from ${ABF_NORMATIVE}/${f}: "
                    "the ABF1 sources must be byte-identical on every tier.")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                     ${COMPONENT_DIR}/abf/${f} ${ABF_NORMATIVE}/${f})
    endforeach()
    message(STATUS "benchmark_s3: abf/ verified identical to ../../harness")
endif()
string(TIMESTAMP ABF_BUILD_STAMP "%Y%m%dT%H%M" UTC)
set(ABF_BUILD_ID "Release ${ABF_BUILD_STAMP}")
set(CLASSICAL ${COMPONENT_DIR}/../../classical)
# BENCH_WITH_CDSP=0 drops the classical arm (binding + bypass + OLA only): the
# build-twice twin of the RNNoise image below, as on the F411/H723 (2026-10-26).
if(NOT DEFINED BENCH_WITH_CDSP)
    set(BENCH_WITH_CDSP 1)
endif()
if(BENCH_WITH_TINYUSB)
    set(BENCH_ABF_SRCS "abf/abf_frame.c" "abf/abf_cdc.c" "abf/abf_cdc_tinyusb.c"
                       "abf/abf_plat_ccount.c" "abf/abf_proc.c")
    if(BENCH_WITH_CDSP)
        list(APPEND BENCH_ABF_SRCS "${CLASSICAL}/cdsp.c" "${CLASSICAL}/cdsp_select.c"
                                   "${CLASSICAL}/cdsp_fft_espdsp.c")
        message(STATUS "benchmark_s3: classical arm LINKED")
    else()
        message(STATUS "benchmark_s3: classical arm NOT linked (RNNoise image, or its build-twice twin)")
    endif()
else()
    set(BENCH_ABF_SRCS "")
endif()

# --- RNNoise, as shipped (task 9 -> 17, 2026-10-28; the F411/H723 block's twin)
# Xiph v0.1.1 from ../../rnnoise/xiph-v0.1.1 compiled UNMODIFIED (the host
# oracle is built from the same files, task 15) behind the oracle's own ABF1
# processor (../../rnnoise/rnnoise_proc.c), selected by CONFIG.algo_id =
# ABF_ALGO_RNNOISE. Geometry is the weights' own: 48 kHz, frame 960 / hop 480.
# The transform is Xiph's mixed-radix kiss_fft as shipped (480 is not a power
# of two, so dsps_fft2r does not apply; this is what the budget costed at the
# plain-C rate). Weights are int8 .rodata (rnn_data.c). rnn_reader.c (model
# from a FILE*) is deliberately not compiled.
#
# Image composition matches the two Cortex-M RNNoise images: RNNoise + bypass +
# OLA, no classical arm, a classical id NAKed (RNNOISE_SELECT_NO_CDSP). Here it
# is a choice, not a RAM constraint (DIRAM would hold both); it keeps the three
# MCU RNNoise images the same thing. BENCH_RNNOISE_WITH_CDSP=1 asks for the
# combined image; BENCH_WITH_CDSP=0 without RNNoise is the build-twice twin.
#
# Two S3-specific facts, both handled in usb_glue.c, not here:
#   * rnnoise_create() mallocs its 18,492 B DenoiseState, and with
#     CONFIG_SPIRAM_USE_MALLOC any malloc above CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL
#     (16,384) is placed in PSRAM FIRST -- the working set would silently leave
#     internal SRAM (1.5x and 60-90x jitter on this tier, RESULTS.md). The glue
#     raises the limit at runtime and witnesses the PSRAM free size unchanged.
#   * mallinfo() is a stub returning zeros on ESP-IDF, so the shared heap
#     witness g_rnnoise_heap_bytes reads 0 here by construction; the internal
#     heap delta printed on the console is this tier's witness.
# The frame is a FreeRTOS task stack, not a linker reservation: the deepest
# -fstack-usage chain on the Cortex-M builds was 50,944 B; BENCH_ABF_STACK_BYTES
# sizes the ABF task, the task's high-water mark is printed as the witness, and
# CONFIG_FREERTOS_CHECK_STACKOVERFLOW_CANARY aborts loudly rather than corrupting.
if(NOT DEFINED BENCH_WITH_RNNOISE)
    set(BENCH_WITH_RNNOISE 0)
endif()
if(NOT DEFINED BENCH_RNNOISE_WITH_CDSP)
    set(BENCH_RNNOISE_WITH_CDSP 0)
endif()
if(NOT DEFINED BENCH_RNNOISE_FP_CONTRACT)
    set(BENCH_RNNOISE_FP_CONTRACT 1)
endif()
if(NOT DEFINED BENCH_ABF_STACK_BYTES)
    if(BENCH_WITH_RNNOISE)
        set(BENCH_ABF_STACK_BYTES 98304)
    else()
        set(BENCH_ABF_STACK_BYTES 8192)
    endif()
endif()
set(RNNOISE_DIR ${COMPONENT_DIR}/../../rnnoise)
set(BENCH_RNNOISE_SRCS "")
if(BENCH_WITH_RNNOISE)
    if(NOT BENCH_WITH_TINYUSB)
        message(FATAL_ERROR "BENCH_WITH_RNNOISE needs the ABF1 binding (BENCH_WITH_TINYUSB=1)")
    endif()
    if(BENCH_WITH_PROBE)
        message(FATAL_ERROR "BENCH_WITH_RNNOISE with the probe is not a measurement image: "
                "configure with -DBENCH_WITH_PROBE=0 (as the F411/H723 RNNoise images are)")
    endif()
    if(BENCH_WITH_DTLN)
        message(FATAL_ERROR "BENCH_WITH_RNNOISE and BENCH_WITH_DTLN in one S3 image is not wired "
                "(no chained selector here); build one network per image")
    endif()
    if(BENCH_WITH_CDSP AND NOT BENCH_RNNOISE_WITH_CDSP)
        message(FATAL_ERROR "BENCH_WITH_RNNOISE: pass -DBENCH_WITH_CDSP=0 (the RNNoise-only image, "
                "as on the F411/H723) or -DBENCH_RNNOISE_WITH_CDSP=1 for the combined image")
    endif()
    set(RNNOISE_SRC ${RNNOISE_DIR}/xiph-v0.1.1/src)
    set(RNNOISE_FILES ${RNNOISE_SRC}/denoise.c ${RNNOISE_SRC}/kiss_fft.c ${RNNOISE_SRC}/pitch.c
                      ${RNNOISE_SRC}/celt_lpc.c ${RNNOISE_SRC}/rnn.c ${RNNOISE_SRC}/rnn_data.c)
    set(BENCH_RNNOISE_SRCS ${RNNOISE_FILES} ${RNNOISE_DIR}/rnnoise_proc.c ${RNNOISE_DIR}/rnnoise_select.c)
    # Diagnostic variants (2026-10-28), never the measurement image. The
    # as-shipped smoke came in at 3.14 M cycles/hop, 6.4x the budget's cost and
    # every hop over the 10 ms deadline; no image carries stage stamps, so the
    # two candidate terms are measured by removing them one at a time:
    #   ship   : as shipped (the measurement image; build id unsuffixed)
    #   splibm : +F  log10/sqrt/floor -> log10f/sqrtf/floorf (rnnoise_sp_libm.h,
    #            force-included; vendored text unchanged). LX7 has no double FPU.
    #   ramw   : +W  the 88,399 B of weights deep-copied into internal SRAM at
    #            the first rnnoise_create (rnnoise_relocate.c, --wrap at the link)
    #   both   : +FW
    # The host names the kernel from the HELLO build id; keep each variant's
    # outputs in its own directory, since the CSV name does not carry it.
    if(NOT DEFINED BENCH_RNNOISE_VARIANT)
        set(BENCH_RNNOISE_VARIANT "ship")
    endif()
    set(RNNOISE_EXTRA_OPTS "")
    if(BENCH_RNNOISE_VARIANT STREQUAL "ship")
        message(STATUS "benchmark_s3: RNNoise variant = as shipped (measurement image)")
    elseif(BENCH_RNNOISE_VARIANT STREQUAL "splibm" OR BENCH_RNNOISE_VARIANT STREQUAL "both")
        set(RNNOISE_EXTRA_OPTS "-include;${COMPONENT_DIR}/rnnoise_sp_libm.h")
        set(ABF_BUILD_ID "${ABF_BUILD_ID}+F")
        message(STATUS "benchmark_s3: RNNoise variant +F: single-precision libm")
    endif()
    if(BENCH_RNNOISE_VARIANT STREQUAL "ramw" OR BENCH_RNNOISE_VARIANT STREQUAL "both")
        list(APPEND BENCH_RNNOISE_SRCS "rnnoise_relocate.c")
        set(ABF_BUILD_ID "${ABF_BUILD_ID}+W")
        message(STATUS "benchmark_s3: RNNoise variant +W: weights copied to internal SRAM")
    endif()
    if(NOT BENCH_RNNOISE_VARIANT MATCHES "^(ship|splibm|ramw|both)$")
        message(FATAL_ERROR "BENCH_RNNOISE_VARIANT=${BENCH_RNNOISE_VARIANT}: must be ship, splibm, ramw or both")
    endif()
    # -DBENCH_RNNOISE_STAMPS=1 (2026-11-02): per-stage CCOUNT sums inside
    # rnnoise_process_frame(), the F411/H723 stamped build ported. The stamped
    # copy and the storage file are byte-identical to the F411's (asserted), and
    # the stamped copy must be the vendored denoise.c plus RN_STAMP lines and
    # nothing else (diff -I RN_STAMP), or the sums decompose a different program
    # than the one measured. Only rnnoise_stamps.h differs (CCOUNT for DWT).
    # Diagnostic image, never the measurement image; orthogonal to the variant.
    # Sums go to the UART console once a run has been quiet for 1 s (main.c).
    if(NOT DEFINED BENCH_RNNOISE_STAMPS)
        set(BENCH_RNNOISE_STAMPS 0)
    endif()
    if(BENCH_RNNOISE_STAMPS)
        set(RNNOISE_STAMPED ${COMPONENT_DIR}/rnnoise_denoise_stamped.c)
        if(NOT CMAKE_BUILD_EARLY_EXPANSION)
            foreach(_f rnnoise_stamps.c rnnoise_denoise_stamped.c)
                file(SHA256 ${COMPONENT_DIR}/${_f} _h_here)
                file(SHA256 ${COMPONENT_DIR}/../../benchmark_f411/${_f} _h_f411)
                if(NOT _h_here STREQUAL _h_f411)
                    message(FATAL_ERROR "main/${_f} is not byte-identical to benchmark_f411/${_f}")
                endif()
            endforeach()
            execute_process(COMMAND diff -I RN_STAMP ${RNNOISE_SRC}/denoise.c ${RNNOISE_STAMPED}
                            RESULT_VARIABLE _rn_stamp_diff OUTPUT_VARIABLE _rn_stamp_out ERROR_VARIABLE _rn_stamp_out)
            if(NOT _rn_stamp_diff EQUAL 0)
                message(FATAL_ERROR "rnnoise_denoise_stamped.c diverges from the vendored denoise.c beyond RN_STAMP lines:\n${_rn_stamp_out}")
            endif()
        endif()
        list(REMOVE_ITEM RNNOISE_FILES ${RNNOISE_SRC}/denoise.c)
        list(APPEND RNNOISE_FILES ${RNNOISE_STAMPED})
        list(REMOVE_ITEM BENCH_RNNOISE_SRCS ${RNNOISE_SRC}/denoise.c)
        list(APPEND BENCH_RNNOISE_SRCS ${RNNOISE_STAMPED} "rnnoise_stamps.c")
        # -I: the stamped copy sits outside Xiph's src/ and includes "kiss_fft.h" etc. by bare name.
        # Appended, so a variant's -include (rnnoise_sp_libm.h) still applies to the stamped file.
        set(RNNOISE_EXTRA_OPTS "${RNNOISE_EXTRA_OPTS};-I${RNNOISE_SRC};-include;${COMPONENT_DIR}/rnnoise_stamps.h")
        message(STATUS "benchmark_s3: RNNoise stamps ON (CCOUNT; stamped copy = F411's = vendored + RN_STAMP lines; diagnostic)")
    endif()
    message(STATUS "benchmark_s3: RNNoise v0.1.1 LINKED (fp-contract ${BENCH_RNNOISE_FP_CONTRACT}, "
                   "ABF task stack ${BENCH_ABF_STACK_BYTES} B, classical ${BENCH_WITH_CDSP})")
endif()

# --- DTLN 16x8 on TFLite Micro (2026-10-21; the H723 port's twin) -----------
# Same processor, selector and per-tensor-dense artifact as the H723
# (../../dtln/port); the runtime is the tflm component (reference kernels).
# -DBENCH_WITH_DTLN=0 gives the classical-only twin for the flash/RAM delta.
if(NOT DEFINED BENCH_WITH_DTLN)
    set(BENCH_WITH_DTLN 0)
endif()
set(DTLN_PORT ${COMPONENT_DIR}/../../dtln/port)
if(BENCH_WITH_DTLN AND BENCH_WITH_TINYUSB)
    set(BENCH_DTLN_SRCS "${DTLN_PORT}/dtln_proc.cc" "${DTLN_PORT}/dtln_select.c"
                        "${DTLN_PORT}/dtln_model_p020_i16pt.c" "dtln_relocate.c")
    set(BENCH_DTLN_REQ tflm)
    message(STATUS "benchmark_s3: DTLN 16x8 on TFLite Micro LINKED")
else()
    set(BENCH_DTLN_SRCS "")
    set(BENCH_DTLN_REQ "")
endif()

# --- DTLN matvec kernel on this tier (2026-10-22) ----------------------------
# ref  : TFLite Micro's portable C++ reference (the 2026-10-21 measurement
#        image, RTF 4.60) -- the default, so existing builds are unchanged
# pie  : dtln_fc_pie.{h,S} + _core.c + _tflm.cc -- the PIE assembly dot
#        product (int16 x int8 -> 40-bit ACCX) behind FULLY_CONNECTED and
#        1x1 CONV_2D, registered by dtln_proc.cc under DTLN_FC_S3_PIE
# piec : the same glue and epilogue with its plain-C dot product, to separate
#        what the ISA buys from what leaving the int64 reference loop buys
# The HELLO build_id carries "+P" / "+C" so the host log names the kernel;
# g_dtln.fc_kernel witnesses it on the device. Any other value is an error.
if(NOT DEFINED BENCH_DTLN_FC)
    set(BENCH_DTLN_FC "ref")
endif()
set(BENCH_FC_SRCS "")
set(BENCH_FC_DEFS "")
if(BENCH_WITH_DTLN AND BENCH_WITH_TINYUSB)
    if(BENCH_DTLN_FC STREQUAL "ref")
        message(STATUS "benchmark_s3: DTLN matvec = TFLM reference C++")
    elseif(BENCH_DTLN_FC STREQUAL "pie" OR BENCH_DTLN_FC STREQUAL "piec")
        # The PIE glue reads the stage-2 basis rows in place, which needs them
        # 16-aligned where they sit: link the --align-inplace emission of the
        # same flatbuffers (12 leading zero bytes on stage 2; byte-identical
        # otherwise, SHA-256 in its header). The ref build keeps the plain
        # array, so the recorded measurement image is unchanged.
        list(REMOVE_ITEM BENCH_DTLN_SRCS "${DTLN_PORT}/dtln_model_p020_i16pt.c")
        list(APPEND BENCH_DTLN_SRCS "${DTLN_PORT}/dtln_model_p020_i16pt_a16.c")
    endif()
    if(BENCH_DTLN_FC STREQUAL "ref")
        # (handled above)
    elseif(BENCH_DTLN_FC STREQUAL "pie")
        set(BENCH_FC_SRCS "dtln_fc_pie_core.c" "dtln_fc_pie_tflm.cc" "dtln_fc_pie.S")
        set(BENCH_FC_DEFS DTLN_FC_S3_PIE=1 DTLN_FC_PIE_ASM=1)
        set(ABF_BUILD_ID "${ABF_BUILD_ID}+P")
        message(STATUS "benchmark_s3: DTLN matvec = PIE assembly (dtln_fc_pie.S)")
    elseif(BENCH_DTLN_FC STREQUAL "piec")
        set(BENCH_FC_SRCS "dtln_fc_pie_core.c" "dtln_fc_pie_tflm.cc")
        set(BENCH_FC_DEFS DTLN_FC_S3_PIE=1 DTLN_FC_PIE_ASM=0)
        set(ABF_BUILD_ID "${ABF_BUILD_ID}+C")
        message(STATUS "benchmark_s3: DTLN matvec = dtln_fc_pie glue with its C dot product")
    else()
        message(FATAL_ERROR "BENCH_DTLN_FC=${BENCH_DTLN_FC}: must be ref, pie or piec")
    endif()
endif()

idf_component_register(
    SRCS "main.c" ${BENCH_USB_SRCS} ${BENCH_ABF_SRCS} ${BENCH_DTLN_SRCS} ${BENCH_FC_SRCS}
         ${BENCH_RNNOISE_SRCS}
         ${BENCH_PROBE_SRCS}
         "bench/bench_ccount.c"
         "bench/bench_plat.c"
    INCLUDE_DIRS "." "bench" "abf" "${CLASSICAL}" "${DTLN_PORT}"
                 "${RNNOISE_DIR}" "${RNNOISE_DIR}/xiph-v0.1.1/include"
    REQUIRES esp_timer esp_psram espressif__esp-nn espressif__esp-dsp
             app_update esp_partition bootloader_support esp_app_format
             ${BENCH_USB_REQ} ${BENCH_DTLN_REQ}
    LDFRAGMENTS ${BENCH_LDFRAGMENTS})
if(BENCH_WITH_DTLN AND BENCH_WITH_TINYUSB)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BENCH_WITH_DTLN)
    # Explicit, beside REQUIRES tflm: the tensor struct layout depends on
    # TF_LITE_STATIC_MEMORY, so the processor must see exactly the library's defines.
    set(TFLM_TREE ${COMPONENT_DIR}/../../tflm_tree_64b3de8_generic)
    target_include_directories(${COMPONENT_LIB} PRIVATE ${TFLM_TREE} ${TFLM_TREE}/third_party/flatbuffers/include)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE TF_LITE_STATIC_MEMORY TF_LITE_STRIP_ERROR_STRINGS TF_LITE_DISABLE_X86_NEON)
    target_link_libraries(${COMPONENT_LIB} PRIVATE idf::tflm)
    # boundary arithmetic rounds as the host's numpy did: no FMA contraction
    set_source_files_properties(${DTLN_PORT}/dtln_proc.cc PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    if(BENCH_FC_DEFS)
        target_compile_definitions(${COMPONENT_LIB} PRIVATE ${BENCH_FC_DEFS})
    endif()
    if(BENCH_DTLN_STAMPS)
        target_compile_definitions(${COMPONENT_LIB} PRIVATE DTLN_STAMP_DWT)
    endif()
    # model placement, a measured variable on this tier (see dtln_relocate.c)
    if(BENCH_DTLN_MODEL STREQUAL "psram")
        target_compile_definitions(${COMPONENT_LIB} PRIVATE DTLN_MODEL_PSRAM)
        message(STATUS "benchmark_s3: DTLN model copied to PSRAM at setup")
    endif()
endif()

if(BENCH_WITH_PROBE)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BENCH_WITH_PROBE)
endif()
if(BENCH_WITH_TINYUSB)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BENCH_WITH_TINYUSB
        ABF_PROC_MAX_FRAME=512u CDSP_FFT_MAX=512u ABF_BUILD_ID="${ABF_BUILD_ID}"
        BENCH_ABF_STACK_BYTES=${BENCH_ABF_STACK_BYTES})
    if(BENCH_WITH_CDSP)
        target_compile_definitions(${COMPONENT_LIB} PRIVATE BENCH_WITH_CDSP)
    endif()
endif()
if(BENCH_WITH_RNNOISE)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BENCH_WITH_RNNOISE RNNOISE_PROC_MCU)
    if(NOT BENCH_WITH_CDSP)
        target_compile_definitions(${COMPONENT_LIB} PRIVATE RNNOISE_SELECT_NO_CDSP)
    endif()
    # Vendored files compiled as shipped: -w, no warning flags of ours. The
    # contraction convention is the classical rows' (GCC default ON; OFF builds
    # the bit-exact twin of the x86 oracle). -fstack-usage emits the .su files
    # the task stack is sized from; it changes no code.
    if(BENCH_RNNOISE_FP_CONTRACT)
        set_source_files_properties(${RNNOISE_FILES} PROPERTIES COMPILE_OPTIONS "-w;-fstack-usage;${RNNOISE_EXTRA_OPTS}")
        set_source_files_properties(${RNNOISE_DIR}/rnnoise_proc.c PROPERTIES COMPILE_OPTIONS "-fstack-usage")
    else()
        set_source_files_properties(${RNNOISE_FILES} PROPERTIES COMPILE_OPTIONS "-w;-fstack-usage;-ffp-contract=off;${RNNOISE_EXTRA_OPTS}")
        set_source_files_properties(${RNNOISE_DIR}/rnnoise_proc.c PROPERTIES COMPILE_OPTIONS "-fstack-usage;-ffp-contract=off")
    endif()
    if(BENCH_RNNOISE_VARIANT STREQUAL "ramw" OR BENCH_RNNOISE_VARIANT STREQUAL "both")
        target_compile_definitions(${COMPONENT_LIB} PRIVATE RNNOISE_RAM_WEIGHTS)
        # rnn_data.h / rnn.h / opus_types.h live in the vendored src/, not include/
        set_source_files_properties(rnnoise_relocate.c PROPERTIES INCLUDE_DIRECTORIES "${RNNOISE_DIR}/xiph-v0.1.1/src")
        target_link_options(${COMPONENT_LIB} INTERFACE "-Wl,--wrap=rnnoise_create")
    endif()
    if(BENCH_RNNOISE_STAMPS)
        target_compile_definitions(${COMPONENT_LIB} PRIVATE BENCH_RNNOISE_STAMPS)
    endif()
endif()
if(BENCH_CDSP_STAMPS)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE CDSP_STAMP_DWT)
endif()

# --- Configuration assertions -------------------------------------------------
# The ESP counterpart of the FATAL_ERROR guard in benchmark_h723/CMakeLists.txt,
# added for the same reason and after the same defect: on 2026-09-11 the H723
# probe was found to have been built at -Os with assert() live, because a
# setting that READ correctly had been silently discarded. It presented as a
# hardware anomaly, not a build bug.
#
# ESP-IDF defaults to exactly the two things that went wrong there --
# CONFIG_COMPILER_OPTIMIZATION_SIZE (-Os) and assertions enabled -- so a
# sdkconfig that drifts back to defaults must break the BUILD, not the results.
# sdkconfig.defaults is only a default: it is ignored once sdkconfig exists, so
# these checks are the thing that actually holds.
if(NOT CONFIG_COMPILER_OPTIMIZATION_PERF)
    message(FATAL_ERROR
        "CONFIG_COMPILER_OPTIMIZATION_PERF is not set: the probe would build at "
        "-Os. Benchmark numbers from an -Os build are not comparable to the "
        "F411/H723 rows. See 'Defect 1' in CLAUDE.md.")
endif()
if(NOT CONFIG_COMPILER_OPTIMIZATION_ASSERTIONS_DISABLE)
    message(FATAL_ERROR
        "Assertions are enabled: assert() would be live inside timed regions. "
        "This is the -DNDEBUG requirement from the STM32 tiers.")
endif()
if(NOT CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ EQUAL 240)
    message(FATAL_ERROR
        "CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ is ${CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ}, "
        "expected 240. Benchmarking at the IDF default of 160 MHz while "
        "reporting 240 is this platform's version of benchmarking on the HSI.")
endif()
message(STATUS "benchmark_s3: -O2 + NDEBUG + ${CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ} MHz asserted")
