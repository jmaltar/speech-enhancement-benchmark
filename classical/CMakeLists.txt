# Classical speech enhancement in portable C (task 8): library, the WAV tool
# that drives it through the ABF1 file binding, and the unit tests.
#
# Default build type is Release (-O2 -DNDEBUG), the same settings the firmware
# tiers measure at. Set it BEFORE project() -- build TYPE is safe there; it is
# the per-language FLAG variables that project() re-initialises (trap 1), and
# none are set here. -ffast-math is declined on purpose: it would reassociate
# float sums and forfeit the cross-platform equivalence task 15 rests on.
cmake_minimum_required(VERSION 3.16)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
project(cdsp C)
# The firmware tiers measure at -O2 -DNDEBUG; CMake's own Release default is
# -O3. Set AFTER project(), which is where per-language flags survive (trap 1).
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../harness ${CMAKE_BINARY_DIR}/harness)

add_library(cdsp STATIC cdsp.c cdsp_fft_c.c cdsp_select.c)
target_link_libraries(cdsp PUBLIC abf_file)   # cdsp_select needs the reference processors
target_include_directories(cdsp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../harness)
target_compile_options(cdsp PRIVATE -Wall -Wextra -Wconversion)
find_library(M_LIB m)
if(M_LIB)
  target_link_libraries(cdsp PUBLIC ${M_LIB})
endif()

add_executable(cdsp_wav cdsp_wav.c)
target_link_libraries(cdsp_wav PRIVATE cdsp abf_file)
target_compile_options(cdsp_wav PRIVATE -Wall -Wextra)

add_executable(test_cdsp test_cdsp.c)
target_link_libraries(test_cdsp PRIVATE cdsp)
target_compile_options(test_cdsp PRIVATE -Wall -Wextra)

enable_testing()
add_test(NAME cdsp COMMAND test_cdsp)

message(STATUS "cdsp: CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}  C flags (release): ${CMAKE_C_FLAGS_RELEASE}")

# The RPi5 tier's run tool: processor chosen from CONFIG.algo_id via
# cdsp_select, exactly as the MCU images do. POSIX scheduling calls, so Linux
# hosts and the Pi only.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(cdsp_run cdsp_run.c)
  target_link_libraries(cdsp_run PRIVATE cdsp abf_file)
  target_compile_options(cdsp_run PRIVATE -Wall -Wextra)
endif()

# -DCDSP_STAMPS=ON: per-stage stage sums inside cdsp_process() (DWT on Cortex-M,
# CCOUNT on Xtensa, CLOCK_MONOTONIC on Linux), printed by cdsp_run. Diagnostic
# build only: it adds a timer read per stage and must never be the build a
# cost row is quoted from.
option(CDSP_STAMPS "classical arm: per-stage stage sums" OFF)
if(CDSP_STAMPS)
  target_compile_definitions(cdsp PRIVATE CDSP_STAMP_DWT)
  if(TARGET cdsp_run)
    target_compile_definitions(cdsp_run PRIVATE CDSP_STAMP_DWT)
  endif()
endif()
