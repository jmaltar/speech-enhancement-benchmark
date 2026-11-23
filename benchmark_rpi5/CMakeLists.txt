# benchmark_rpi5 -- Raspberry Pi 5 kernel throughput probe (project task 3).
#
# The two build-system lessons this project has already paid for are applied
# here BEFORE the first measurement rather than after a defect:
#
#  1. The H723's `-Os` defect. Its CMakeLists set the right flags at line 33 --
#     ABOVE project() -- and project() re-initialises the per-language flag
#     variables, restoring the generator's defaults. The setting was present,
#     read correctly under review, and did nothing. It presented as a Cortex-M7
#     hardware anomaly and cost a full re-run.
#  2. Therefore the assertion at the BOTTOM of this file, not next to the
#     set(). An assertion placed immediately after the set() can never fire: it
#     re-reads a variable assigned two lines earlier. The defect being guarded
#     is a LATER re-initialisation, so the guard belongs after project() and
#     every add_subdirectory() have run.
#
# Verify the guard the way the H723's was verified: reproduce the bug (move the
# set() above project()) and confirm configure FAILS. An untested guard is
# decoration.

cmake_minimum_required(VERSION 3.16)
project(benchmark_rpi5 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -O2 -g3 -DNDEBUG: identical in intent to the STM32 and S3 tiers.
#   -O2   not -O3 and not -Os. -Os cost the H723 a re-run; -O3 would be a
#         different optimisation level from the other three tiers and the
#         comparison is the point.
#   -g3   changes no generated code, so Release stays debuggable -- which
#         matters here because "disassemble the kernel before theorising about
#         the core" is a standing rule on this project.
#   -DNDEBUG  assert() inside a timed region is measurement contamination.
#
# NOT USED, deliberately:
#   -ffast-math / -Ofast  would reassociate float arithmetic and change the
#         output. The cross-platform checksums are how task 22's numerical
#         equivalence is established -- 11 of 13 already match bit-for-bit
#         across ARM and Xtensa -- and fast-math would forfeit that for speed
#         the other tiers did not get.
#   -march=native  unreproducible by construction.
#
# -mcpu=cortex-a76 IS used: the STM32 tiers build with -mcpu=cortex-m4/m7 and
# the S3 tunes for its LX7, so tuning for the actual core is the consistent
# choice, not a thumb on the scale.
#
# NOTE FOR THE RESULTS, recorded before they exist: on AArch64 NEON is
# mandatory and GCC auto-vectorises float loops at -O2. The "plain C" rows on
# this tier may therefore already be SIMD, where the Cortex-M rows cannot be.
# That is the honest reading of "portable C at -O2 on this platform", but it
# must be CHECKED BY DISASSEMBLY and stated, or the later NEON pairing will
# look disappointing for the wrong reason. See PROVENANCE.md.
set(CMAKE_C_FLAGS_RELEASE "-O2 -g3 -DNDEBUG -mcpu=cortex-a76")

add_executable(benchmark_rpi5
  main.c
  bench/bench_kernels.c
  bench/bench_clock.c
  bench/bench_plat.c
  bench/bench_report.c
)
target_include_directories(benchmark_rpi5 PRIVATE bench)
target_link_libraries(benchmark_rpi5 PRIVATE m)

# --- the guard ---------------------------------------------------------------
#
# CORRECTED 2026-09-12, and the correction is the reusable part.
#
# The first version of this guard re-read CMAKE_C_FLAGS_RELEASE at the bottom of
# the file and compared it against "-O2". It was tested the way this project
# requires -- by reintroducing the H723 defect, moving the set() above
# project() -- and it did not fire. NEITHER DID THE BUG: the flags were still
# correct. Two things were learned:
#
#   1. The H723 defect does not reproduce in a plain native CMake project.
#      project() initialises CMAKE_C_FLAGS_RELEASE from the compiler module's
#      _INIT value into the CACHE, but a NORMAL variable of the same name set
#      earlier shadows the cache entry. CubeMX's generated structure is what
#      made it bite there, not `project()` alone. A lesson learned on one
#      generator must be re-verified on the next, not assumed to transfer.
#
#   2. Worse, because the set() above is unconditional, the guard could never
#      fire AT ALL: `cmake -DCMAKE_C_FLAGS_RELEASE=-Os` is silently overwritten
#      too. A guard that re-reads a variable the same file just forced is the
#      decoration this project's own rule warns about -- the identical error
#      the H723 guard made by sitting two lines under its set().
#
# So the check moved to where it cannot be fooled: THE ARTIFACT. -g3 makes GCC
# record its real command line in DW_AT_producer, so the post-build step below
# reads the optimisation level out of the object file that was actually
# compiled. NDEBUG does not appear there (it is a preprocessor define), so it
# is asserted from inside C instead -- see the #error in bench_clock.c.
#
# Verified by reproducing a bug this one CAN have: patch -O2 to -Os in the
# set() above and the BUILD fails at the post-build step, not merely configure.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
if(_bt STREQUAL "RELEASE")
  set(_flags "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
  if(_flags MATCHES "fast-math|Ofast")
    message(FATAL_ERROR
      "Release build enables fast-math (flags: ${_flags}). That reassociates "
      "float arithmetic and forfeits the cross-platform checksum equivalence.")
  endif()
  message(STATUS "Benchmark Release C flags: ${_flags}")

  add_custom_command(TARGET benchmark_rpi5 POST_BUILD
    COMMAND ${CMAKE_COMMAND}
            -DOBJ=${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/benchmark_rpi5.dir/bench/bench_kernels.c.o
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_flags.cmake
    VERBATIM
    COMMENT "Verifying -O2 in the compiled object's DW_AT_producer")
endif()
