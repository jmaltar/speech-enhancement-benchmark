# Host-side build of the ABF1 reference codec and its tests.
#
# The codec itself is compiled into each firmware target separately; this
# project exists so the tests are scriptable, which task 26 (reproducibility
# artifact) requires and which the two-build flash/SRAM delta established as
# the house rule.
cmake_minimum_required(VERSION 3.16)
project(abf_harness C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(abf_frame STATIC abf_frame.c)
target_include_directories(abf_frame PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(abf_frame PRIVATE -Wall -Wextra -Wconversion)

add_library(abf_file STATIC abf_file.c abf_proc.c abf_wav.c)
target_link_libraries(abf_file PUBLIC abf_frame)
target_compile_options(abf_file PRIVATE -Wall -Wextra -Wconversion)
find_library(M_LIB m)
if(M_LIB)
  target_link_libraries(abf_file PUBLIC ${M_LIB})
endif()

add_executable(test_abf test_abf.c)
target_link_libraries(test_abf PRIVATE abf_frame)
target_compile_options(test_abf PRIVATE -Wall -Wextra -Wconversion)

add_executable(test_abf_file test_abf_file.c)
target_link_libraries(test_abf_file PRIVATE abf_file)
target_compile_options(test_abf_file PRIVATE -Wall -Wextra)

# abf_cdc.c has no TinyUSB dependency of its own -- the four transport
# primitives are extern, mapped onto tud_cdc_* by abf_cdc_tinyusb.c on hardware
# and onto a memory pipe by the tests. That is what lets the MCU binding be
# proved against the file binding here, with no board.
add_executable(test_abf_cdc test_abf_cdc.c abf_cdc.c)
target_link_libraries(test_abf_cdc PRIVATE abf_file)
target_compile_options(test_abf_cdc PRIVATE -Wall -Wextra)

add_executable(abf_run abf_run.c)
target_link_libraries(abf_run PRIVATE abf_file)
target_compile_options(abf_run PRIVATE -Wall -Wextra)

enable_testing()
add_test(NAME codec COMMAND test_abf)
add_test(NAME file_binding COMMAND test_abf_file)
add_test(NAME cdc_binding COMMAND test_abf_cdc)

# The C CRC and the host's zlib must agree, or every host-side integrity check
# silently disagrees with the device. Checked, not assumed.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME crc_matches_zlib
           COMMAND ${CMAKE_COMMAND}
                   -DTEST_BIN=$<TARGET_FILE:test_abf>
                   -DPY=${Python3_EXECUTABLE}
                   -DSCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/check_vector.py
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/run_vector.cmake)
endif()
