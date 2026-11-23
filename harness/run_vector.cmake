# Runs test_abf, captures its VECTOR lines, and has Python confirm the CRC
# against zlib. Two implementations agreeing is the point; one would prove
# nothing.
execute_process(COMMAND ${TEST_BIN} OUTPUT_VARIABLE out RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "test_abf failed (rc=${rc})")
endif()
set(vec "${CMAKE_BINARY_DIR}/vector.txt")
file(WRITE "${vec}" "${out}")
execute_process(COMMAND ${PY} ${SCRIPT} "${vec}" RESULT_VARIABLE prc)
if(NOT prc EQUAL 0)
  message(FATAL_ERROR "C codec CRC disagrees with zlib")
endif()
