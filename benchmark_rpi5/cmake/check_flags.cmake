# Post-build verification: read the optimisation level out of the OBJECT THAT
# WAS ACTUALLY COMPILED, not out of a CMake variable this project forced.
#
# -g3 makes GCC write its real command line into DW_AT_producer. That string is
# the compiler's own account of what it was told, so it cannot be shadowed,
# re-initialised, or overwritten by build-system ordering -- which is precisely
# how the H723's -Os defect survived review for a day.
if(NOT EXISTS "${OBJ}")
  message(FATAL_ERROR "object not found: ${OBJ} -- the check cannot be skipped silently")
endif()
foreach(o ${OBJ})
  if(o MATCHES "bench_kernels")
    execute_process(COMMAND readelf --debug-dump=info ${o}
                    OUTPUT_VARIABLE info ERROR_QUIET)
    string(REGEX MATCH "DW_AT_producer[^\n]*" prod "${info}")
    if(NOT prod)
      message(FATAL_ERROR "no DW_AT_producer in ${o}: cannot verify the build. "
                          "Is -g3 still set?")
    endif()
    if(NOT prod MATCHES "-O2")
      message(FATAL_ERROR
        "the compiled object was NOT built at -O2.\n  producer: ${prod}\n"
        "This is the H723 defect class: the build system reported one thing "
        "and the compiler was told another. Measurements from this build are "
        "invalid.")
    endif()
    message(STATUS "verified from object: ${prod}")
  endif()
endforeach()
