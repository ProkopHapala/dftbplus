set(projectdir ${PROJECT_SOURCE_DIR})

# Sources for the library
set(sources-lib-fpp
  libdftbcore.F90)

set(fypp_flags ${FYPP_BUILD_FLAGS})
list(APPEND fypp_flags -I${projectdir}/src/dftbp/include)

dftbp_preprocess("${FYPP}" "${fypp_flags}" "F90" "f90" "${sources-lib-fpp}" sources-lib-f90-preproc)

# Shared library target for Python ctypes interface
add_library(dftbcore SHARED ${sources-lib-f90-preproc})
set_target_properties(dftbcore PROPERTIES
  OUTPUT_NAME "dftbcore"
  POSITION_INDEPENDENT_CODE ON)

target_link_libraries(dftbcore PRIVATE dftbplus)

if(WITH_MPI)
  target_link_libraries(dftbcore PRIVATE MPI::MPI_Fortran)
endif()

install(TARGETS dftbcore
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
