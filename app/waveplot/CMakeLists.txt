set(projectdir ${PROJECT_SOURCE_DIR})

set(sources-fpp
  gridcache.F90
  initwaveplot.F90
  molorb.F90
  slater.F90
  waveplot.F90)

set(sources-lib-fpp
  gridcache.F90
  molorb.F90
  slater.F90
  libwaveplot.F90)

set(fypp_flags ${FYPP_BUILD_FLAGS})
list(APPEND fypp_flags -I${projectdir}/src/dftbp/include)

dftbp_preprocess("${FYPP}" "${fypp_flags}" "F90" "f90" "${sources-fpp}" sources-f90-preproc)
dftbp_preprocess("${FYPP}" "${fypp_flags}" "F90" "f90" "${sources-lib-fpp}" sources-lib-f90-preproc)

add_executable(waveplot ${sources-f90-preproc})

if(WITH_MPI)
  target_link_libraries(waveplot PRIVATE MPI::MPI_Fortran)
endif()

target_link_libraries(waveplot PRIVATE dftbplus)

install(TARGETS waveplot DESTINATION "${CMAKE_INSTALL_BINDIR}")

# Shared library target for Python ctypes interface

# =================== WARRNING ------ We forget to add libwaveplot.F90 to the repository (by git add, do it on machine at work)
# NOTE: libwaveplot.F90 is missing from repository, disabling this build
# add_library(waveplotlib SHARED ${sources-lib-f90-preproc})
# set_target_properties(waveplotlib PROPERTIES
#   OUTPUT_NAME "waveplot"
#   POSITION_INDEPENDENT_CODE ON)
# target_link_libraries(waveplotlib PRIVATE dftbplus)
# if(WITH_MPI)
#   target_link_libraries(waveplotlib PRIVATE MPI::MPI_Fortran)
# endif()
# install(TARGETS waveplotlib
#   LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
