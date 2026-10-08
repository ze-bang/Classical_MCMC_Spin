# build_info.cmake — writes classical_spin_build_info.h (git describe, compiler,
# build type and flags) for spin_solver's run_info.txt. Runs at every build;
# configure_file only touches the header when its content changes, so an
# unchanged checkout does not trigger a rebuild.
#
# Inputs (-D): SOURCE_DIR, OUTPUT, TEMPLATE, CLASSICAL_SPIN_CXX_COMPILER,
#              CLASSICAL_SPIN_BUILD_TYPE, CLASSICAL_SPIN_CXX_FLAGS
set(CLASSICAL_SPIN_GIT_DESCRIBE "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --always --dirty --tags
        WORKING_DIRECTORY ${SOURCE_DIR}
        OUTPUT_VARIABLE _describe
        RESULT_VARIABLE _status
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_status EQUAL 0 AND NOT _describe STREQUAL "")
        set(CLASSICAL_SPIN_GIT_DESCRIBE "${_describe}")
    endif()
endif()
string(REPLACE "\"" "\\\"" CLASSICAL_SPIN_CXX_FLAGS "${CLASSICAL_SPIN_CXX_FLAGS}")
configure_file(${TEMPLATE} ${OUTPUT} @ONLY)
