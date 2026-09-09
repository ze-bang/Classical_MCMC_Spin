cmake_minimum_required(VERSION 3.18)
project(numerics_practice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# -DNP_SOLUTIONS=ON builds the tests against solutions/ instead of exercises/.
# Both directories expose the same header names and the same namespaces, so the
# test files never change.
# -----------------------------------------------------------------------------
option(NP_SOLUTIONS "Build the tests against the reference solutions" OFF)
if(NP_SOLUTIONS)
  set(NP_IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/solutions)
  message(STATUS "numerics_practice: building against SOLUTIONS")
else()
  set(NP_IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/exercises)
  message(STATUS "numerics_practice: building against EXERCISES "
                 "(-DNP_SOLUTIONS=ON to check against the reference)")
endif()

include(CTest)
enable_testing()

set(NP_EXERCISES
  ex01_ode_explicit
  ex02_symplectic
  ex03_implicit_stiff
  ex04_parabolic
  ex05_hyperbolic
  ex06_elliptic_spectral
  ex07_sde_basics
  ex08_overdamped
  ex09_underdamped
  ex10_gle_sllg)

set(_i 1)
foreach(ex IN LISTS NP_EXERCISES)
  string(SUBSTRING "${ex}" 0 4 tag)          # ex01, ex02, ...
  add_executable(test_${tag} tests/test_${tag}.cpp)
  target_include_directories(test_${tag} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${NP_IMPL_DIR})
  # -Wno-array-bounds: GCC 13 false positive when lu_solve is inlined for n=1
  target_compile_options(test_${tag} PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-array-bounds)
  add_test(NAME ${tag} COMMAND test_${tag})
  set_tests_properties(${tag} PROPERTIES TIMEOUT 300)
  math(EXPR _i "${_i}+1")
endforeach()
