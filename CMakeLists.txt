cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

option(ENABLE_CUDA "Enable CUDA support" OFF)

if(ENABLE_CUDA)
    project(ClassicalSpin_Cpp LANGUAGES CXX CUDA)
else()
    project(ClassicalSpin_Cpp LANGUAGES CXX)
endif()

option(CLASSICAL_SPIN_ENABLE_WARNINGS "Enable recommended warning flags" ON)
option(CLASSICAL_SPIN_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

# Enable using *_ROOT environment variables for find_package
cmake_policy(SET CMP0144 NEW)

# Use Config mode for Boost (avoids deprecated FindBoost module)
# CMP0167 is only available in CMake 3.30+
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Modern CMake policies for CUDA
    cmake_policy(SET CMP0104 NEW)
    cmake_policy(SET CMP0105 NEW)
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

# CUDA architectures
if(ENABLE_CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(_classical_spin_default_arch "native")
        if(DEFINED ENV{CLASSICAL_SPIN_CUDA_ARCHS} AND NOT "$ENV{CLASSICAL_SPIN_CUDA_ARCHS}" STREQUAL "")
            set(_classical_spin_default_arch "$ENV{CLASSICAL_SPIN_CUDA_ARCHS}")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES "${_classical_spin_default_arch}" CACHE STRING "CUDA architectures" FORCE)
    endif()
    unset(_classical_spin_default_arch)
endif()

# ============================================================================
# DEPENDENCIES
# ============================================================================

if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(Thrust REQUIRED CONFIG)
endif()
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost 1.65 REQUIRED CONFIG)

# Use HDF5_DIR environment variable if available (set by module system)
if(DEFINED ENV{HDF5_DIR} AND NOT HDF5_ROOT)
    set(HDF5_ROOT "$ENV{HDF5_DIR}")
endif()
find_package(HDF5 QUIET COMPONENTS C CXX)

if(NOT HDF5_FOUND)
    find_package(PkgConfig QUIET)

    set(CLASSICAL_SPIN_HDF5_INCLUDE_HINTS "")
    set(CLASSICAL_SPIN_HDF5_LIBRARY_HINTS "")

    if(PkgConfig_FOUND)
        pkg_check_modules(CLASSICAL_SPIN_HDF5_PKG QUIET hdf5-serial)
        if(NOT CLASSICAL_SPIN_HDF5_PKG_FOUND)
            pkg_check_modules(CLASSICAL_SPIN_HDF5_PKG QUIET hdf5)
        endif()

        if(CLASSICAL_SPIN_HDF5_PKG_FOUND)
            list(APPEND CLASSICAL_SPIN_HDF5_INCLUDE_HINTS ${CLASSICAL_SPIN_HDF5_PKG_INCLUDE_DIRS})
            list(APPEND CLASSICAL_SPIN_HDF5_LIBRARY_HINTS ${CLASSICAL_SPIN_HDF5_PKG_LIBRARY_DIRS})
            if(NOT HDF5_VERSION)
                set(HDF5_VERSION "${CLASSICAL_SPIN_HDF5_PKG_VERSION}")
            endif()
        endif()
    endif()

    list(APPEND CLASSICAL_SPIN_HDF5_INCLUDE_HINTS
        /usr/include/hdf5/serial
        /usr/include/hdf5/openmpi
        /usr/local/include/hdf5/serial
        /usr/local/include/hdf5/openmpi
    )
    list(APPEND CLASSICAL_SPIN_HDF5_LIBRARY_HINTS
        /usr/lib/x86_64-linux-gnu/hdf5/serial
        /usr/lib/x86_64-linux-gnu/hdf5/openmpi
        /usr/lib64/hdf5/serial
        /usr/lib64/hdf5/openmpi
        /usr/lib/hdf5/serial
        /usr/lib/hdf5/openmpi
        /usr/local/lib/hdf5/serial
        /usr/local/lib/hdf5/openmpi
    )

    find_path(CLASSICAL_SPIN_HDF5_INCLUDE_DIR
        NAMES hdf5.h H5Cpp.h
        HINTS ${CLASSICAL_SPIN_HDF5_INCLUDE_HINTS}
    )
    find_library(CLASSICAL_SPIN_HDF5_C_LIBRARY
        NAMES hdf5
        HINTS ${CLASSICAL_SPIN_HDF5_LIBRARY_HINTS}
    )
    find_library(CLASSICAL_SPIN_HDF5_CXX_LIBRARY
        NAMES hdf5_cpp
        HINTS ${CLASSICAL_SPIN_HDF5_LIBRARY_HINTS}
    )
    find_library(CLASSICAL_SPIN_HDF5_HL_LIBRARY
        NAMES hdf5_hl
        HINTS ${CLASSICAL_SPIN_HDF5_LIBRARY_HINTS}
    )
    find_library(CLASSICAL_SPIN_HDF5_HL_CXX_LIBRARY
        NAMES hdf5_hl_cpp
        HINTS ${CLASSICAL_SPIN_HDF5_LIBRARY_HINTS}
    )

    if(CLASSICAL_SPIN_HDF5_INCLUDE_DIR AND CLASSICAL_SPIN_HDF5_C_LIBRARY AND CLASSICAL_SPIN_HDF5_CXX_LIBRARY)
        set(HDF5_FOUND TRUE)
        set(HDF5_INCLUDE_DIRS ${CLASSICAL_SPIN_HDF5_INCLUDE_DIR})
        set(HDF5_C_LIBRARIES ${CLASSICAL_SPIN_HDF5_C_LIBRARY})
        set(HDF5_CXX_LIBRARIES ${CLASSICAL_SPIN_HDF5_CXX_LIBRARY} ${CLASSICAL_SPIN_HDF5_C_LIBRARY})

        if(CLASSICAL_SPIN_HDF5_HL_LIBRARY)
            list(APPEND HDF5_C_LIBRARIES ${CLASSICAL_SPIN_HDF5_HL_LIBRARY})
            list(APPEND HDF5_CXX_LIBRARIES ${CLASSICAL_SPIN_HDF5_HL_LIBRARY})
        endif()
        if(CLASSICAL_SPIN_HDF5_HL_CXX_LIBRARY)
            list(APPEND HDF5_CXX_LIBRARIES ${CLASSICAL_SPIN_HDF5_HL_CXX_LIBRARY})
        endif()

        set(HDF5_LIBRARIES ${HDF5_CXX_LIBRARIES})
        list(REMOVE_DUPLICATES HDF5_C_LIBRARIES)
        list(REMOVE_DUPLICATES HDF5_CXX_LIBRARIES)
        list(REMOVE_DUPLICATES HDF5_LIBRARIES)

        message(STATUS "HDF5 found via pkg-config/manual fallback")
    endif()

    unset(CLASSICAL_SPIN_HDF5_INCLUDE_HINTS)
    unset(CLASSICAL_SPIN_HDF5_LIBRARY_HINTS)
endif()

if(NOT HDF5_FOUND)
    message(FATAL_ERROR
        "Unable to find HDF5 C and C++ development files. "
        "Set HDF5_ROOT/HDF5_DIR or install the HDF5 development package."
    )
endif()

# ============================================================================
# COMPILER FLAGS
# ============================================================================

# Compiler warnings configuration
set(CLASSICAL_SPIN_CXX_WARNING_FLAGS "")
set(CLASSICAL_SPIN_CUDA_WARNING_FLAGS "")

if(CLASSICAL_SPIN_ENABLE_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND CLASSICAL_SPIN_CXX_WARNING_FLAGS
            -Wall
            -Wextra
            -Wformat=2
            -Wnon-virtual-dtor
            # Suppress noisy warnings common in scientific computing code:
            -Wno-sign-compare          # size_t vs int comparisons (very common with Eigen)
            -Wno-sign-conversion       # signed/unsigned conversions
            -Wno-conversion            # narrowing conversions (common with Eigen Index types)
            -Wno-shadow                # constructor parameters shadowing members (common pattern)
            -Wno-old-style-cast        # C-style casts (from external headers like HDF5)
            -Wno-deprecated-copy       # deprecated implicit copy operators (HDF5 library issue)
            -Wno-float-conversion      # float/double conversions
            -Wno-unused-parameter      # unused parameters in callbacks/interfaces
            -Wno-pedantic              # strict ISO compliance (e.g., __int128)
            -Wno-reorder               # member initialization order
            -Wno-unused-variable       # unused variables (reserved for future use)
            -Wno-stringop-overread     # false positive in AVX intrinsics (GCC bug)
            -Wno-stringop-overflow     # false positive in AVX intrinsics (GCC bug)
        )
    elseif(MSVC)
        list(APPEND CLASSICAL_SPIN_CXX_WARNING_FLAGS /W3 /permissive-)
    endif()

    if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
        list(APPEND CLASSICAL_SPIN_CUDA_WARNING_FLAGS 
            -Xcompiler=-Wall 
            -Xcompiler=-Wextra
            -Xcompiler=-Wno-sign-compare
            -Xcompiler=-Wno-sign-conversion
            -Xcompiler=-Wno-conversion
            -Xcompiler=-Wno-shadow
            -Xcompiler=-Wno-old-style-cast
            -Xcompiler=-Wno-deprecated-copy
            -Xcompiler=-Wno-float-conversion
            -Xcompiler=-Wno-unused-parameter
            -Xcompiler=-Wno-pedantic
            -Xcompiler=-Wno-reorder
            -Xcompiler=-Wno-unused-variable
            -Xcompiler=-Wno-stringop-overread
            -Xcompiler=-Wno-stringop-overflow
        )
    endif()

    if(CLASSICAL_SPIN_WARNINGS_AS_ERRORS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            list(APPEND CLASSICAL_SPIN_CXX_WARNING_FLAGS -Werror)
        elseif(MSVC)
            list(APPEND CLASSICAL_SPIN_CXX_WARNING_FLAGS /WX)
        endif()

        if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
            list(APPEND CLASSICAL_SPIN_CUDA_WARNING_FLAGS -Xcompiler=-Werror)
        endif()
    endif()
endif()

function(classical_spin_apply_warnings target)
    if(CLASSICAL_SPIN_ENABLE_WARNINGS)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${CLASSICAL_SPIN_CXX_WARNING_FLAGS}>
            $<$<COMPILE_LANGUAGE:CUDA>:${CLASSICAL_SPIN_CUDA_WARNING_FLAGS}>
        )
    endif()
endfunction()

# CUDA compiler flags
if(ENABLE_CUDA)
    # --extended-lambda allows __device__ annotation on lambdas (required for Thrust)
    set(CMAKE_CUDA_FLAGS "-D__CUDA_NO_HOST_COMPILER_CHECK --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-gpu-targets")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo")
    
    # Suppress nvlink warnings about incompatible libraries (32-bit libs on 64-bit system)
    set(CMAKE_CUDA_LINK_FLAGS "${CMAKE_CUDA_LINK_FLAGS} -Xnvlink=-w")
endif()

# Optimization flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CXX_OPTIMIZATION_FLAGS
        -O3 -DNDEBUG -march=native -mtune=native -flto
        -funroll-loops -fomit-frame-pointer -fno-stack-protector
        -ffp-contract=fast -fno-math-errno -fno-trapping-math -fno-rounding-math
        -Wno-stringop-overread -Wno-stringop-overflow  # suppress false positives in AVX intrinsics during LTO
    )
    
    set(CUDA_OPTIMIZATION_FLAGS
        -O3 -DNDEBUG --use_fast_math --ptxas-options=-v,-O3
        -lineinfo --maxrregcount=32 --extra-device-vectorization
    )
    
    set(CUDA_HOST_OPTIMIZATION_FLAGS
        -Xcompiler=-O3 -Xcompiler=-ffast-math -Xcompiler=-funroll-loops
    )
    
    # AMX intrinsics workaround for nvcc/GCC 12
    set(CUDA_AMX_WORKAROUND
        --compiler-options=-D_AMXTILEINTRIN_H_INCLUDED
        --compiler-options=-D_AMX_INT8INTRIN_H_INCLUDED
        --compiler-options=-D_AMX_BF16INTRIN_H_INCLUDED
    )
    
    if(OpenMP_CXX_FOUND)
        list(APPEND CXX_OPTIMIZATION_FLAGS -fopenmp-simd)
    endif()
endif()

# ============================================================================
# SOURCE FILES
# ============================================================================

set(CLASSICAL_SPIN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CLASSICAL_SPIN_HEADERS
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/simple_linear_alg.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/spin_config.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/unitcell.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/unitcell_builders.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/gneb.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/gneb_math.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice/lattice.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice/mixed_lattice.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice/phonon_lattice.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice/phonon_config.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice/kitaev_bonds.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/io/hdf5_io.h
)

if(ENABLE_CUDA)
    list(APPEND CLASSICAL_SPIN_HEADERS
        ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/gpu_common_helpers.cuh
        ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/lattice_gpu.cuh
        ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/lattice_gpu_api.h
        ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/mixed_lattice_gpu.cuh
        ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/mixed_lattice_gpu_api.h
    )
endif()

set(CORE_SOURCES
    src/core/simple_linear_alg.cpp
    src/core/unitcell_builders.cpp
    src/core/spin_config.cpp
    src/core/phonon_lattice.cpp
    src/core/phonon_config.cpp
    src/core/gneb.cpp
    # Per-concern TUs split out of the `lattice.h` / `mixed_lattice.h`
    # god-object headers. Each one hosts a topical slice of the class so
    # the header stays slim and the slices build in parallel.
    src/core/lattice_accumulator.cpp
    src/core/lattice_mc.cpp
    src/core/lattice_pt.cpp
    src/core/lattice_observables.cpp
    src/core/lattice_md.cpp
    src/core/mixed_lattice_mc.cpp
    src/core/mixed_lattice_pt.cpp
    src/core/mixed_lattice_observables.cpp
    src/core/mixed_lattice_md.cpp
)

if(ENABLE_CUDA)
    set(GPU_SOURCES
        src/gpu/gpu_common_helpers.cu
        src/gpu/lattice_gpu.cu
        src/gpu/mixed_lattice_gpu.cu
    )
endif()

# ============================================================================
# LIBRARIES
# ============================================================================

add_library(classical_spin_headers INTERFACE)
target_sources(classical_spin_headers INTERFACE ${CLASSICAL_SPIN_HEADERS})
target_include_directories(classical_spin_headers INTERFACE 
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin>
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core>
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice>
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu>
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/io>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/classical_spin>
    $<INSTALL_INTERFACE:include/classical_spin/core>
    $<INSTALL_INTERFACE:include/classical_spin/lattice>
    $<INSTALL_INTERFACE:include/classical_spin/gpu>
    $<INSTALL_INTERFACE:include/classical_spin/io>
)
target_compile_features(classical_spin_headers INTERFACE cxx_std_20)
target_link_libraries(classical_spin_headers INTERFACE 
    OpenMP::OpenMP_CXX 
    MPI::MPI_CXX 
    Eigen3::Eigen
    ${HDF5_CXX_LIBRARIES}
)
if(ENABLE_CUDA)
    target_link_libraries(classical_spin_headers INTERFACE CUDA::cudart)
endif()
target_include_directories(classical_spin_headers INTERFACE ${HDF5_INCLUDE_DIRS})
target_compile_definitions(classical_spin_headers INTERFACE 
    OPENMP_ENABLED MPI_ENABLED EIGEN3_ENABLED HDF5_ENABLED
)
if(ENABLE_CUDA)
    target_compile_definitions(classical_spin_headers INTERFACE CUDA_ENABLED)
endif()

add_library(classical_spin_core STATIC ${CORE_SOURCES})
target_compile_features(classical_spin_core PUBLIC cxx_std_20)
target_link_libraries(classical_spin_core PUBLIC 
    classical_spin_headers
    Eigen3::Eigen
    ${HDF5_CXX_LIBRARIES}
)
target_include_directories(classical_spin_core PUBLIC ${HDF5_INCLUDE_DIRS})
target_include_directories(classical_spin_core PUBLIC 
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(classical_spin_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_OPTIMIZATION_FLAGS}>
)
classical_spin_apply_warnings(classical_spin_core)

# CUDA library
if(ENABLE_CUDA AND GPU_SOURCES)
    add_library(classical_spin_gpu STATIC ${GPU_SOURCES})
    
    set_target_properties(classical_spin_gpu PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    target_include_directories(classical_spin_gpu PUBLIC 
        $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${Boost_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
    )
    
    target_link_libraries(classical_spin_gpu PUBLIC 
        classical_spin_headers
        Eigen3::Eigen
        CUDA::cudart_static CUDA::cuda_driver CUDA::cublas CUDA::curand
        MPI::MPI_CXX
        ${HDF5_CXX_LIBRARIES}
    )
    
    target_compile_definitions(classical_spin_gpu PUBLIC 
        THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
        MPI_ENABLED
        HDF5_ENABLED
    )
    
    target_compile_options(classical_spin_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_AMX_WORKAROUND}>
    )
    classical_spin_apply_warnings(classical_spin_gpu)
endif()

# ============================================================================
# HELPER FUNCTION FOR EXECUTABLES
# ============================================================================

function(add_optimized_executable exec_name source_file)
    # Accept any additional source files via ARGN so multi-TU executables
    # (e.g. spin_solver, which was split into runners_*.cpp) can use the
    # same wrapper as single-file targets.
    add_executable(${exec_name} ${source_file} ${ARGN})
    target_compile_features(${exec_name} PRIVATE cxx_std_20)
    target_link_libraries(${exec_name} PRIVATE classical_spin_headers classical_spin_core)
    target_include_directories(${exec_name} PRIVATE ${CLASSICAL_SPIN_INCLUDE_DIR})
    
    if(TARGET classical_spin_gpu)
        target_link_libraries(${exec_name} PRIVATE classical_spin_gpu)
        set_target_properties(${exec_name} PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    endif()
    
    target_compile_options(${exec_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CXX_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:-g1>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
    # Suppress LTO warnings from AVX intrinsics (GCC bug with -flto and avx512)
    target_link_options(${exec_name} PRIVATE
        -Wno-stringop-overread -Wno-stringop-overflow
    )
    classical_spin_apply_warnings(${exec_name})
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set_target_properties(${exec_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endfunction()

# ============================================================================
# EXECUTABLES
# ============================================================================

# Main spin solver executable — main() plus the per-lattice runner TUs
# (split out of spin_solver.cpp for parallel compilation and readability).
add_optimized_executable(spin_solver
    src/apps/spin_solver.cpp
    src/apps/system_factory.cpp
    src/apps/runners_lattice.cpp
    src/apps/runners_phonon.cpp
    src/apps/runners_population_annealing.cpp
    src/apps/runners_mixed.cpp
    src/apps/runners_parameter_sweep.cpp
)

# Provenance for spin_solver's run_info.txt: git describe ("unknown" outside a
# git checkout), compiler, build type and flags. Regenerated at every build;
# the header only changes (and spin_solver only recompiles) when they do.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _classical_spin_bt)
string(JOIN " " _classical_spin_flags ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_classical_spin_bt}}
       ${CXX_OPTIMIZATION_FLAGS})
add_custom_target(classical_spin_build_info
    COMMAND ${CMAKE_COMMAND}
            -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DTEMPLATE=${CMAKE_CURRENT_SOURCE_DIR}/src/apps/build_info.h.in
            -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/generated/classical_spin_build_info.h
            "-DCLASSICAL_SPIN_CXX_COMPILER=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
            "-DCLASSICAL_SPIN_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
            "-DCLASSICAL_SPIN_CXX_FLAGS=${_classical_spin_flags}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/src/apps/build_info.cmake
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/generated/classical_spin_build_info.h
    COMMENT "Recording git describe / build flags for run_info.txt"
    VERBATIM)
add_dependencies(spin_solver classical_spin_build_info)
target_include_directories(spin_solver PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

# Symmetry + finite-difference regression tests for the NCTO E1
# magnetoelastic Hamiltonian (PhononLattice). Verifies the form-factor
# sum rule, the explicit channel expansion, the absence of any linear-in-ε
# exchange-striction term, force/energy consistency on both the spin and
# phonon sides, and one-bond C_3 invariance.
add_optimized_executable(test_e1_phonon_regressions src/apps/test_e1_phonon_regressions.cpp)

# 3Q / zigzag operating point and static magnetoelastic bias: T=0 quench from a
# seed at frozen E1 amplitude, over a J7 grid.
add_optimized_executable(ncto_operating_point src/apps/ncto_operating_point.cpp)

# THz pulse scan: does one pulse convert 3Q into zigzag, as a function of J7,
# field amplitude and polarization?  Reports the M-point order parameters vs time.
add_optimized_executable(ncto_pulse_scan src/apps/ncto_pulse_scan.cpp)

# Same pulse at FINITE temperature (Langevin). Barrier crossing is impossible in the
# T=0 deterministic dynamics, so nucleation can only be tested here.
add_optimized_executable(ncto_pulse_thermal src/apps/ncto_pulse_thermal.cpp)

# Same, but with the Bose-coloured thermostat, an uninterrupted noise stream, and the
# linear (E-channel) tilt emitted directly as spin_phonon_energy()/N.  Supersedes
# ncto_pulse_thermal; see the header comment there for why all three matter.
add_optimized_executable(ncto_pulse_thermal2 src/apps/ncto_pulse_thermal2.cpp)

# Exact PhononLattice energy evaluator for offline reaction-coordinate diagnostics.
add_optimized_executable(evaluate_phonon_energy_path src/apps/evaluate_phonon_energy_path.cpp)

# GNEB force/energy evaluator: per-site effective field + energy + strain force
# for a batch of band images (real geodesic nudged elastic band, driven from Python).
add_optimized_executable(gneb_field_eval src/apps/gneb_field_eval.cpp)

# Regression tests for mixed SU2-SU3 trilinear bookkeeping
add_optimized_executable(test_mixed_trilinear_regressions src/apps/test_mixed_trilinear_regressions.cpp)

# SA ground-state finder for the full mixed (Fe+Tm) TmFeO3 system.
# Uses biased Ka=0/Kc=-0.029 anisotropy to pin the Neel vector along c
# and reliably reach the Gamma_2 (F_x, G_z) ground state from random starts.
add_optimized_executable(diag_tmfeo3_gamma2_sa src/apps/diag_tmfeo3_gamma2_sa.cpp)

# K^- component scan: turns on each of the 9 K^- entries one at a time
# and reports the resulting Fe+Tm ground state.
add_optimized_executable(diag_tmfeo3_kminus_scan src/apps/diag_tmfeo3_kminus_scan.cpp)
add_optimized_executable(diag_tmfeo3_k2y_ramp   src/apps/diag_tmfeo3_k2y_ramp.cpp)

# Regression test that legacy TmFeO3 frame-selection flags are ignored by the
# current mixed builder.
add_optimized_executable(test_global_frame_equivalence src/apps/test_tmfeo3_legacy_frame_flags.cpp)

# Companion regression test that the standalone Fe-only TmFeO3 builder also
# ignores the retired frame-selection flags.
add_optimized_executable(test_fe_global_frame_equivalence src/apps/test_tmfeo3_fe_legacy_frame_flags.cpp)

# Stand-alone utility to project an existing TmFeO3 SU(3) seed file onto
# the closest physical qutrit pure-state representation, fixing the
# negative-density-eigenvalue defect documented in
# docs/tmfeo3_notes.tex (positive-cone section).
add_optimized_executable(fix_tmfeo3_su3_seed src/apps/fix_tmfeo3_su3_seed.cpp)

# Regression test for Ingredient XVIII: pulse-drive trajectories at
# the new 1e-8 default must agree with the legacy 1e-10 reference to
# physics tolerance (~ 5e-6 on the magnetisation observable).
add_optimized_executable(test_pulse_drive_tolerance src/apps/test_pulse_drive_tolerance.cpp)

# Regression test for Ingredient XX (audit B1): pulse_chunking helper
# must snap every segment endpoint to the T_step grid so Boost's
# `integrate_const` never drops a sub-T_step remainder at a seam.
add_optimized_executable(test_pulse_chunking_alignment src/apps/test_pulse_chunking_alignment.cpp)

# HDF5 parallel-tempering writer smoke test (no MPI / GPU needed)
add_optimized_executable(test_hdf5_pt src/test_hdf5_pt.cpp)

# Monte-Carlo kernel micro-benchmark (sweeps/sec for each MC algorithm on
# a few canonical models). Intentionally single-rank — meant for measuring
# the local-update kernels in isolation.
add_optimized_executable(bench_mc src/apps/bench_mc.cpp)

# Molecular-dynamics RHS micro-benchmark (RHS-calls/sec for each LLG /
# spin-strain ODE kernel on a few canonical models). Same intent as
# bench_mc but for the time-evolution side of the library.
add_optimized_executable(bench_md src/apps/bench_md.cpp)

# PhononLattice (NCTO spin-phonon) hot-path micro-benchmark: coupled RHS,
# Metropolis / overrelaxation sweeps, local field, total energy.
add_optimized_executable(bench_phonon src/apps/bench_phonon.cpp)

# Integrator-level micro-benchmark for Ingredient XVIII (pump-probe
# default tolerance relaxation + rk54 dispatch fix). Times
# single_pulse_drive() on a small TmFeO3 mixed lattice across method
# and tolerance combinations and prints speedup ratios.
add_optimized_executable(bench_integrators src/apps/bench_integrators.cpp)

# CPU-vs-GPU equivalence test for the in-house gpu::ode integrator module
# on the FULLY COUPLED SU(2)/SU(3) TmFeO3 mixed lattice (Fe<->Tm mixed
# bilinear chi + mixed trilinear W coupling). Drives a single pulse and
# compares CPU vs GPU magnetisation trajectories across integration
# methods (euler/rk2/rk4/ssprk53/dopri5). No-op when built without CUDA.
add_optimized_executable(test_mixed_gpu_cpu_equivalence src/apps/test_mixed_gpu_cpu_equivalence.cpp)

# MPI driver test of the spin-phonon model (registered below, run with 2 and 3 ranks)
add_optimized_executable(test_phonon_mpi tests/test_phonon_mpi.cpp)

# Physics validation suite (tests/): every kernel is checked against a result
# known independently of the code -- closed-form statistical mechanics, exact
# Boltzmann quadrature for tiny clusters, analytic precession / magnon
# frequencies and known classical ground states.
set(CLASSICAL_SPIN_PHYSICS_TESTS
    test_rng
    test_mc_statistics
    test_mc_exact
    test_md_exact
    test_md_drivers
    test_sa_ground_state
    test_mixed_md
    test_mixed_pump_probe
    test_phonon_dynamics
    test_pt_ladder
    test_unitcell_geometry
    test_population_annealing
    test_config
    test_config_spin_io
)
foreach(t IN LISTS CLASSICAL_SPIN_PHYSICS_TESTS)
    add_optimized_executable(${t} tests/${t}.cpp)
endforeach()
# test_config parses every example configuration and scans the sources for
# get_param keys (the Hamiltonian-key registry must cover them).
target_compile_definitions(test_config PRIVATE CLASSICAL_SPIN_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_optimized_executable(test_md_mpi_scan tests/test_md_mpi_scan.cpp)

if(BUILD_TESTING)
    foreach(t IN LISTS CLASSICAL_SPIN_PHYSICS_TESTS)
        add_test(NAME physics_${t} COMMAND ${t})
        set_tests_properties(physics_${t} PROPERTIES TIMEOUT 300 LABELS physics
                             ENVIRONMENT "OMP_NUM_THREADS=2")
    endforeach()
    # MPI delay scan: dynamic scheduling, sub-communicators, error propagation.
    add_test(NAME physics_md_mpi_scan
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 --oversubscribe --allow-run-as-root
                     $<TARGET_FILE:test_md_mpi_scan>)
    set_tests_properties(physics_md_mpi_scan PROPERTIES TIMEOUT 300 LABELS physics
                         ENVIRONMENT "OMP_NUM_THREADS=1")
    # The pump-probe test also runs on 4 MPI ranks (dynamic scheduling, error
    # propagation); the single-rank entry above exercises the serial fallback.
    add_test(NAME physics_test_mixed_pump_probe_mpi4
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe --allow-run-as-root
                     $<TARGET_FILE:test_mixed_pump_probe>)
    set_tests_properties(physics_test_mixed_pump_probe_mpi4 PROPERTIES TIMEOUT 300 LABELS physics
                         ENVIRONMENT "OMP_NUM_THREADS=1")
    # Parallel tempering on 4 MPI ranks against exact results (oversubscribed:
    # the CI container may have fewer cores; it runs as root).
    add_optimized_executable(test_pt_mpi tests/test_pt_mpi.cpp)
    add_test(NAME physics_test_pt_mpi
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 --oversubscribe --allow-run-as-root
                     $<TARGET_FILE:test_pt_mpi>)
    # Exchange rounds synchronise rank pairs; yielding while idle keeps the
    # oversubscribed ranks from starving each other.
    set_tests_properties(physics_test_pt_mpi PROPERTIES TIMEOUT 600 LABELS "physics;mpi"
                         ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_MCA_mpi_yield_when_idle=1")
    # Population annealing: the same population split over 3 ranks must give
    # bitwise the result of one rank (layout-independent random streams).
    add_test(NAME physics_test_population_annealing_mpi3
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 --oversubscribe --allow-run-as-root
                     $<TARGET_FILE:test_population_annealing>)
    set_tests_properties(physics_test_population_annealing_mpi3 PROPERTIES TIMEOUT 300 LABELS "physics;mpi"
                         ENVIRONMENT "OMP_NUM_THREADS=1")
    add_test(NAME e1_phonon_regressions COMMAND test_e1_phonon_regressions)
    add_test(NAME mixed_trilinear_regressions COMMAND test_mixed_trilinear_regressions)
    add_test(NAME global_frame_equivalence COMMAND test_global_frame_equivalence)
    add_test(NAME fe_global_frame_equivalence COMMAND test_fe_global_frame_equivalence)
    add_test(NAME pulse_drive_tolerance COMMAND test_pulse_drive_tolerance)
    set_tests_properties(pulse_drive_tolerance PROPERTIES TIMEOUT 120)
    add_test(NAME pulse_chunking_alignment COMMAND test_pulse_chunking_alignment)
    set_tests_properties(pulse_chunking_alignment PROPERTIES TIMEOUT 30)
    add_test(NAME mixed_gpu_cpu_equivalence COMMAND test_mixed_gpu_cpu_equivalence)
    set_tests_properties(mixed_gpu_cpu_equivalence PROPERTIES TIMEOUT 300)
    # PhononLattice MPI drivers: tau-parallel 2DCS file == serial file, replica exchange
    # carries the lattice sector; and the spin_solver NCTO 2DCS runner terminates under MPI
    # (it deadlocked: a COMM_WORLD collective was called from rank 0 only).
    if(MPIEXEC_EXECUTABLE)
        foreach(np 2 3)
            add_test(NAME mpi_phonon_drivers_np${np}
                     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} --oversubscribe
                             --allow-run-as-root $<TARGET_FILE:test_phonon_mpi>)
            set_tests_properties(mpi_phonon_drivers_np${np} PROPERTIES TIMEOUT 180 LABELS mpi
                                 ENVIRONMENT "OMP_NUM_THREADS=1"
                                 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        endforeach()
        add_test(NAME mpi_ncto_2dcs
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 --oversubscribe --allow-run-as-root
                         $<TARGET_FILE:spin_solver> ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/ncto_2dcs_mpi.param)
        set_tests_properties(mpi_ncto_2dcs PROPERTIES TIMEOUT 180 LABELS mpi
                             PASS_REGULAR_EXPRESSION "PhononLattice 2DCS completed"
                             ENVIRONMENT "OMP_NUM_THREADS=1"
                             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        # Parameter sweep with 3 points on 2 ranks (deadlocked on COMM_WORLD
        # barriers inside the runners) over a typed key.
        add_test(NAME mpi_sweep_uneven
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 --oversubscribe --allow-run-as-root
                         $<TARGET_FILE:spin_solver> ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/sweep_uneven_mpi.param)
        set_tests_properties(mpi_sweep_uneven PROPERTIES TIMEOUT 180 LABELS mpi
                             PASS_REGULAR_EXPRESSION "Parameter sweep completed \\(3 points\\)"
                             ENVIRONMENT "OMP_NUM_THREADS=1"
                             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        # A failing run reports "[rank r] error: ..." and aborts every rank
        # (instead of hanging, or continuing from random spins).
        add_test(NAME mpi_runtime_error_aborts
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 --oversubscribe --allow-run-as-root
                         $<TARGET_FILE:spin_solver> ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/missing_seed_mpi.param)
        set_tests_properties(mpi_runtime_error_aborts PROPERTIES TIMEOUT 120 LABELS mpi
                             PASS_REGULAR_EXPRESSION "\\[rank [0-9]+\\] error: .*no_such_seed_file"
                             FAIL_REGULAR_EXPRESSION "completed successfully"
                             ENVIRONMENT "OMP_NUM_THREADS=1"
                             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    # End-to-end smoke tests of every simulation mode, sweeps (uneven MPI
    # splits, typed keys, PT groups), provenance and error exits.
    add_test(NAME smoke_spin_solver
             COMMAND bash ${CMAKE_SOURCE_DIR}/tests/smoke/run_smoke.sh $<TARGET_FILE:spin_solver>)
    set_tests_properties(smoke_spin_solver PROPERTIES TIMEOUT 900 LABELS smoke
                         ENVIRONMENT "OMP_NUM_THREADS=1")
    add_test(NAME hdf5_pt_writer_smoke COMMAND test_hdf5_pt)
    set_tests_properties(hdf5_pt_writer_smoke PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

endif()


# ============================================================================
# BUILD CONFIGURATION
# ============================================================================

# Enable parallel builds
if(NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
    include(ProcessorCount)
    ProcessorCount(N)
    if(NOT N EQUAL 0)
        set(CMAKE_BUILD_PARALLEL_LEVEL ${N} CACHE STRING "Number of parallel build jobs")
    endif()
endif()

# Ninja generator optimization
if(CMAKE_GENERATOR STREQUAL "Ninja")
    set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
endif()

# ============================================================================
# BUILD SUMMARY
# ============================================================================

message(STATUS "================ ClassicalSpin_Cpp Build =================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
if(ENABLE_CUDA)
    message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA: Disabled")
endif()
message(STATUS "Parallel Build Jobs: ${CMAKE_BUILD_PARALLEL_LEVEL}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  OpenMP: ${OpenMP_CXX_VERSION}")
message(STATUS "  MPI: ${MPI_CXX_VERSION}")
if(ENABLE_CUDA)
    message(STATUS "  CUDA: ${CUDAToolkit_VERSION}")
endif()
message(STATUS "  Eigen3: ${Eigen3_VERSION}")
message(STATUS "  Boost: ${Boost_VERSION}")
message(STATUS "  HDF5: ${HDF5_VERSION}")
message(STATUS "")
message(STATUS "Source Files:")
message(STATUS "  Core sources: ${CORE_SOURCES}")
if(TARGET classical_spin_gpu)
    message(STATUS "  CUDA sources: ${GPU_SOURCES}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "")
    message(STATUS "Optimization Flags:")
    message(STATUS "  C++: -O3 -march=native -flto ...")
    message(STATUS "  CUDA: -O3 --use_fast_math ...")
endif()

message(STATUS "===========================================================")