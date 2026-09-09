#!/usr/bin/env bash
# Build ClassicalSpin_Cpp on the Alliance Canada / Rorqual software stack.
#
# Usage (from anywhere):
#   ./build_rorqual.sh
#   CLASSICAL_SPIN_ENABLE_CUDA=ON ./build_rorqual.sh
#   ./build_rorqual.sh --no-clean    # keep existing build/ and reconfigure
#
# Optional: CLASSICAL_SPIN_CUDA_ARCHS (e.g. 80) is read by CMake when CUDA is on.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

CLEAN=1
for arg in "$@"; do
  case "$arg" in
    --no-clean) CLEAN=0 ;;
    -h|--help)
      echo "Usage: $0 [--no-clean]"
      echo "  --no-clean  Do not remove build/ before configuring."
      exit 0
      ;;
  esac
done

# StdEnv/2023 typical stack on Rorqual: compiler, MPI, and deps for CMake find_package
module load StdEnv/2023
module load cmake eigen boost hdf5-mpi

# Default Release; set CMAKE_BUILD_TYPE=Debug to override
: "${CMAKE_BUILD_TYPE:=Release}"
# OFF unless you also load cuda/cudatoolkit and want GPU paths in this project
: "${CLASSICAL_SPIN_ENABLE_CUDA:=OFF}"

if [[ "${CLEAN}" -eq 1 ]]; then
  rm -rf build
fi

cmake -S "${ROOT}" -B "${ROOT}/build" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -DENABLE_CUDA="${CLASSICAL_SPIN_ENABLE_CUDA}"

cmake --build "${ROOT}/build" -j "$(nproc)"

echo "Build finished: ${ROOT}/build"
