#pragma once
/**
 * device_select.h — one place that decides whether this process can use a
 * CUDA device, and binds it to one.
 *
 * The runners used to call cudaGetDeviceCount on an uninitialised int without
 * checking the return code, bind by GLOBAL rank modulo the device count, and
 * print "falling back to CPU" while still taking the GPU path. Here:
 *   - the CUDA return codes are checked (no driver / no device -> false);
 *   - the device is chosen by the node-local rank, read from the launcher's
 *     environment (Open MPI, MPICH/Intel MPI, MVAPICH, Slurm) so that the
 *     call is NOT collective and is safe from any subset of ranks; the global
 *     rank is the fallback;
 *   - the decision is made once per process and reported once; later calls
 *     return the cached answer.
 * Callers pass the returned flag on as use_gpu (false = run on the CPU).
 */

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace classical_spin {
namespace gpu {

namespace detail {

struct DeviceState {
    std::once_flag once;
    bool usable = false;
    int device = -1;
    int count = 0;
    std::string reason;
};

inline DeviceState& device_state() {
    static DeviceState s;
    return s;
}

/// Node-local rank from the MPI launcher's environment, or -1.
inline int local_rank_from_environment() {
    for (const char* var : {"OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "MV2_COMM_WORLD_LOCAL_RANK",
                            "SLURM_LOCALID", "PMI_LOCAL_RANK"}) {
        if (const char* v = std::getenv(var)) {
            char* end = nullptr;
            const long r = std::strtol(v, &end, 10);
            if (end != v && r >= 0) return int(r);
        }
    }
    return -1;
}

inline void probe_device(int fallback_rank) {
    DeviceState& s = device_state();
#ifdef CUDA_ENABLED
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        s.reason = (err != cudaSuccess) ? std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(err)
                                        : std::string("no CUDA device found");
        (void)cudaGetLastError();  // clear the sticky error
        return;
    }
    const int local = local_rank_from_environment();
    const int device = ((local >= 0) ? local : (fallback_rank >= 0 ? fallback_rank : 0)) % count;
    const cudaError_t set = cudaSetDevice(device);
    if (set != cudaSuccess) {
        s.reason = std::string("cudaSetDevice(") + std::to_string(device) + ") failed: " + cudaGetErrorString(set);
        (void)cudaGetLastError();
        return;
    }
    s.usable = true;
    s.device = device;
    s.count = count;
#else
    (void)fallback_rank;
    s.reason = "compiled without CUDA (CUDA_ENABLED)";
#endif
}

}  // namespace detail

/**
 * Whether GPU acceleration should be used. When `requested`, probes the
 * devices once per process and binds this process to device
 * (node-local rank) mod (device count); prints one line either way (the
 * device, or the reason for running on the CPU). `rank` labels the message
 * (< 0: no label) and is the fallback when the launcher exports no local
 * rank. Returns false when not requested or not usable.
 */
inline bool select_device(bool requested, int rank = -1) {
    if (!requested) return false;
    detail::DeviceState& s = detail::device_state();
    std::call_once(s.once, [&] {
        detail::probe_device(rank);
        const std::string who = (rank >= 0) ? "[Rank " + std::to_string(rank) + "] " : std::string();
        if (s.usable)
            std::cout << who << "GPU " << s.device << " of " << s.count << std::endl;
        else
            std::cerr << who << "Warning: GPU requested but not usable (" << s.reason << "); running on the CPU"
                      << std::endl;
    });
    return s.usable;
}

/**
 * Library-side guard for the GPU dispatch: true when a device is usable
 * (probes and reports once if select_device() was never called).
 */
inline bool device_available() { return select_device(true); }

}  // namespace gpu
}  // namespace classical_spin
