#pragma once
/**
 * gpu_handle.h — ownership of the opaque GPU data handles held by Lattice and
 * MixedLattice.
 *
 * The lattices used to keep a raw `GPU...Handle*` that no destructor freed
 * (device memory leaked once per object, e.g. per parameter-sweep point) and
 * that the implicit copy operations copied (two objects sharing, and later
 * both freeing, one device context; MixedLattice clones in the OpenMP 2DCS
 * path did exactly that). DeviceHandle frees the handle in its destructor and
 * never shares it: a copy starts EMPTY (the copy re-uploads its own
 * Hamiltonian on first GPU use), a move transfers ownership.
 */

#include <utility>

namespace classical_spin {
namespace gpu {

template <class Handle, void (*Destroy)(Handle*)>
class DeviceHandle {
public:
    DeviceHandle() = default;
    ~DeviceHandle() { reset(); }

    DeviceHandle(const DeviceHandle&) noexcept {}
    DeviceHandle& operator=(const DeviceHandle& other) noexcept {
        if (this != &other) reset();
        return *this;
    }
    DeviceHandle(DeviceHandle&& other) noexcept : h_(std::exchange(other.h_, nullptr)) {}
    DeviceHandle& operator=(DeviceHandle&& other) noexcept {
        if (this != &other) {
            reset();
            h_ = std::exchange(other.h_, nullptr);
        }
        return *this;
    }

    Handle* get() const noexcept { return h_; }
    explicit operator bool() const noexcept { return h_ != nullptr; }

    /// Free the current handle (errors while freeing are swallowed: this runs
    /// in destructors) and take ownership of `h`.
    void reset(Handle* h = nullptr) noexcept {
        if (h_ && h_ != h) {
            try {
                Destroy(h_);
            } catch (...) {
            }
        }
        h_ = h;
    }

private:
    Handle* h_ = nullptr;
};

}  // namespace gpu
}  // namespace classical_spin
