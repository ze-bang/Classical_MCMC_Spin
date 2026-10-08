/**
 * simple_linear_alg.cpp - Implementation of linear algebra utilities
 * 
 * This file contains the implementation of non-template functions that
 * were previously inline in the header file. Template functions remain
 * in the header.
 */

#include "classical_spin/core/simple_linear_alg.h"

#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif

// Per-thread Lehmer state. A thread whose generation lags the global one
// (fresh OpenMP worker, or a reseed happened since its last draw) derives
// its stream from (master seed, thread id) on its next draw.
thread_local unsigned __int128 lehman_state = 0;

namespace {
std::atomic<unsigned long long> lehman_master_seed{0x853C49E6748FEA9BULL};
std::atomic<unsigned long long> lehman_generation{1};
thread_local unsigned long long lehman_thread_generation = 0;
thread_local bool lehman_has_spare_normal = false;
thread_local double lehman_spare_normal = 0.0;

constexpr unsigned __int128 kLehmerMultiplier =
    (unsigned __int128)0x12e15e35b500f16eULL << 64 | 0x2e714eb2b37916a5ULL;

// Expand a 64-bit key into a well-mixed odd 128-bit state and step past the
// first outputs (whose high bits are weak for small states).
inline unsigned __int128 expand_seed(unsigned long long key) {
    const unsigned long long hi = splitmix64(key);
    const unsigned long long lo = splitmix64(key ^ 0xD1B54A32D192ED03ULL) | 1ULL;
    unsigned __int128 state = ((unsigned __int128)hi << 64) | lo;
    for (int i = 0; i < 4; ++i) state *= kLehmerMultiplier;
    return state;
}

inline void set_thread_state(unsigned __int128 state) {
    lehman_state = state;
    lehman_thread_generation = lehman_generation.load(std::memory_order_acquire);
    lehman_has_spare_normal = false;
}

inline void lazy_seed_thread() {
    // Key the stream on the full OpenMP thread ancestry, not just the
    // innermost thread number: threads of different outer teams that first
    // draw inside nested regions must not share a stream.
    unsigned long long tid = 0;
#ifdef _OPENMP
    const int level = omp_get_level();
    for (int l = 1; l <= level; ++l)
        tid = splitmix64(tid ^ (static_cast<unsigned long long>(omp_get_ancestor_thread_num(l)) +
                                0x9E3779B97F4A7C15ULL * static_cast<unsigned long long>(l)));
#endif
    const unsigned long long master = lehman_master_seed.load(std::memory_order_relaxed);
    set_thread_state(expand_seed(master ^ splitmix64(tid + 0x632BE59BD9B4E019ULL)));
}
} // namespace

// Structure constants initialization
SpinTensor3 StructureConstants::SU2_structure_constant() {
    SpinTensor3 result(3, MatrixXd::Zero(3, 3));
    
    // Set the antisymmetric structure constants for SU(2)
    // f^{ijk} = epsilon^{ijk}
    set_permutation(result, 0, 1, 2, 1.0);
    
    return result;
}

SpinTensor3 StructureConstants::SU3_structure_constant() {
    SpinTensor3 result(8, MatrixXd::Zero(8, 8));
    
    // Set the Gell-Mann structure constants
    set_permutation(result, 0, 1, 2, 1.0);
    set_permutation(result, 0, 3, 6, 0.5);
    set_permutation(result, 0, 4, 5, -0.5);
    set_permutation(result, 1, 3, 5, 0.5);
    set_permutation(result, 1, 4, 6, 0.5);
    set_permutation(result, 2, 3, 4, 0.5);
    set_permutation(result, 2, 5, 6, -0.5);
    set_permutation(result, 3, 4, 7, sqrt(3.0)/2.0);
    set_permutation(result, 5, 6, 7, sqrt(3.0)/2.0);
    
    return result;
}

void StructureConstants::set_permutation(SpinTensor3& A, size_t a, size_t b, size_t c, double val) {
    if (a >= A.size() || b >= A[a].cols() || c >= A[a].rows()) {
        throw out_of_range("Index out of range in set_permutation");
    }
    if (a == b || b == c || a == c) {
        throw invalid_argument("Indices must be distinct in set_permutation");
    }
    
    // Set all permutations with appropriate signs
    A[a](b, c) = val;
    A[a](c, b) = -val;
    A[b](a, c) = -val;
    A[b](c, a) = val;
    A[c](a, b) = val;
    A[c](b, a) = -val;
}

// Global structure constants (definition)
const SpinTensor3& get_SU2_structure() {
    static const SpinTensor3 SU2_structure = StructureConstants::SU2_structure_constant();
    return SU2_structure;
}

const SpinTensor3& get_SU3_structure() {
    static const SpinTensor3 SU3_structure = StructureConstants::SU3_structure_constant();
    return SU3_structure;
}

// Cross product for SU(2) (standard 3D cross product)
SpinVector cross_prod_SU2(const SpinVector& a, const SpinVector& b) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::invalid_argument("SU2 cross product requires 3D vectors");
    }
    
    SpinVector result(3);
    result(0) = a(1) * b(2) - a(2) * b(1);
    result(1) = a(2) * b(0) - a(0) * b(2);
    result(2) = a(0) * b(1) - a(1) * b(0);
    
    return result;
}

// Cross product for SU(3) using structure constants
// Computes (a × b)_i = sum_{jk} f_{ijk} a_j b_k
SpinVector cross_prod_SU3(const SpinVector& a, const SpinVector& b) {
    if (a.size() != 8 || b.size() != 8) {
        throw std::invalid_argument("SU3 cross product requires 8D vectors");
    }
    
    const auto& f = get_SU3_structure();
    SpinVector result = SpinVector::Zero(8);
    
    for (size_t i = 0; i < 8; ++i) {
        // result(i) = sum_j a_j * (sum_k f[i](j,k) * b_k) = a^T * f[i] * b
        result(i) = a.dot(f[i] * b);
    }
    
    return result;
}

// General cross product that dispatches based on dimension
SpinVector cross_product(const SpinVector& a, const SpinVector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have same dimension for cross product");
    }
    
    if (a.size() == 3) {
        return cross_prod_SU2(a, b);
    } else if (a.size() == 8) {
        return cross_prod_SU3(a, b);
    } else {
        throw std::invalid_argument("Cross product only defined for N=3 (SU2) or N=8 (SU3)");
    }
}

// Trilinear contraction: sum_ijk M_ijk * a_i * b_j * c_k
double contract_trilinear(const SpinTensor3& M, const SpinVector& a, 
                         const SpinVector& b, const SpinVector& c) {
    double result = 0.0;
    size_t N1 = a.size();
    size_t N2 = b.size();
    
    for (size_t i = 0; i < N1; ++i) {
        for (size_t j = 0; j < N2; ++j) {
            result += a(i) * b(j) * M[i].row(j).dot(c);
        }
    }
    
    return result;
}

// Trilinear field contraction: returns vector from trilinear * two vectors
SpinVector contract_trilinear_field(const SpinTensor3& M, 
                                    const SpinVector& b, 
                                    const SpinVector& c) {
    size_t N1 = M.size();
    SpinVector result = SpinVector::Zero(N1);
    
    for (size_t i = 0; i < N1; ++i) {
        result(i) = (M[i] * c).dot(b);
    }
    
    return result;
}

// Transpose 3D tensor (rearrange indices)
SpinTensor3 transpose3D(const SpinTensor3& T, size_t N1, size_t N2, size_t N3) {
    SpinTensor3 result(N2, MatrixXd::Zero(N3, N1));
    
    for (size_t i = 0; i < N1; ++i) {
        for (size_t j = 0; j < N2; ++j) {
            for (size_t k = 0; k < N3; ++k) {
                result[j](k, i) = T[i](j, k);
            }
        }
    }
    
    return result;
}

// Random number generation (see the header for the stream/seeding model).
void seed_lehman(unsigned __int128 seed) {
    const unsigned long long master =
        static_cast<unsigned long long>(seed) ^ static_cast<unsigned long long>(seed >> 64);
    lehman_master_seed.store(master, std::memory_order_relaxed);
    lehman_generation.fetch_add(1, std::memory_order_acq_rel);
    set_thread_state(expand_seed(master));
}

uint64_t lehman_next() {
    if (__builtin_expect(lehman_thread_generation !=
                         lehman_generation.load(std::memory_order_relaxed), 0)) {
        lazy_seed_thread();
    }
    const uint64_t result = static_cast<uint64_t>(lehman_state >> 64);
    lehman_state *= kLehmerMultiplier;
    return result;
}

double random_double_lehman(double min, double max) {
    const double u = static_cast<double>(lehman_next() >> 11) * 0x1.0p-53;  // [0, 1)
    return min + (max - min) * u;
}

size_t random_index_lehman(size_t size) {
    return static_cast<size_t>(
        (static_cast<unsigned __int128>(lehman_next()) * static_cast<uint64_t>(size)) >> 64);
}

int random_int_lehman(int size) {
    return static_cast<int>(random_index_lehman(static_cast<size_t>(size)));
}

double random_normal_lehman() {
    if (lehman_has_spare_normal) {
        lehman_has_spare_normal = false;
        return lehman_spare_normal;
    }
    double u, v, s;
    do {
        u = random_double_lehman(-1.0, 1.0);
        v = random_double_lehman(-1.0, 1.0);
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    const double f = std::sqrt(-2.0 * std::log(s) / s);
    lehman_spare_normal = v * f;
    lehman_has_spare_normal = true;
    return u * f;
}

void random_point_on_sphere(double* out, size_t n, double radius) {
    if (n == 3) {
        double u1, u2, s;
        do {
            u1 = random_double_lehman(-1.0, 1.0);
            u2 = random_double_lehman(-1.0, 1.0);
            s = u1 * u1 + u2 * u2;
        } while (s >= 1.0);
        const double f = 2.0 * std::sqrt(1.0 - s);
        out[0] = radius * f * u1;
        out[1] = radius * f * u2;
        out[2] = radius * (1.0 - 2.0 * s);
        return;
    }
    double sum_sq;
    do {
        sum_sq = 0.0;
        for (size_t i = 0; i < n; ++i) {
            out[i] = random_normal_lehman();
            sum_sq += out[i] * out[i];
        }
    } while (sum_sq < 1e-300);
    const double scale = radius / std::sqrt(sum_sq);
    for (size_t i = 0; i < n; ++i) out[i] *= scale;
}

void seed_lehman_from_rank(unsigned long long key) {
    const unsigned long long master = lehman_master_seed.load(std::memory_order_relaxed);
    seed_lehman(splitmix64(master ^ splitmix64(key ^ 0x9E3779B97F4A7C15ULL)));
}

void seed_lehman_thread(unsigned long long key) {
    const unsigned long long master = lehman_master_seed.load(std::memory_order_relaxed);
    set_thread_state(expand_seed(master ^ splitmix64(key + 0xA0761D6478BD642FULL)));
}

void seed_lehman_stream(unsigned long long stream_seed) {
    set_thread_state(expand_seed(splitmix64(stream_seed ^ 0x8BB84B93962EACC9ULL)));
}

unsigned long long lehman_master_seed_value() {
    return lehman_master_seed.load(std::memory_order_relaxed);
}

unsigned long long derive_seed_from_master(unsigned long long key) {
    const unsigned long long master =
        lehman_master_seed.load(std::memory_order_relaxed);
    return splitmix64(master ^ (key * 0xD1B54A32D192ED03ULL));
}

// Norm functions for arrays of vectors
double norm_average(const std::vector<SpinVector>& vecs) {
    double total = 0.0;
    size_t count = 0;
    
    for (const auto& v : vecs) {
        total += v.squaredNorm();
        count += v.size();
    }
    
    return total / count;
}
