// np/harness.hpp -- tiny test harness + convergence-order estimator.
// Provided infrastructure: you never need to edit this file.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace np {

// Thrown by NP_TODO() from an unimplemented exercise stub.
struct TodoError : std::runtime_error {
    explicit TodoError(const std::string& what) : std::runtime_error(what) {}
};
// Thrown by a failed NP_CHECK.
struct CheckError : std::runtime_error {
    explicit CheckError(const std::string& what) : std::runtime_error(what) {}
};

struct TestCase {
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

#define NP_TEST(name)                                                    \
    static void name();                                                  \
    static ::np::Registrar np_reg_##name(#name, &name);                  \
    static void name()

#define NP_TODO(what)                                                    \
    throw ::np::TodoError(std::string("not implemented: ") + (what))

#define NP_CHECK(cond)                                                   \
    do {                                                                 \
        if (!(cond))                                                     \
            throw ::np::CheckError(std::string(__FILE__) + ":" +         \
                                   std::to_string(__LINE__) +            \
                                   "  NP_CHECK(" #cond ")");             \
    } while (0)

#define NP_CHECK_MSG(cond, ...)                                          \
    do {                                                                 \
        if (!(cond)) {                                                   \
            char np_buf_[512];                                           \
            std::snprintf(np_buf_, sizeof np_buf_, __VA_ARGS__);         \
            throw ::np::CheckError(std::string(__FILE__) + ":" +         \
                                   std::to_string(__LINE__) + "  " +     \
                                   np_buf_);                             \
        }                                                                \
    } while (0)

#define NP_CLOSE(a, b, tol)                                              \
    do {                                                                 \
        double np_a_ = (a), np_b_ = (b), np_t_ = (tol);                  \
        if (!(std::fabs(np_a_ - np_b_) <= np_t_))                        \
            NP_CHECK_MSG(false, #a " = %.12g  !=  " #b " = %.12g   "     \
                                "(|diff| = %.3g > tol %.3g)",            \
                         np_a_, np_b_, std::fabs(np_a_ - np_b_), np_t_); \
    } while (0)

// ---------------------------------------------------------------------------
// Convergence-order estimation: least-squares slope of log(err) vs log(h).
// ---------------------------------------------------------------------------
inline double fit_order(const std::vector<double>& hs,
                        const std::vector<double>& errs) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0;
    for (size_t i = 0; i < hs.size(); ++i) {
        if (!(errs[i] > 0) || !std::isfinite(errs[i])) continue;
        double x = std::log(hs[i]), y = std::log(errs[i]);
        sx += x; sy += y; sxx += x * x; sxy += x * y; ++n;
    }
    if (n < 2) return 0.0;
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

inline void print_convergence(const char* label,
                              const std::vector<double>& hs,
                              const std::vector<double>& errs) {
    std::printf("      %-28s  %10s %12s %8s\n", label, "h", "error", "rate");
    for (size_t i = 0; i < hs.size(); ++i) {
        double rate = (i == 0) ? 0.0
                               : std::log(errs[i - 1] / errs[i]) /
                                     std::log(hs[i - 1] / hs[i]);
        if (i == 0)
            std::printf("      %-28s  %10.3g %12.4e %8s\n", "", hs[i], errs[i], "-");
        else
            std::printf("      %-28s  %10.3g %12.4e %8.2f\n", "", hs[i], errs[i], rate);
    }
    std::printf("      %-28s  fitted order = %.3f\n", "", fit_order(hs, errs));
}

// Assert the fitted convergence order; prints the table on failure.
#define NP_CHECK_ORDER(label, hs, errs, expected, tol)                       \
    do {                                                                     \
        double np_p_ = ::np::fit_order((hs), (errs));                        \
        if (!(std::fabs(np_p_ - (expected)) <= (tol))) {                     \
            ::np::print_convergence((label), (hs), (errs));                  \
            NP_CHECK_MSG(false, "%s: fitted order %.3f, expected %.3f +- %.3f", \
                         (label), np_p_, (double)(expected), (double)(tol)); \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
inline int run_all(int argc, char** argv) {
    const char* filter = (argc > 1) ? argv[1] : nullptr;
    int pass = 0, fail = 0, todo = 0, skip = 0;
    for (const auto& t : registry()) {
        if (filter && !std::strstr(t.name, filter)) { ++skip; continue; }
        std::printf("[ RUN  ] %s\n", t.name);
        std::fflush(stdout);
        try {
            t.fn();
            std::printf("[  OK  ] %s\n", t.name);
            ++pass;
        } catch (const TodoError& e) {
            std::printf("[ TODO ] %s -- %s\n", t.name, e.what());
            ++todo;
        } catch (const std::exception& e) {
            std::printf("[ FAIL ] %s\n         %s\n", t.name, e.what());
            ++fail;
        }
    }
    std::printf("\n%d passed, %d failed, %d todo%s\n", pass, fail, todo,
                skip ? " (some filtered out)" : "");
    return (fail || todo) ? 1 : 0;
}

}  // namespace np

#define NP_MAIN() \
    int main(int argc, char** argv) { return ::np::run_all(argc, argv); }
