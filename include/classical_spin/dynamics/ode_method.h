#pragma once
/**
 * ode_method.h — the integrator names accepted by the Lattice spin-dynamics
 * drivers, parsed and validated in ONE place.
 *
 * Previously the name was compared against string literals at every
 * integrate call and an unknown name (a typo such as "rk45") silently ran
 * dopri5. parse_ode_method throws std::invalid_argument listing the valid
 * names instead; it is called when the configuration is loaded and again at
 * every API entry point.
 *
 * Families (see Lattice::integrate_on_grid for how each is driven):
 *   fixed step   euler, rk2, rk4, adams_bashforth, adams_moulton: dt is the
 *                step; the output spacing must be a multiple of it.
 *   adaptive     dopri5 and bulirsch_stoer use dense output (steps are free,
 *                samples are interpolated at the exact grid times);
 *                cash_karp54 and rkf78 have no dense output in odeint and
 *                step exactly onto every sample time instead. dt is only the
 *                initial step.
 *   geometric    spherical_midpoint, depondt, color_split, color_split4
 *                (dynamics/spin_integrators.h): fixed step, |S_i| exact,
 *                the only family that supports Langevin noise.
 *
 * rosenbrock4 / implicit_euler were removed: they built a dense N x N
 * finite-difference Jacobian (N RHS calls and O(N²) memory per step, 7 GB at
 * 10^4 sites) for an equation that is not stiff. spherical_midpoint is the
 * implicit (and symplectic, norm-preserving) replacement.
 */

#include <stdexcept>
#include <string>
#include <string_view>

#include "classical_spin/dynamics/spin_integrators.h"

namespace classical_spin::dynamics {

enum class OdeMethod {
    Euler, RK2, RK4, AdamsBashforth5, AdamsMoulton5,   // fixed-step Runge-Kutta / multistep
    CashKarp54, Dopri5, Fehlberg78, BulirschStoer,     // error-controlled
    SphericalMidpoint, Depondt, ColorSplit, ColorSplit4 // geometric (spin_integrators.h)
};

inline const char* valid_ode_method_names() {
    return "euler, rk2 (midpoint), rk4, adams_bashforth (ab), adams_moulton (am), "
           "rk5 (rkck54, rk54, rkf54: Cash-Karp 5(4)), dopri5, rk78 (rkf78), bulirsch_stoer (bs), "
           "spherical_midpoint (implicit_midpoint), depondt (heun_sphere), "
           "color_split (suzuki_trotter), color_split4";
}

inline OdeMethod parse_ode_method(std::string_view name) {
    if (name == "euler") return OdeMethod::Euler;
    if (name == "rk2" || name == "midpoint") return OdeMethod::RK2;
    if (name == "rk4") return OdeMethod::RK4;
    if (name == "adams_bashforth" || name == "ab") return OdeMethod::AdamsBashforth5;
    if (name == "adams_moulton" || name == "am") return OdeMethod::AdamsMoulton5;
    // Boost.Odeint has no Fehlberg 5(4) stepper; rk54/rkf54 have always meant Cash-Karp 5(4).
    if (name == "rk5" || name == "rkck54" || name == "rk54" || name == "rkf54") return OdeMethod::CashKarp54;
    if (name == "dopri5") return OdeMethod::Dopri5;
    if (name == "rk78" || name == "rkf78") return OdeMethod::Fehlberg78;
    if (name == "bulirsch_stoer" || name == "bs") return OdeMethod::BulirschStoer;
    if (is_geometric_method(name)) {
        switch (parse_geometric_method(name)) {
            case Method::SphericalMidpoint: return OdeMethod::SphericalMidpoint;
            case Method::Depondt:           return OdeMethod::Depondt;
            case Method::ColorSplit:        return OdeMethod::ColorSplit;
            case Method::ColorSplit4:       return OdeMethod::ColorSplit4;
        }
    }
    if (name == "rosenbrock4" || name == "rb4" || name == "implicit_euler" || name == "ie")
        throw std::invalid_argument(
            "integrator '" + std::string(name) + "' was removed: it built a dense N x N finite-difference "
            "Jacobian (O(N^2) memory) for a non-stiff equation. Use spherical_midpoint (implicit, "
            "symplectic, norm-preserving) instead");
    throw std::invalid_argument("unknown integrator '" + std::string(name) + "' (valid: " +
                                valid_ode_method_names() + ")");
}

inline const char* ode_method_name(OdeMethod m) {
    switch (m) {
        case OdeMethod::Euler:             return "euler";
        case OdeMethod::RK2:               return "rk2";
        case OdeMethod::RK4:               return "rk4";
        case OdeMethod::AdamsBashforth5:   return "adams_bashforth";
        case OdeMethod::AdamsMoulton5:     return "adams_moulton";
        case OdeMethod::CashKarp54:        return "rkck54";
        case OdeMethod::Dopri5:            return "dopri5";
        case OdeMethod::Fehlberg78:        return "rkf78";
        case OdeMethod::BulirschStoer:     return "bulirsch_stoer";
        case OdeMethod::SphericalMidpoint: return "spherical_midpoint";
        case OdeMethod::Depondt:           return "depondt";
        case OdeMethod::ColorSplit:        return "color_split";
        case OdeMethod::ColorSplit4:       return "color_split4";
    }
    return "?";
}

inline bool is_geometric(OdeMethod m) {
    return m == OdeMethod::SphericalMidpoint || m == OdeMethod::Depondt ||
           m == OdeMethod::ColorSplit || m == OdeMethod::ColorSplit4;
}

inline bool is_adaptive(OdeMethod m) {
    return m == OdeMethod::CashKarp54 || m == OdeMethod::Dopri5 ||
           m == OdeMethod::Fehlberg78 || m == OdeMethod::BulirschStoer;
}

inline Method to_geometric(OdeMethod m) {
    switch (m) {
        case OdeMethod::SphericalMidpoint: return Method::SphericalMidpoint;
        case OdeMethod::Depondt:           return Method::Depondt;
        case OdeMethod::ColorSplit:        return Method::ColorSplit;
        case OdeMethod::ColorSplit4:       return Method::ColorSplit4;
        default: break;
    }
    throw std::invalid_argument(std::string("'") + ode_method_name(m) + "' is not a geometric integrator");
}

}  // namespace classical_spin::dynamics
