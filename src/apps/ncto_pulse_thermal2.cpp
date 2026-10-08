/**
 * ncto_pulse_thermal2.cpp — THz pulse at finite T, instrumented for the LINEAR channel.
 *
 * Supersedes ncto_pulse_thermal.cpp.  Three things were wrong or missing there:
 *
 *  (1) THERMOSTAT.  It set only langevin_temperature, so it ran the CLASSICAL
 *      white-noise bath.  Classical spins overstate the entropy: the measured
 *      classical tipping temperature is T*_cl ~ 6-6.5 K, so a grid at T = 6/12/20 K
 *      is mostly ABOVE the model's own ordering temperature and melts rather than
 *      nucleates.  With the Bose-coloured thermostat (langevin_quantum) 3Q survives
 *      to ~25 K, i.e. the real T_N = 27 K, with no rescaling.
 *
 *  (2) NOISE CONTINUITY.  It chopped the run into dt_save-length integrate_langevin
 *      calls with a fresh seed each time.  Each chunk is a valid stationary segment,
 *      so this is not wrong for white noise, but it truncates the noise correlations
 *      at the chunk length — and the Bose noise correlation time is hbar/k_B T, which
 *      is 1.9 code units at 6 K versus a 2.0-unit chunk.  It also regenerated a
 *      20.5-unit spectral block (langevin_block = 4096 steps) to consume 2 units of
 *      it, ~10x wasted FFT work.  Here the whole trajectory is ONE integrate_langevin
 *      call and observables come out through the on_save observer hook.
 *
 *  (3) THE OBSERVABLE THAT MATTERS.  The E1 doublet Q transforms as E; zigzag has a
 *      nonzero E-component, and 3Q — being C3 symmetric — has EXACTLY zero (see
 *      ESTABLISHED.md 3b: relaxed |Q| = 0 for 3Q, 1.0637e-2 for ZZ).  So the
 *      symmetry-allowed bilinear -g(Q.phi) is a bias that acts on zigzag and is
 *      forbidden from acting on 3Q.  With all quadratic couplings zero in the
 *      leading-order config (lambda_E1_J7_0 = 0 included), H_sp-ph is PURELY LINEAR
 *      in Q, so
 *
 *          spin_phonon_energy() / N   ==   the instantaneous linear tilt, in meV/site
 *
 *      is exactly the quantity the campaign has been inferring indirectly.  It is
 *      emitted here directly, along with dH/dQ (= g*phi, the force conjugate to the
 *      doublet, zero in 3Q by symmetry) and an Euler self-check that fails loudly if
 *      a quadratic coupling is switched on and the "purely linear" reading breaks.
 *
 * Calibration built in at startup: load the ZZ seed, relax Q, and the code must
 * reproduce ESTABLISHED.md 3b — |Q| = 1.0637e-2, E_tilt = -34.1 ueV/site (the
 * coupling term; the +17.1 ueV elastic term brings the net gain to -16.5).
 *
 * Usage:
 *   ncto_pulse_thermal2 CFG SEED J7 E0 theta t_end dt_save T
 *                       [disorder|none] [rngseed] [lin|circ] [quantum 0|1] [t_therm]
 *
 * T is k_B T in meV: 6 K = 0.517, 12 K = 1.034, 20 K = 1.723, 27 K = 2.327.
 * Time is in hbar/meV = 0.658212 ps.  t_therm is pre-pulse thermalisation, run in the
 * SAME integrate_langevin call (so the noise stream is continuous across it) and
 * reported at negative t_ps.
 *
 * "circ" builds circular polarization out of the two-pulse drive: pulse 2 at
 * theta + pi/2 with a -pi/2 phase, both amplitudes scaled by 1/sqrt(2) so that the
 * time-integrated |E|^2 — the fluence — matches the linear case exactly.  Circular
 * holds |Q| constant instead of oscillating it through zero, but rotates the favoured
 * zigzag domain at the carrier frequency; that is a near parameter-free discriminator.
 */
#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/phonon_lattice.h"
#include "classical_spin/lattice/phonon_config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>

using namespace std;

int main(int argc, char** argv) {
    if (argc < 9) {
        cerr << "Usage: " << argv[0]
             << " config seed.txt J7 E0 theta t_end dt_save T"
                " [disorder|none] [rngseed] [lin|circ] [quantum 0|1] [t_therm]"
                " [nn_sigma] [nn_seed] [nn_file]\n"
                "  nn_sigma > 0 applies bond-anisotropic exchange disorder: every unique NN bond's\n"
                "  3x3 matrix is scaled by exp(N(0, nn_sigma)) — a random FIELD on the nematic\n"
                "  order parameter (breaks C3 locally). Per-plaquette J7 disorder cannot do that.\n";
        return 1;
    }
    const string config_file = argv[1];
    const string seed_file   = argv[2];
    const double j7      = atof(argv[3]);
    const double E0      = atof(argv[4]);
    const double theta   = atof(argv[5]);
    const double t_end   = atof(argv[6]);
    const double dt_save = atof(argv[7]);
    const double Temp    = atof(argv[8]);
    const string dis     = (argc >  9) ? argv[9]  : "none";
    const uint64_t rseed = (argc > 10) ? strtoull(argv[10], nullptr, 10) : 12345ULL;
    const string pol     = (argc > 11) ? argv[11] : "lin";
    const int    quantum = (argc > 12) ? atoi(argv[12]) : 1;
    const double t_therm = (argc > 13) ? atof(argv[13]) : 30.0;

    if (pol != "lin" && pol != "circ") {
        cerr << "ERROR: polarization must be 'lin' or 'circ', got '" << pol << "'\n";
        return 1;
    }

    // Trap 3: the library prints setup diagnostics to stdout, which would interleave
    // with the CSV.  Send cout to cerr and write the CSV through its own stream.
    std::ostream csv(std::cout.rdbuf());
    std::cout.rdbuf(std::cerr.rdbuf());

    SpinConfig config = SpinConfig::from_file(config_file);
    UnitCell uc = build_phonon_honeycomb(config);
    PhononLattice lattice(uc, config.lattice_size[0], config.lattice_size[1],
                          config.lattice_size[2], config.spin_length);

    SpinPhononCouplingParams sp; PhononParams ph; DriveParams dr;
    TimeDependentSpinPhononParams td;
    build_phonon_params(config, sp, ph, dr, td);
    sp.J7 = j7;

    // ---- drive: linear, or circular at matched fluence ----------------------
    // Linear   : |E|^2 = E0^2 env^2 cos^2(w dt)      -> cycle average E0^2 env^2 / 2
    // Circular : |E|^2 = (E0/sqrt2)^2 env^2 (c^2+s^2) ->               E0^2 env^2 / 2
    // Same time-integrated |E|^2, so the deposited energy is matched by construction
    // and any difference is polarization physics, not fluence.
    dr.theta_1 = theta;
    if (pol == "circ") {
        const double a = E0 / std::sqrt(2.0);
        dr.E0_1 = a;
        dr.E0_2 = a;
        dr.omega_2 = dr.omega_1;
        dr.sigma_2 = dr.sigma_1;
        dr.t_2     = dr.t_1;
        dr.theta_2 = theta + M_PI / 2.0;
        dr.phi_2   = dr.phi_1 - M_PI / 2.0;
    } else {
        dr.E0_1 = E0;
        dr.E0_2 = 0.0;
    }
    // Pre-pulse thermalisation happens inside the same integration, so shift the
    // pulse centres past it rather than restarting the integrator.
    dr.t_1 += t_therm;
    dr.t_2 += t_therm;

    lattice.set_parameters(sp, ph, dr);
    lattice.set_time_dependent_spin_phonon(td);
    // Extra lattice modes and cubic anharmonic transfers from the config (n_extra_modes,
    // mode<i>_irrep/omega/gamma/quartic/lamJ7..., n_anharmonic, anh<j>_target/lam/lamp/g):
    // the Subedi Q_R Q_IR^2 machinery.  Absent keys -> no change to the primary E1 mode.
    build_lattice_modes(config, lattice);
    lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.05);
    lattice.langevin_temperature = Temp;

    // ---- thermostat ---------------------------------------------------------
    lattice.langevin_quantum = (quantum != 0);
    lattice.langevin_block   = static_cast<int>(config.get_param("langevin_block", 4096.0));

    Eigen::Vector3d B;
    B << config.field_strength * config.field_direction[0],
         config.field_strength * config.field_direction[1],
         config.field_strength * config.field_direction[2];
    lattice.set_field(B);

    if (dis != "none") lattice.apply_plaquette_j7_disorder_from_file(dis);

    // ---- optional bond-anisotropic (nematic) exchange disorder --------------
    // Perfect 3Q is C3-symmetric, so every E-symmetric coupling is conjugate to an
    // order parameter that is identically zero on it — no handle.  A random scale
    // on each NN bond is a random FIELD on that order parameter: it carves local
    // minima with phi != 0 and lowers the local stiffness, which (stage 6) is the
    // barrier.  Per-plaquette J7 disorder is A1 per plaquette and only shifts the
    // local energy difference, which is not the barrier.
    const double   nn_sigma = (argc > 14) ? atof(argv[14]) : 0.0;
    const uint64_t nn_seed  = (argc > 15) ? strtoull(argv[15], nullptr, 10) : 7001ULL;
    const string   nn_file  = (argc > 16) ? argv[16] : "";
    // arg 22: nn_xi — correlation length (lattice spacings) of the bond disorder.  0 or 1 =
    // independent per bond (as before).  Otherwise a per-SITE Gaussian field is smoothed
    // by k = round(xi^2) rounds of neighbour averaging (diffusive kernel, correlation
    // length ~ sqrt(k)), renormalised to rms nn_sigma, and each bond takes
    // exp((h_i + h_j)/2).  Mesoscopic (Na-order / strain) disorder rather than atomic.
    const double nn_xi = (argc > 22) ? atof(argv[22]) : 0.0;
    // args 24-26: ANNEAL the spliced initial condition before the measurement starts.
    // Splicing a zigzag band into a 3Q box puts the box 0.15 meV/site above its relaxed
    // energy — 40x the 4 ueV free-energy difference the runs are meant to resolve — and
    // that energy is released in ~2 ps as spin waves that destroy the M-point order of
    // the 3Q half even at 1 K (stage 30/31 diagnosis).  So dissipate it first: run
    // anneal_t code units at anneal_T with Gilbert damping anneal_alpha (large, so the
    // excess is removed rather than propagated), in a separate integration whose output
    // is discarded, then restore alpha and T for the measurement.
    const double anneal_T     = (argc > 23) ? atof(argv[23]) : 0.0;
    const double anneal_alpha = (argc > 24) ? atof(argv[24]) : 0.5;
    const double anneal_t     = (argc > 25) ? atof(argv[25]) : 0.0;
    // arg 27: match the inserted region's spin frame and M-point phase to the host before
    // splicing (see the splice block below).
    const bool splice_opt     = (argc > 26) ? (atoi(argv[26]) != 0) : false;
    if (nn_sigma > 0.0) {
        std::mt19937_64 g(nn_seed);
        std::normal_distribution<double> nrm(0.0, nn_sigma);
        std::vector<double> h;
        if (nn_xi > 1.0) {
            const size_t Ns = lattice.lattice_size;
            h.resize(Ns); for (auto& v : h) v = nrm(g);
            const int k = int(std::lround(nn_xi * nn_xi));
            std::vector<double> h2(Ns);
            for (int r = 0; r < k; ++r) {
                for (size_t i = 0; i < Ns; ++i) {
                    double s = h[i]; for (size_t j : lattice.nn_partners[i]) s += h[j];
                    h2[i] = s / (1.0 + lattice.nn_partners[i].size());
                }
                h.swap(h2);
            }
            double m = 0, v = 0; for (double x : h) { m += x; } m /= Ns;
            for (double x : h) v += (x - m) * (x - m);
            v = std::sqrt(v / Ns);
            for (auto& x : h) x = (x - m) * (nn_sigma / v);
        }
        // Many concurrent runs share (sigma, seed): write a per-process temp, apply
        // from it, then atomically rename so the shared record is never half-written.
        const string tmp = (nn_file.empty() ? string("nn_disorder") : nn_file)
                           + "." + std::to_string(getpid());
        size_t nb = 0;
        {
            std::ofstream f(tmp);
            f << "# NN exchange disorder: scale = exp(N(0," << nn_sigma
              << ")) per unique bond, seed " << nn_seed << ", xi " << nn_xi << "\n" << std::setprecision(12);
            for (size_t i = 0; i < lattice.lattice_size; ++i)
                for (size_t j : lattice.nn_partners[i])
                    if (i < j) {
                        const double x = h.empty() ? nrm(g) : 0.5 * (h[i] + h[j]);
                        f << i << " " << j << " " << std::exp(x) << "\n"; ++nb;
                    }
        }
        lattice.apply_nn_exchange_disorder_from_file(tmp);
        if (!nn_file.empty()) std::rename(tmp.c_str(), nn_file.c_str());
        else                  std::remove(tmp.c_str());
        cerr << "  nematic bond disorder: sigma=" << nn_sigma << " seed=" << nn_seed
             << " on " << nb << " bonds\n";
    }

    // ---- optional zigzag DROPLET inside the 3Q seed (lifetime / pinning studies) ------
    // args 17-19: droplet_R (lattice spacings, 0 = none)  zz_seed (path)  j7_step (meV)
    // A disc of radius R around the box centre is overwritten with the ZZ seed's spins;
    // j7_step is added to J7 on every hexagon whose centre lies inside the disc (a
    // mesoscopic pin: the wall sits on a step in the ring exchange).  With a ZZ reference
    // present the CSV gains three columns: f_ZZ (fraction of sites whose spin overlaps the
    // ZZ reference better than the 3Q one — the switched fraction), m_3Q and m_ZZ (mean
    // overlaps).  Without a ZZ reference the CSV is unchanged (16 columns).
    const double droplet_R = (argc > 17) ? atof(argv[17]) : 0.0;
    const string zz_seed   = (argc > 18) ? argv[18] : "";
    const double j7_step   = (argc > 19) ? atof(argv[19]) : 0.0;
    const bool   has_zz    = !zz_seed.empty();
    // arg 20: emit the scalar spin chirality kappa = <S_i . (S_a x S_b)> over NN pairs
    // (a < b by site index) around each site, per site.  Nonzero in the non-coplanar
    // triple-q state, zero in collinear zigzag: the manuscript's probe signal is
    // d(eta) ~ -d(kappa).  Opt-in so that the CSV layout of earlier stages is unchanged.
    const bool emit_kappa  = (argc > 20) ? (atoi(argv[20]) != 0) : false;
    // arg 21: droplet geometry "disc" (default) or "strip".  A strip is the band
    // |x - x_c| < R spanning the periodic box: two FLAT walls, no curvature term, so the
    // wall moves at v = mu * dF (a droplet below R* = sigma/dF shrinks even when zigzag
    // is favoured — stage 15B).  The strip also measures the wall tension: the relaxed
    // energy minus the uniform-3Q energy is 2 sigma L_wall / N per site.
    // geom "chi": the band |x - x_c| < R is overwritten with the TIME-REVERSED 3Q seed
    // (S -> -S, a degenerate 3Q state of opposite chirality kappa -> -kappa): two flat
    // chirality domain walls and NO zigzag seeded.  Tests whether a kappa wall grows a
    // zigzag core and whether it moves (stage 24).  With "strip" or "chi" and kappa on,
    // the CSV also carries the chirality of the band and of the two sea halves.
    const string geom      = (argc > 21) ? argv[21] : "disc";
    const bool   band_geom = (geom == "strip" || geom == "chi");
    const bool   rev_geom  = (geom == "chi" || geom == "chidisc");   // time-reversed 3Q inside (band or disc)
    // Box geometry: L x L rhombus, a1 = (1,0), a2 = (1/2, sqrt3/2).  All regions are
    // defined by the MINIMUM-IMAGE displacement from the box centroid A (L/2, L/2).
    // (Earlier stages used site N/2 = cell (18,0) as "centre", which lies on the box
    // edge: the disc was a half-disc and the strip a vertical band truncated by the
    // rhombus edge on its upper half.)  The band is |df1| < R in fractional a1 units,
    // i.e. walls PARALLEL TO a2 — a lattice direction, so the walls are periodic and
    // straight (perpendicular width 2R sin60, wall length L|a2| = L).
    const double Lbox = double(lattice.dim1), Lbox2 = double(lattice.dim2);   // rectangular boxes allowed
    Eigen::Matrix2d Abox; Abox << 1.0, 0.5, 0.0, std::sqrt(3.0) / 2.0;
    const Eigen::Matrix2d Aboxi = Abox.inverse();
    const Eigen::Vector2d fcen(Lbox / 2.0, Lbox2 / 2.0);
    auto dfrac = [&](const Eigen::Vector3d& p) {          // min-image fractional displacement from centroid
        Eigen::Vector2d f = Aboxi * Eigen::Vector2d(p.x(), p.y()) - fcen;
        f(0) -= Lbox * std::round(f(0) / Lbox); f(1) -= Lbox2 * std::round(f(1) / Lbox2);
        return f;
    };
    auto inside = [&](const Eigen::Vector3d& p, const Eigen::Vector3d& /*c unused*/) {
        const Eigen::Vector2d f = dfrac(p);
        return band_geom ? (std::abs(f(0)) < droplet_R) : ((Abox * f).norm() < droplet_R);
    };

    std::vector<Eigen::Vector3d> ref3Q, refZZ;
    if (has_zz) {
        lattice.load_spin_config(zz_seed);
        refZZ.resize(lattice.lattice_size);
        for (size_t i = 0; i < lattice.lattice_size; ++i) refZZ[i] = lattice.spins[i];
    }
    lattice.load_spin_config(seed_file);
    ref3Q.resize(lattice.lattice_size);
    for (size_t i = 0; i < lattice.lattice_size; ++i) ref3Q[i] = lattice.spins[i];
    size_t n_in = 0;
    // ---- geom "pie3": a three-domain zigzag mosaic with a triple junction at the centre.
    // Domains k = 1, 2 are the ZZ reference rotated by 120k deg about the box centre:
    // sites are permuted by the in-plane rotation (with PBC on the L x L rhombus, a1 =
    // (1,0), a2 = (1/2, sqrt3/2)) and spins rotated by the same angle about the honeycomb
    // normal.  Because the global spin frame is not guaranteed to have z along the normal
    // (audit item F5), three spin-rotation axes are tried and the one that reproduces the
    // zigzag energy per site is used — the construction validates itself.  Motivation:
    // a triple junction carries all three M-point components and is a ready-made 3Q
    // nucleus; a single-domain region has none (lifetime candidate 4).
    if (has_zz && geom == "pie3") {
        const size_t Ns = lattice.lattice_size;
        const double L = double(lattice.dim1);
        Eigen::Matrix2d A; A << 1.0, 0.5, 0.0, std::sqrt(3.0) / 2.0;
        const Eigen::Matrix2d Ai = A.inverse();
        auto wrap = [&](Eigen::Vector2d r) { Eigen::Vector2d f = Ai * r; f(0) -= L * std::floor(f(0) / L); f(1) -= L * std::floor(f(1) / L); return Eigen::Vector2d(A * f); };
        auto mind = [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b) { Eigen::Vector2d f = Ai * (a - b); f(0) -= L * std::round(f(0) / L); f(1) -= L * std::round(f(1) / L); return (A * f).norm(); };
        std::vector<Eigen::Vector2d> P(Ns);
        for (size_t i = 0; i < Ns; ++i) P[i] = wrap(lattice.site_positions[i].head<2>());
        // C3 centre: the site nearest the box centroid (the rotation must be about a site)
        size_t icen = 0; { double bd = 1e9; const Eigen::Vector2d cc = A * Eigen::Vector2d(L / 2, L / 2);
            for (size_t i = 0; i < Ns; ++i) { const double d = mind(P[i], cc); if (d < bd) { bd = d; icen = i; } } }
        const Eigen::Vector2d c2 = P[icen];
        const double th = 2.0 * M_PI / 3.0;
        Eigen::Matrix2d R2; R2 << std::cos(th), -std::sin(th), std::sin(th), std::cos(th);
        std::vector<size_t> inv(Ns); double maxd = 0.0;
        for (size_t i = 0; i < Ns; ++i) {               // site i maps to j under +120 deg: inv[j] = i
            const Eigen::Vector2d q = wrap(c2 + R2 * (P[i] - c2));
            size_t bj = 0; double bd = 1e9;
            for (size_t j = 0; j < Ns; ++j) { const double d = mind(P[j], q); if (d < bd) { bd = d; bj = j; } }
            inv[bj] = i; if (bd > maxd) maxd = bd;
        }
        cerr << "  pie3: site map max mismatch " << maxd << " a (must be ~0)\n";
        // reference zigzag energy
        for (size_t i = 0; i < Ns; ++i) lattice.spins[i] = refZZ[i];
        const double Ezz = lattice.total_energy() / Ns;
        // candidate spin-rotation axes; pick the one reproducing Ezz for the rotated domain
        // Both senses of rotation about each axis: with a transposed spin frame (audit F5)
        // the spin rotation that accompanies a +120 deg site rotation is -120 deg.
        // The spin rotation accompanying the +120 deg site rotation is not known a priori
        // (frame conventions, audit F5): search axis (Fibonacci sphere) x angle {+-120 deg},
        // then refine (axis polar/azimuth, angle) by coordinate descent until E_dom1 = E_ZZ.
        std::vector<std::vector<Eigen::Vector3d>> dom(3, std::vector<Eigen::Vector3d>(Ns));
        dom[0] = refZZ;
        auto Edom = [&](double pol, double az, double ang) {
            const Eigen::Vector3d n(std::sin(pol) * std::cos(az), std::sin(pol) * std::sin(az), std::cos(pol));
            const Eigen::AngleAxisd Rs(ang, n);
            for (size_t i = 0; i < Ns; ++i) lattice.spins[i] = Rs * refZZ[inv[i]];
            return std::abs(lattice.total_energy() / Ns - Ezz);
        };
        double bp = 0, ba = 0, bg = th, bestErr = 1e9;
        const int NF = 1500;
        for (int k = 0; k < NF; ++k) {
            const double z = 1.0 - 2.0 * (k + 0.5) / NF, pol = std::acos(z), az = M_PI * (3.0 - std::sqrt(5.0)) * k;
            for (double ang : { th, -th }) {
                const double e = Edom(pol, az, ang);
                if (e < bestErr) { bestErr = e; bp = pol; ba = az; bg = ang; }
            }
        }
        cerr << "  pie3: coarse search best |dE| = " << bestErr << " at axis (pol, az) = (" << bp << ", " << ba
             << ") angle " << bg << "\n";
        double h = 0.05;
        for (int it = 0; it < 60 && bestErr > 1e-10; ++it) {
            bool moved = false;
            for (int d = 0; d < 3; ++d) for (double s : { -h, h }) {
                double p = bp, a = ba, g = bg; (d == 0 ? p : d == 1 ? a : g) += s;
                const double e = Edom(p, a, g);
                if (e < bestErr) { bestErr = e; bp = p; ba = a; bg = g; moved = true; }
            }
            if (!moved) h *= 0.5;
        }
        const Eigen::Vector3d nbest(std::sin(bp) * std::cos(ba), std::sin(bp) * std::sin(ba), std::cos(bp));
        cerr << "  pie3: refined axis n = (" << nbest.transpose() << "), angle = " << bg * 180.0 / M_PI
             << " deg, |dE| = " << bestErr << " meV/site\n";
        if (bestErr > 1e-4) { cerr << "  pie3: CONSTRUCTION INVALID (no spin rotation reproduces E_ZZ) - aborting\n"; return 3; }
        const int best = 0;
        const Eigen::AngleAxisd Rs(bg, nbest);
        for (size_t i = 0; i < Ns; ++i) { dom[1][i] = Rs * refZZ[inv[i]]; }
        for (size_t i = 0; i < Ns; ++i) { dom[2][i] = Rs * dom[1][inv[i]]; }
        size_t nsec[3] = { 0, 0, 0 };
        for (size_t i = 0; i < Ns; ++i) {
            Eigen::Vector2d fd = Ai * (P[i] - c2);     // minimum-image displacement from the C3 centre
            fd(0) -= L * std::round(fd(0) / L); fd(1) -= L * std::round(fd(1) / L);
            const Eigen::Vector2d d = A * fd;
            double ang = std::atan2(d(1), d(0)); if (ang < 0) ang += 2.0 * M_PI;
            const int k = std::min(2, int(ang / th));
            lattice.spins[i] = dom[k][i]; ++n_in; ++nsec[k];
        }
        cerr << "  pie3: axis " << best << " (|dE| = " << bestErr << " meV/site); three-domain mosaic set, E_site = "
             << lattice.total_energy() / Ns << "; sector sizes " << nsec[0] << " / " << nsec[1] << " / " << nsec[2] << "\n";
    }
    if ((has_zz || rev_geom) && droplet_R > 0.0 && geom != "pie3") {
        const Eigen::Vector3d c0 = lattice.site_positions[lattice.lattice_size / 2];
        // The two seeds were annealed independently, so their global spin frames and the
        // PHASE of their shared M-point component are uncorrelated.  Splicing them as they
        // come makes the worst possible interface: measured at t = 0 the zigzag half's M3
        // amplitude arrives ANTI-phase with the 3Q half's and the two cancel (S_M3 = 165
        // instead of 2410), leaving the box 0.15 meV/site above its relaxed energy — 40x
        // the free-energy difference under study, which is why the 3Q half was destroyed
        // in every earlier band run.  With splice_opt on, pick the global spin rotation
        // (and inversion, which flips the M-point phase by pi) of the inserted region that
        // MINIMISES the spliced energy, i.e. matches the two orders across the interface.
        if (splice_opt && !rev_geom) {
            std::vector<Eigen::Vector3d>& ins = refZZ;
            std::vector<Eigen::Vector3d> base = ins;
            auto splice_E = [&](double pol, double az, double ang, double sgn) {
                const Eigen::Vector3d n(std::sin(pol) * std::cos(az), std::sin(pol) * std::sin(az), std::cos(pol));
                const Eigen::AngleAxisd R(ang, n);
                for (size_t i = 0; i < lattice.lattice_size; ++i)
                    lattice.spins[i] = inside(lattice.site_positions[i], c0)
                                     ? Eigen::Vector3d(sgn * (R * base[i])) : ref3Q[i];
                return lattice.total_energy() / double(lattice.lattice_size);
            };
            const double E_id = splice_E(0, 0, 0, +1.0);
            double bp = 0, ba = 0, bg = 0, bs = 1.0, bE = E_id;
            const int NF = 400;
            for (int k = 0; k < NF; ++k) {
                const double z = 1.0 - 2.0 * (k + 0.5) / NF, pol = std::acos(z),
                             az = M_PI * (3.0 - std::sqrt(5.0)) * k;
                for (int ia = 0; ia < 6; ++ia) for (double sg : { 1.0, -1.0 }) {
                    const double ang = ia * M_PI / 3.0;
                    const double e = splice_E(pol, az, ang, sg);
                    if (e < bE) { bE = e; bp = pol; ba = az; bg = ang; bs = sg; }
                }
            }
            double h = 0.2;
            for (int it = 0; it < 80; ++it) {
                bool moved = false;
                for (int d = 0; d < 3; ++d) for (double s : { -h, h }) {
                    double p = bp, a = ba, g = bg; (d == 0 ? p : d == 1 ? a : g) += s;
                    const double e = splice_E(p, a, g, bs);
                    if (e < bE) { bE = e; bp = p; ba = a; bg = g; moved = true; }
                }
                if (!moved) h *= 0.6;
            }
            const Eigen::Vector3d nb(std::sin(bp) * std::cos(ba), std::sin(bp) * std::sin(ba), std::cos(bp));
            const Eigen::AngleAxisd Rb(bg, nb);
            for (size_t i = 0; i < lattice.lattice_size; ++i) ins[i] = bs * (Rb * base[i]);
            cerr << "  splice_opt: E/site as-given " << E_id << " -> matched " << bE
                 << " (gain " << (E_id - bE) * 1e3 << " ueV/site); axis (" << nb.transpose()
                 << ") angle " << bg * 180.0 / M_PI << " deg, sign " << bs << "\n";
        }
        for (size_t i = 0; i < lattice.lattice_size; ++i)
            if (inside(lattice.site_positions[i], c0)) {
                lattice.spins[i] = rev_geom ? Eigen::Vector3d(-ref3Q[i]) : refZZ[i]; ++n_in;
            }
        if (j7_step != 0.0) {
            const string pf = string("droplet_j7step.") + std::to_string(getpid());
            std::ofstream f(pf);
            size_t nh = 0;
            for (size_t h = 0; h < lattice.hexagons.size(); ++h) {
                Eigen::Vector3d hc = Eigen::Vector3d::Zero();
                for (int k = 0; k < 6; ++k) hc += lattice.site_positions[lattice.hexagons[h][k]];
                hc /= 6.0;
                if (inside(hc, c0)) { f << h << " " << j7_step << "\n"; ++nh; }
            }
            f.close();
            lattice.apply_plaquette_j7_disorder_from_file(pf);
            std::remove(pf.c_str());
            cerr << "  droplet J7 step " << j7_step << " on " << nh << " hexagons\n";
        }
        cerr << "  zigzag droplet: R=" << droplet_R << " -> " << n_in << " of "
             << lattice.lattice_size << " sites (" << 100.0 * n_in / lattice.lattice_size << " %)\n";
    }

    const double N   = double(lattice.lattice_size);
    const double w2  = ph.omega_E1 * ph.omega_E1;
    const double NRM = lattice.phonon_norm();   // the EOM divides dH/dQ by this

    // ---- start the doublet at its magnetostrictive equilibrium ---------------
    // omega^2 Q = -dH_ME/dQ / phonon_norm().  For a 3Q seed this is exactly 0 (C3);
    // for a ZZ seed it is the 1.0637e-2 spontaneous distortion of ESTABLISHED.md 3b.
    // Starting a ZZ run at Q = 0 instead injects that 16.5 ueV/site as a coherent E1
    // ring-down — the same trap sld_relax exists to avoid in the acoustic sector.
    // The coupling is linear in Q here, so this converges immediately; iterate anyway
    // so it stays correct if a quadratic channel is switched on.
    lattice.phonons.Q_x_E1 = 0.0; lattice.phonons.Q_y_E1 = 0.0;
    lattice.phonons.V_x_E1 = 0.0; lattice.phonons.V_y_E1 = 0.0;
    for (int k = 0; k < 200; ++k) {
        const double qx = -lattice.dH_dQx_E1() / (NRM * w2);
        const double qy = -lattice.dH_dQy_E1() / (NRM * w2);
        const double dq = std::hypot(qx - lattice.phonons.Q_x_E1, qy - lattice.phonons.Q_y_E1);
        lattice.phonons.Q_x_E1 = qx; lattice.phonons.Q_y_E1 = qy;
        if (dq < 1e-14) break;
    }

    // ---- observables --------------------------------------------------------
    const Eigen::Vector3d M1(M_PI, M_PI / std::sqrt(3.0), 0.0);          // 30 deg
    const Eigen::Vector3d M2(0.0, 2.0 * M_PI / std::sqrt(3.0), 0.0);     // 90 deg
    const Eigen::Vector3d M3(-M_PI, M_PI / std::sqrt(3.0), 0.0);         // 150 deg

    const double dt = config.md_timestep > 0.0 ? config.md_timestep : 0.005;
    const size_t save_every = std::max<size_t>(1, size_t(std::lround(dt_save / dt)));
    const double HBAR_PS = 0.658212;

    csv << "t_ps,E_site,Qx,Qy,Qabs,dHdQx_site,dHdQy_site,E_tilt_site,euler,"
           "S_M1,S_M2,S_M3,min_over_max,phi_x,phi_y,E_phonon_site";
    if (has_zz) csv << ",f_ZZ,m_3Q,m_ZZ";
    if (emit_kappa) csv << ",kappa";
    const bool region_kappa = emit_kappa && droplet_R > 0.0 && geom != "pie3";   // disc: 0 inside, 1 outside
    if (region_kappa) csv << ",kappa_band,kappa_seaA,kappa_seaB";
    // Reference-free band metrics.  f_ZZ/m_3Q/m_ZZ compare every spin with the two seed
    // configurations, so they stop meaning anything the moment the box leaves the seeds'
    // own domain: a 3Q state reached by a global spin rotation (or a translation of the
    // 3Q order) has m_3Q < 0 and f_ZZ = 1 while kappa correctly reads -0.708.  That is
    // what most of the 1 ns runs did.  The local scalar chirality does not care — it is
    // ~0.7 in any 3Q domain of either chirality and 0 in any collinear (zigzag) domain.
    // So classify each site by its NN-smoothed |kappa_i| and report
    //   f_col   fraction of sites that are collinear (|<kappa>_i| < KCUT)
    //   w_cols  number of a1 columns whose mean |<kappa>| is below KCUT — the band width
    //           in a1 cells, directly comparable to the old w = L1 * f_ZZ.
    const bool band_metric = emit_kappa;
    const double KCUT = 0.35;
    std::vector<int> col(lattice.lattice_size, 0);
    if (band_metric) {
        csv << ",f_col,w_cols";
        for (size_t i = 0; i < lattice.lattice_size; ++i) {
            double f1 = dfrac(lattice.site_positions[i])(0) + Lbox / 2.0;
            f1 -= Lbox * std::floor(f1 / Lbox);
            col[i] = std::min(int(Lbox) - 1, int(f1));
        }
    }
    // region of each site for the band geometries: 0 band, 1 sea at x > x_c + R, 2 sea at x < x_c - R
    std::vector<int> region(lattice.lattice_size, 0);
    if (region_kappa) {
        for (size_t i = 0; i < lattice.lattice_size; ++i) {
            const double dx = dfrac(lattice.site_positions[i])(0);
            region[i] = band_geom ? ((std::abs(dx) < droplet_R) ? 0 : (dx > 0 ? 1 : 2)) : (inside(lattice.site_positions[i], Eigen::Vector3d::Zero()) ? 0 : 1);
        }
    }
    csv << "\n";
    csv << scientific << setprecision(10);

    auto emit = [&](double t) {
        const double s1 = lattice.structure_factor(M1);
        const double s2 = lattice.structure_factor(M2);
        const double s3 = lattice.structure_factor(M3);
        const double smax = std::max({s1, s2, s3});
        const double smin = std::min({s1, s2, s3});

        // E-doublet projection of the three M-point intensities.  The M points sit at
        // 30/90/150 deg and the doublet lives at 2*theta, i.e. 60/180/300 deg — 120 deg
        // apart, so this is the standard E projection of a three-state clock variable.
        // Identically zero for 3Q (s1 = s2 = s3); |phi| = the domain amplitude for ZZ.
        const double c1 = std::cos(M_PI / 3.0),   sn1 = std::sin(M_PI / 3.0);
        const double c2 = std::cos(M_PI),         sn2 = std::sin(M_PI);
        const double c3 = std::cos(5.0 * M_PI / 3.0), sn3 = std::sin(5.0 * M_PI / 3.0);
        const double phix = (2.0 / 3.0) * (s1 * c1 + s2 * c2 + s3 * c3);
        const double phiy = (2.0 / 3.0) * (s1 * sn1 + s2 * sn2 + s3 * sn3);

        const double qx = lattice.phonons.Q_x_E1, qy = lattice.phonons.Q_y_E1;
        // One force evaluation, not two: dH_dQx_E1()/dH_dQy_E1() each rebuild the whole
        // raw force vector, and this runs every save_every = 10 steps in stage 2.
        const std::vector<double> F = lattice.lattice_forces_raw();
        const double gx = F[0] / N, gy = F[1] / N;
        const double tilt = lattice.spin_phonon_energy() / N;
        // Euler's theorem: a function homogeneous of degree 1 in Q obeys
        // Q.grad_Q H = H.  Ratio == 1 <=> H_sp-ph is purely linear in Q, which is the
        // premise of reading E_tilt_site as "the linear channel".  Deviations mean a
        // quadratic coupling is on and the reading must be revisited.
        // Below 1e-15 meV/site the tilt is roundoff (the 3Q seed gives ~1e-30) and the
        // ratio is 0/0 noise; the identity holds trivially there, so report 1.
        const double euler = (std::abs(tilt) > 1e-15) ? (qx * gx + qy * gy) / tilt : 1.0;

        csv << (t - t_therm) * HBAR_PS << "," << lattice.total_energy() / N << ","
            << qx << "," << qy << "," << lattice.E1_amplitude() << ","
            << gx << "," << gy << "," << tilt << "," << euler << ","
            << s1 << "," << s2 << "," << s3 << ","
            << (smax > 0.0 ? smin / smax : 0.0) << ","
            << phix << "," << phiy << ","
            << lattice.phonon_energy() / N;
        if (has_zz) {
            size_t nzz = 0; double m3 = 0.0, mz = 0.0;
            for (size_t i = 0; i < lattice.lattice_size; ++i) {
                const double o3 = lattice.spins[i].dot(ref3Q[i]), oz = lattice.spins[i].dot(refZZ[i]);
                m3 += o3; mz += oz; if (oz > o3) ++nzz;
            }
            csv << "," << double(nzz) / N << "," << m3 / N << "," << mz / N;
        }
        if (emit_kappa) {
            double kap = 0.0, kreg[3] = { 0, 0, 0 }; size_t nreg[3] = { 0, 0, 0 };
            static std::vector<double> ksite;
            ksite.assign(lattice.lattice_size, 0.0);
            for (size_t i = 0; i < lattice.lattice_size; ++i) {
                const auto& nb = lattice.nn_partners[i];
                const Eigen::Vector3d si = lattice.spins[i];   // fixed-size copies: cross() needs Vector3d
                double ki = 0.0;
                for (size_t a = 0; a < nb.size(); ++a)
                    for (size_t b = a + 1; b < nb.size(); ++b) {
                        const size_t ia = std::min(nb[a], nb[b]), ib = std::max(nb[a], nb[b]);
                        const Eigen::Vector3d sa = lattice.spins[ia], sb = lattice.spins[ib];
                        ki += si.dot(sa.cross(sb));
                    }
                ksite[i] = ki;
                kap += ki;
                if (region_kappa) { kreg[region[i]] += ki; ++nreg[region[i]]; }
            }
            csv << "," << kap / N;
            if (region_kappa)
                for (int r = 0; r < 3; ++r) csv << "," << (nreg[r] ? kreg[r] / double(nreg[r]) : 0.0);
            if (band_metric) {
                const int nc = int(Lbox);
                std::vector<double> csum(nc, 0.0); std::vector<size_t> ccnt(nc, 0);
                size_t n_col = 0;
                for (size_t i = 0; i < lattice.lattice_size; ++i) {
                    double kb = ksite[i]; size_t m = 1;
                    for (size_t j : lattice.nn_partners[i]) { kb += ksite[j]; ++m; }
                    kb = std::abs(kb / double(m));
                    if (kb < KCUT) ++n_col;
                    csum[col[i]] += kb; ++ccnt[col[i]];
                }
                int wcols = 0;
                for (int c = 0; c < nc; ++c) if (ccnt[c] && csum[c] / double(ccnt[c]) < KCUT) ++wcols;
                csv << "," << double(n_col) / N << "," << wcols;
            }
        }
        csv << "\n";
        csv.flush();
    };

    cerr << "N=" << lattice.lattice_size << " J7=" << j7 << " E0=" << E0
         << " pol=" << pol << " T=" << Temp << " (" << Temp / 0.086173 << " K)"
         << " thermostat=" << (quantum ? "BOSE (semi-quantum)" : "CLASSICAL white")
         << " block=" << lattice.langevin_block
         << " disorder=" << dis << "\n";
    cerr << "  seed state: |Q|_relaxed=" << lattice.E1_amplitude()
         << "  E_tilt=" << lattice.spin_phonon_energy() / N * 1e3 << " ueV/site"
         << "  E_phonon=" << lattice.phonon_energy() / N * 1e3 << " ueV/site\n";
    cerr << "  (ESTABLISHED 3b reference for a ZZ seed: |Q|=1.0637e-02,"
            " E_tilt=-34.1 ueV/site, E_phonon=+17.1 ueV/site; 3Q must give 0,0,0)\n";
    cerr << "  thermalise 0 -> " << t_therm << ", pulse centred " << dr.t_1
         << ", run to " << t_therm + t_end << " (code units)\n";
    if (quantum) {
        const double blk = lattice.langevin_block * dt;
        cerr << "  Bose block = " << blk << " units (" << blk * HBAR_PS << " ps),"
             << " d(hbar w) = " << 2.0 * M_PI / blk << " meV vs k_B T = " << Temp
             << " -- want the former well below the latter to resolve the Bose cutoff\n";
    }

    // ONE continuous call: the noise stream is never restarted, so its correlations
    // are preserved across the whole trajectory including the thermalisation.
    // integrate_langevin saves BEFORE stepping, so t = 0 is emitted by the observer.
    if (anneal_t > 0.0) {
        const double a_keep = lattice.alpha_gilbert, T_keep = lattice.langevin_temperature;
        lattice.alpha_gilbert        = anneal_alpha;
        lattice.langevin_temperature = (anneal_T > 0.0) ? anneal_T : 1e-4;
        const double e_before = lattice.total_energy() / N;
        lattice.integrate_langevin(0.0, anneal_t, dt, "", size_t(1) << 30, rseed ^ 0xA5A5A5A5ULL);
        lattice.alpha_gilbert        = a_keep;
        lattice.langevin_temperature = T_keep;
        cerr << "  anneal: " << anneal_t << " units at k_B T = " << lattice.langevin_temperature
             << " -> " << ((anneal_T > 0.0) ? anneal_T : 1e-4) << ", alpha " << a_keep << " -> " << anneal_alpha
             << ";  E/site " << e_before << " -> " << lattice.total_energy() / N << "\n";
    }
    lattice.integrate_langevin(0.0, t_therm + t_end, dt, "", save_every, rseed, emit);
    emit(t_therm + t_end);   // the loop's last save need not land on t_end
    return 0;
}
