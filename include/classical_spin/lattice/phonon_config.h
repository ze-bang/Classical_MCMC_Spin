#pragma once
/**
 * phonon_config.h — the single SpinConfig → PhononLattice (NCTO model) construction path.
 *
 * spin_solver, the parameter sweep and the ncto_* tools previously assembled the model
 * each in their own way (the sweep skipped the extra lattice modes, the disorder and
 * pinning files and used a different damping default), so one config could describe
 * different Hamiltonians depending on the entry point. Everything that defines the model
 * now goes through make_ncto_lattice().
 */

#include "classical_spin/core/spin_config.h"
#include "classical_spin/lattice/phonon_lattice.h"

#include <cstdint>

/**
 * Coupling, phonon, drive and time-dependence parameters from a config. Absent keys take
 * the struct defaults (the Na2Co2TeO6 operating point); `legacy_kitaev_frame = 1` selects
 * the pre-2026-10 spin frame. Pulse shapes: pump/probe frequency and width fall back to the
 * measured NCTO pulse unless the key is given explicitly. The amplitudes returned here are
 * the configured ones; resolve_ncto_drive() decides which pulses a simulation applies.
 */
void build_phonon_params(const SpinConfig& config,
                         SpinPhononCouplingParams& sp_params,
                         PhononParams& ph_params,
                         DriveParams& dr_params,
                         TimeDependentSpinPhononParams& td_sp_params);

/**
 * Complete lattice sector (audit 2026-08): extra zone-centre modes and cubic
 * anharmonic transfers.  Keys (i = 1..n_extra_modes):
 *   n_extra_modes
 *   mode<i>_irrep      0 = E polar/E1 (IR, drives with Zstar), 1 = E2-type (Raman / in-plane
 *                      strain), 2 = A1 (Raman / breathing strain), 3 = A2 (c-polarised)
 *   mode<i>_omega, _gamma, _quartic, _Zstar, _frozen, _Q1, _Q2   (initial or frozen values)
 *   first order : mode<i>_cE<k> (k=1..9), _aA1_<k> (1..5), _dA2_<k> (1..4),
 *                 _lamJ7, _lamJ2A, _lamJ2B, _lamJ3
 *   second order: mode<i>_bEsq<k> (1..9), _aA1sq_<k> (1..5), _lamJ7sq, _lamJ2Asq, _lamJ2Bsq, _lamJ3sq
 * The primary mode (i = 0) accepts the same keys on top of the legacy lambda_E1_* values
 * (e.g. mode0_cE5 .. mode0_cE9 for the five non-channel linear tensors, mode0_aA1sq_5 for
 * the DM-parallel term, mode0_lamJ2A for the J2 nematic).
 * Anharmonic transfers: n_anharmonic, anh<j>_target, anh<j>_lam, anh<j>_lamp, anh<j>_g
 * (mode indices, 0 = primary), energy −N g Q_target (Q_lam ⊗ Q_lamp).
 */
void build_lattice_modes(const SpinConfig& config, PhononLattice& lattice);

/**
 * THz pulses a simulation applies through drive_params:
 *   pump_probe          pump with pump_amplitude (default 1.0 — the protocol's pulse);
 *                       second pulse only if probe_amplitude is set explicitly;
 *   2dcs                none (the spectroscopy driver sets both pulses per run);
 *   everything else     (molecular_dynamics, annealing, ...) a pulse only if its
 *                       amplitude key is set explicitly — equilibrium dynamics is undriven.
 * Warns when a pulse centre lies within 4σ of md_time_start (abrupt switch-on).
 */
DriveParams resolve_ncto_drive(const SpinConfig& config, SimulationType simulation);

/**
 * Build the NCTO model from a config: unit cell, couplings (incl. the storage frame),
 * extra lattice modes, quenched disorder files, magnetoelastic schedule, Zeeman and
 * pinning fields, the local_field_h triple-q stabiliser (applied once), Gilbert damping
 * (default 0; Langevin runners substitute 0.05 when alpha_gilbert is absent), the
 * spin–lattice dynamics and Langevin settings, the local MC update and the initial
 * state (initial_spin_config, else ferromagnetic, else random — the priority of every trial
 * start — plus initial_eps_* phonons).
 * File problems throw.
 */
PhononLattice make_ncto_lattice(const SpinConfig& config);

/**
 * Seed of the Langevin noise stream of a trial: derived (splitmix64) from `langevin_seed`
 * if given, else from the master `seed`, and the trial index, so trials (and therefore
 * ranks) never share a noise realisation and every run is reproducible.
 */
uint64_t ncto_trial_seed(const SpinConfig& config, int trial);
