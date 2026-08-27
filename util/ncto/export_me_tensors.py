#!/usr/bin/env python3
"""
Export the D3-covariant nearest-neighbour magnetoelastic tensors of the NCTO honeycomb
layer to a C++ header (ClassicalSpin_Cpp/include/classical_spin/lattice/ncto_me_tensors.h).

Frame: Kitaev (cubic) spin frame; n = (1,1,1)/sqrt3 is the layer normal; e_x is the
A->B direction of the x bond; e_y = n x e_x.  Bond gamma's directed vector d_gamma is
e_x rotated by 0, 120, 240 deg about n.

Tables (each entry a 3x3 matrix M with bond energy S_i^T M S_j for the A->B orientation):
  E doublets k = 0..8 : T1[k][gamma], T2[k][gamma]  ->  dM_gamma = Q1*T1 + Q2*T2 is D3-invariant
      k=0..3 : J, K, Gamma, Gamma' channel structures x bond projection (cos2th, -sin2th)
               -> exactly the code's lambda_X1 convention, NOT renormalised
      k=4    : (S_i.e_x)(S_j.n) + h.c., uniform                       [D6: E1, in/out-of-plane mixing]
      k=5    : (S_i.e_x)(S_j.e_x) - (S_i.e_y)(S_j.e_y), uniform       [D6: E2, global-frame nematic]
      k=6    : (S_i x S_j).e_x, uniform                               [D6: E1, in-plane DM]
      k=7    : (S_i x S_j).n  cos2th_gamma                            [out-of-plane DM nematic]
      k=8    : (S_i x S_j).e_x cos2th_gamma                           [in-plane DM nematic]
      (k=4..8 projected onto E by the projector of the same rep as (Q1,Q2), then normalised so
       that the x-bond partner-1 block has unit Frobenius norm)
  A1 k = 0..4 : dJ/dJ, dJ/dK, dJ/dGamma, dJ/dGamma', (S_i x S_j).d_gamma
  A2 k = 0..3 : (S_i x S_j).n uniform ; (S_i x S_j).(n x d_gamma) ; two symmetric completions
"""
import numpy as np
np.set_printoptions(precision=6, suppress=True, linewidth=150)
S3 = np.sqrt(3.0)
n_hat = np.array([1, 1, 1]) / S3


def rot(axis, ang):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


labels = ['x', 'y', 'z']
d = {'z': np.array([1, -1, 0]) / np.sqrt(2)}
d['x'] = rot(n_hat, 2 * np.pi / 3) @ d['z']
d['y'] = rot(n_hat, 4 * np.pi / 3) @ d['z']
ex = d['x'] / np.linalg.norm(d['x']); ey = np.cross(n_hat, ex)
th = {g: np.arctan2(d[g] @ ey, d[g] @ ex) for g in labels}
for g in labels:   # sanity: bonds at 0, 120, 240 deg in this frame
    assert abs(((np.degrees(th[g]) - 120 * labels.index(g)) + 180) % 360 - 180) < 1e-9, th


def bond_perm(R):
    perm = {}
    for g in labels:
        v = R @ d[g]
        for h in labels:
            if np.allclose(v, d[h], atol=1e-9): perm[g] = (h, +1)
            elif np.allclose(v, -d[h], atol=1e-9): perm[g] = (h, -1)
    assert len(perm) == 3
    return perm


def element(axis, ang):
    R = rot(axis, ang)
    D = np.array([[ex @ R @ ex, ex @ R @ ey], [ey @ R @ ex, ey @ R @ ey]])
    return dict(R=R, perm=bond_perm(R), D=D)


G = [element(n_hat, 0), element(n_hat, 2 * np.pi / 3), element(n_hat, 4 * np.pi / 3),
     element(d['x'], np.pi), element(d['y'], np.pi), element(d['z'], np.pi)]


def rep27(el):
    M = np.zeros((27, 27))
    for gi, g in enumerate(labels):
        h, s = el['perm'][g]; hi = labels.index(h)
        for a in range(3):
            for b in range(3):
                col = gi * 9 + a * 3 + b
                for ap in range(3):
                    for bp in range(3):
                        row = hi * 9 + (ap * 3 + bp if s > 0 else bp * 3 + ap)
                        M[row, col] += el['R'][ap, a] * el['R'][bp, b]
    return M


Gam = [rep27(el) for el in G]
chi_C2 = [1, 1, 1, -1, -1, -1]
P_A1 = sum(Gam) / 6
P_A2 = sum(c * g for c, g in zip(chi_C2, Gam)) / 6
PE = {(m, n): (2 / 6) * sum(el['D'][m, n] * g for el, g in zip(G, Gam)) for m in range(2) for n in range(2)}


def blocks(v):
    return [v[9 * i:9 * i + 9].reshape(3, 3) for i in range(3)]


def uniform(M):
    return np.concatenate([M.reshape(9)] * 3)


def per_bond(fn):
    return np.concatenate([fn(g).reshape(9) for g in labels])


def channel(name, g):
    gam = labels.index(g); al = 1 if gam == 0 else 0; be = 3 - gam - al
    M = np.zeros((3, 3))
    if name == 'J': M = np.eye(3)
    if name == 'K': M[gam, gam] = 1
    if name == 'G': M[al, be] = M[be, al] = 1
    if name == "G'": M[gam, al] = M[al, gam] = M[gam, be] = M[be, gam] = 1
    return M


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def sym(a, b): return np.outer(a, b) + np.outer(b, a)


def check_E(T1, T2):
    worst = 0.0
    rng = np.random.default_rng(0)
    for el, g in zip(G, Gam):
        for _ in range(4):
            u = rng.normal(size=2); ug = el['D'] @ u
            worst = max(worst, np.abs(g @ (u[0] * T1 + u[1] * T2) - (ug[0] * T1 + ug[1] * T2)).max())
    return worst


# ---- E doublets
E1 = []; E2 = []; Enames = []
# k = 0..3 analytic (code convention, no renormalisation)
for nm in ['J', 'K', 'G', "G'"]:
    T1 = per_bond(lambda g: np.cos(2 * th[g]) * channel(nm, g))
    T2 = per_bond(lambda g: -np.sin(2 * th[g]) * channel(nm, g))
    assert check_E(T1, T2) < 1e-12
    E1.append(T1); E2.append(T2); Enames.append(f"{nm}-channel, bond-projected: dX = c [Q1 cos2th - Q2 sin2th]")
# k = 4..8 CLOSED-FORM doublets (partner-1 multiplies Q1, partner-2 multiplies Q2):
#   k=4 : (S_i.(n x Q))(S_j.n) + h.c.        T1 = sym(e_y, n),        T2 = -sym(e_x, n)
#   k=5 : S_i^T N(Q) S_j, N(Q) = Q1 (e_x e_x - e_y e_y) - Q2 (e_x e_y + e_y e_x)   (global-frame nematic)
#   k=6 : (S_i x S_j).Q                       T1 = skew(e_x),          T2 = skew(e_y)
#   k=7 : sum_g f_g(Q) (S_i x S_j).n           T1 = cos2th skew(n),     T2 = -sin2th skew(n)
#   k=8 : sum_g f_g(Q) (S_i x S_j).d_g         T1 = cos2th skew(d_g),   T2 = -sin2th skew(d_g)
#   with f_g(Q) = Q1 cos2th_g - Q2 sin2th_g = Q . d_g (the bond projection, same as k=0..3).
closed = [("(S_i.(n x Q))(S_j.n)+h.c. [in-/out-of-plane mixing, D6:E1]",
           uniform(sym(ey, n_hat)), -uniform(sym(ex, n_hat))),
          ("S_i^T N(Q) S_j, N = Q1(e_x e_x - e_y e_y) - Q2(e_x e_y + e_y e_x) [global-frame nematic, D6:E2]",
           uniform(np.outer(ex, ex) - np.outer(ey, ey)), -uniform(sym(ex, ey))),
          ("(S_i x S_j).Q [in-plane DM along the displacement, D6:E1]",
           uniform(skew(ex)), uniform(skew(ey))),
          ("sum_g h_g(Q) (S_i x S_j).n [bond-projected out-of-plane DM; (S x S).n is A2 so the partner "
           "functions are the A2-twisted pair h_g = Q1 sin2th_g + Q2 cos2th_g (selected numerically)]",
           None, None),
          ("sum_g f_g(Q) (S_i x S_j).d_g [bond-projected DM parallel to bond]",
           per_bond(lambda g: np.cos(2 * th[g]) * skew(d[g])), per_bond(lambda g: -np.sin(2 * th[g]) * skew(d[g])))]
# For the A2-type operator (S_i x S_j).n the bond weights pairing with the polar (Q1,Q2) are
# NOT (cos2th, -sin2th): try the four sign/swap variants and keep the one that is invariant.
cands7 = [("h = Q1 sin2th + Q2 cos2th", per_bond(lambda g: np.sin(2 * th[g]) * skew(n_hat)), per_bond(lambda g: np.cos(2 * th[g]) * skew(n_hat))),
          ("h = Q1 sin2th - Q2 cos2th", per_bond(lambda g: np.sin(2 * th[g]) * skew(n_hat)), per_bond(lambda g: -np.cos(2 * th[g]) * skew(n_hat))),
          ("h = Q1 cos2th + Q2 sin2th", per_bond(lambda g: np.cos(2 * th[g]) * skew(n_hat)), per_bond(lambda g: np.sin(2 * th[g]) * skew(n_hat))),
          ("h = Q1 cos2th - Q2 sin2th", per_bond(lambda g: np.cos(2 * th[g]) * skew(n_hat)), per_bond(lambda g: -np.sin(2 * th[g]) * skew(n_hat)))]
for cnm, T1, T2 in cands7:
    e = check_E(T1, T2)
    print(f"  k=7 candidate {cnm}: invariance residual {e:.1e}")
    if e < 1e-12:
        closed[3] = (closed[3][0].replace("(selected numerically)", cnm), T1, T2)
        break
assert closed[3][1] is not None, "no invariant pairing found for (S x S).n"
for nm, T1, T2 in closed:
    err = check_E(T1, T2)
    assert err < 1e-12, (nm, err)
    # each partner must be a pure partner function of the E space (projectors idempotent on it)
    assert np.abs(PE[(0, 0)] @ T1 - T1).max() < 1e-12, nm
    assert np.abs(PE[(1, 1)] @ T2 - T2).max() < 1e-12, nm
    E1.append(T1); E2.append(T2); Enames.append(nm)
# independence
rank = np.linalg.matrix_rank(np.array(E1), tol=1e-9)
assert rank == 9, rank

# ---- A1
A1 = []; A1names = []
for nm in ['J', 'K', 'G', "G'"]:
    v = per_bond(lambda g: channel(nm, g)); assert np.abs(P_A1 @ v - v).max() < 1e-12
    A1.append(v); A1names.append(f"dJ_gamma/d{nm}")
v = per_bond(lambda g: skew(d[g])); assert np.abs(P_A1 @ v - v).max() < 1e-12
A1.append(v); A1names.append("(S_i x S_j).d_gamma  (DM parallel to bond)")
assert np.linalg.matrix_rank(np.array(A1), tol=1e-9) == 5

# ---- A2 (closed forms): two antisymmetric (DM) and two symmetric bond-odd operators
A2 = []; A2names = []
for nm, v in [("(S_i x S_j).n uniform (out-of-plane DM)", uniform(skew(n_hat))),
              ("(S_i x S_j).(n x d_gamma) (in-plane DM perpendicular to the bond)", per_bond(lambda g: skew(np.cross(n_hat, d[g])))),
              ("(S_i.d_gamma)(S_j.n)+h.c. (symmetric, bond-direction/out-of-plane mixing)", per_bond(lambda g: sym(d[g], n_hat))),
              ("(S_i.(n x d_gamma))(S_j.d_gamma)+h.c. (symmetric, in-plane bond-odd)", per_bond(lambda g: sym(np.cross(n_hat, d[g]), d[g])))]:
    assert np.abs(P_A2 @ v - v).max() < 1e-12, nm
    A2.append(v); A2names.append(nm)
assert np.linalg.matrix_rank(np.array(A2), tol=1e-9) == 4, A2names

# ---- write header
def cxx_array(name, mats, comment_names):
    out = [f"// {name}: [k][gamma][a][b]"]
    out.append(f"inline constexpr double {name}[{len(mats)}][3][3][3] = {{")
    for k, v in enumerate(mats):
        out.append(f"  // k={k}: {comment_names[k]}")
        out.append("  {")
        for gi, B in enumerate(blocks(v)):
            rows = ", ".join("{" + ", ".join(f"{B[a, b]: .16e}" for b in range(3)) + "}" for a in range(3))
            out.append(f"    {{{rows}}}{',' if gi < 2 else ''}  // bond {labels[gi]}")
        out.append("  }" + ("," if k < len(mats) - 1 else ""))
    out.append("};")
    return "\n".join(out)

hdr = ["#pragma once",
       "// Generated by ncto_phonon/audit/export_me_tensors.py -- DO NOT EDIT BY HAND.",
       "//",
       "// D3-covariant nearest-neighbour magnetoelastic tensors of the Na2Co2TeO6 honeycomb layer,",
       "// Kitaev (cubic) spin frame, bond energy S_i^T M S_j for the A->B orientation of the bond.",
       "// In-plane doublet frame: e_x = A->B direction of the x bond, e_y = n x e_x, n = (1,1,1)/sqrt3.",
       "// E doublets: dM_gamma(Q) = Q1*NCTO_ME_E1[k][gamma] + Q2*NCTO_ME_E2[k][gamma] is invariant under D3.",
       "// k=0..3: lambda_X1 convention f_g(Q) = Q1 cos2th_g - Q2 sin2th_g = Q.d_g times the J,K,Gamma,Gamma' structure;",
       "// k=4: (S_i.(n x Q))(S_j.n)+h.c.;  k=5: S_i^T [Q1(e_x e_x - e_y e_y) - Q2(e_x e_y + e_y e_x)] S_j;",
       "// k=6: (S_i x S_j).Q;  k=7: sum_g f_g(Q) (S_i x S_j).n;  k=8: sum_g f_g(Q) (S_i x S_j).d_g.",
       "// Kitaev-frame vectors: n = (1,1,1)/sqrt3, e_x = d_x = (0,1,-1)/sqrt2, e_y = (-2,1,1)/sqrt6,",
       "// d_y = (-1,0,1)/sqrt2, d_z = (1,-1,0)/sqrt2.",
       "// A1 (scalar coordinates: A1 Raman modes, eps_xx+eps_yy, eps_zz, |Q|^2) and A2 (u_z of c-polarised modes).",
       "namespace classical_spin::ncto_me {",
       "inline constexpr int N_E = 9, N_A1 = 5, N_A2 = 4;",
       cxx_array("NCTO_ME_E1", E1, Enames),
       cxx_array("NCTO_ME_E2", E2, Enames),
       cxx_array("NCTO_ME_A1", A1, A1names),
       cxx_array("NCTO_ME_A2", A2, A2names),
       "inline const char* const NCTO_ME_E_NAMES[N_E] = {" + ", ".join(f'"{n}"' for n in Enames) + "};",
       "inline const char* const NCTO_ME_A1_NAMES[N_A1] = {" + ", ".join(f'"{n}"' for n in A1names) + "};",
       "inline const char* const NCTO_ME_A2_NAMES[N_A2] = {" + ", ".join(f'"{n}"' for n in A2names) + "};",
       "}  // namespace classical_spin::ncto_me", ""]
path = "/lustre09/project/6003507/zhouzb79/ClassicalSpin_Cpp/include/classical_spin/lattice/ncto_me_tensors.h"
open(path, "w").write("\n".join(hdr))
print("wrote", path)
for k, nm in enumerate(Enames): print(f"E {k}: {nm}")
for k, nm in enumerate(A1names): print(f"A1 {k}: {nm}")
for k, nm in enumerate(A2names): print(f"A2 {k}: {nm}")
print("E partner-1 x-bond blocks:")
for k in range(9): print(k, Enames[k], "\n partner1 x-block\n", blocks(E1[k])[0], "\n partner2 x-block\n", blocks(E2[k])[0])
print("vectors: n=", n_hat, " e_x=", ex, " e_y=", ey, " d_y=", d["y"], " d_z=", d["z"])
print("DONE")
