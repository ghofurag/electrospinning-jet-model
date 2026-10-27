# -*- coding: utf-8 -*-
"""
esjet.py — Discrete viscoelastic bead model of an electrospinning jet
(Reneker et al., J. Appl. Phys. 87 (2000) 4531; Kowalewski et al., IUTAM 2009).

Corrected and vectorised version of the GEMS group's original program.
Changes relative to the original program (see CHANGELOG at the bottom):
  * Maxwell stress evolves with the segment strain rate (it was a constant G/mu).
  * Segment radius is updated from volume conservation (it was fixed at a0).
  * Surface tension acts through the local curvature of the jet axis (Reneker 2000).
  * Every new bead receives a small random lateral perturbation (random seed -> statistics).
  * Beads that reach the collector are deposited (removed) and the diameter of the
    deposited segments is recorded; the fibre diameter is the mean over deposited
    segments after the start-up transient.
  * Adaptive explicit time step (semi-implicit Euler) + exponential Maxwell update.
SI units throughout.
"""
import numpy as np
try:
    from numba import njit
except ImportError:                       # pure-NumPy fallback (slower)
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]

K_E = 8.9875517923e9                      # Coulomb constant, N m^2 C^-2


@njit(cache=True)
def forces(r, sig, e, a0, L0, alpha, EF, F):
    """Total force on every bead. Bead 0 = leading (oldest) bead, bead n-1 = newest.
    Segment k joins bead k and bead k+1; sig[k] is its tensile stress."""
    n = r.shape[0]
    ke2 = K_E * e * e
    for i in range(n):
        F[i, 0] = e * EF; F[i, 1] = 0.0; F[i, 2] = 0.0
    # Coulomb repulsion (all pairs)
    for i in range(n):
        for j in range(i + 1, n):
            dx = r[i, 0] - r[j, 0]; dy = r[i, 1] - r[j, 1]; dz = r[i, 2] - r[j, 2]
            d2 = dx * dx + dy * dy + dz * dz + 1e-30
            inv = ke2 / (d2 * np.sqrt(d2))
            F[i, 0] += dx * inv; F[i, 1] += dy * inv; F[i, 2] += dz * inv
            F[j, 0] -= dx * inv; F[j, 1] -= dy * inv; F[j, 2] -= dz * inv
    # viscoelastic tension along segments: T = pi a^2 sigma = pi a0^2 L0 sigma / l
    for k in range(n - 1):
        dx = r[k + 1, 0] - r[k, 0]; dy = r[k + 1, 1] - r[k, 1]; dz = r[k + 1, 2] - r[k, 2]
        l = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-30
        T = np.pi * a0 * a0 * L0 * sig[k] / l
        F[k, 0] += T * dx / l; F[k, 1] += T * dy / l; F[k, 2] += T * dz / l
        F[k + 1, 0] -= T * dx / l; F[k + 1, 1] -= T * dy / l; F[k + 1, 2] -= T * dz / l
    # surface tension through local curvature: F = alpha pi abar^2 kappa_vec
    for i in range(1, n - 1):
        ax = r[i, 0] - r[i - 1, 0]; ay = r[i, 1] - r[i - 1, 1]; az = r[i, 2] - r[i - 1, 2]
        bx = r[i + 1, 0] - r[i, 0]; by = r[i + 1, 1] - r[i, 1]; bz = r[i + 1, 2] - r[i, 2]
        la = np.sqrt(ax * ax + ay * ay + az * az) + 1e-30
        lb = np.sqrt(bx * bx + by * by + bz * bz) + 1e-30
        cx = 2.0 * (bx / lb - ax / la) / (la + lb)
        cy = 2.0 * (by / lb - ay / la) / (la + lb)
        cz = 2.0 * (bz / lb - az / la) / (la + lb)
        abar2 = 0.5 * (a0 * a0 * L0 / la + a0 * a0 * L0 / lb)
        f = alpha * np.pi * abar2
        F[i, 0] += f * cx; F[i, 1] += f * cy; F[i, 2] += f * cz


def simulate(V=15e3, h=0.12, Q=1.11e-10, mu=1.75e-3, G=3.5e3, alpha=0.041, rho=1022.0,
             a0=3e-6, L0=1e-4, rho_q=1.0, eps=None, seed=0, n_dep=40, n_skip=10,
             tmax=0.5, cfl=0.05, dt_max=1e-5, nmax=4000, record=False, rec_every=50,
             charge_scale_with_V=None):
    """Run one realisation. Returns dict with the deposited jet diameters etc.
    rho_q : volumetric charge density of the jet (C m^-3); jet current I = rho_q Q.
    charge_scale_with_V : if (V_ref) given, rho_q is scaled as rho_q*V/V_ref (I ~ V)."""
    rng = np.random.default_rng(seed)
    if eps is None:
        eps = 1e-3 * L0
    if charge_scale_with_V:
        rho_q = rho_q * V / charge_scale_with_V
    vol = np.pi * a0 * a0 * L0
    m = rho * vol
    e = rho_q * vol
    EF = V / h
    theta = mu / G
    v0 = Q / (np.pi * a0 * a0)
    cap = nmax
    r = np.zeros((cap, 3)); v = np.zeros((cap, 3)); sig = np.zeros(cap); F = np.zeros((cap, 3))
    n = 0
    t = 0.0
    dep_d, dep_t = [], []
    first_path, snaps = [], []
    lead_id = 0; next_id = 0; ids = np.zeros(cap, dtype=np.int64)
    steps = 0
    while t < tmax and len(dep_d) < n_dep:
        # bead insertion at the nozzle (origin)
        if n == 0 or np.sqrt((r[n - 1] ** 2).sum()) >= L0:
            if n >= cap:
                break
            ph = rng.uniform(0, 2 * np.pi)
            r[n] = (0.0, eps * np.cos(ph), eps * np.sin(ph))
            v[n] = (v0, 0.0, 0.0)
            if n > 0:
                sig[n - 1] = 0.0          # new segment, stress-free
            ids[n] = next_id; next_id += 1
            n += 1
        forces(r[:n], sig[:n], e, a0, L0, alpha, EF, F[:n])
        acc = F[:n] / m
        amax = np.sqrt((acc ** 2).sum(1)).max() + 1e-30
        if n > 1:
            lmin = np.sqrt(((r[1:n] - r[:n - 1]) ** 2).sum(1)).min()
        else:
            lmin = L0
        dt = min(dt_max, cfl * np.sqrt(max(lmin, 1e-3 * L0) / amax))
        # segment lengths before update
        if n > 1:
            l_old = np.sqrt(((r[1:n] - r[:n - 1]) ** 2).sum(1))
        v[:n] += acc * dt
        r[:n] += v[:n] * dt
        if n > 1:
            l_new = np.sqrt(((r[1:n] - r[:n - 1]) ** 2).sum(1))
            srate = (l_new - l_old) / (l_old * dt)
            ex = np.exp(-dt / theta)
            sig[:n - 1] = sig[:n - 1] * ex + G * srate * theta * (1.0 - ex)
        t += dt; steps += 1
        if record and ids[0] == 0:
            first_path.append(r[0].copy())
        # deposition of beads that reached the collector
        while n > 1 and r[0, 0] >= h:
            l0seg = np.sqrt(((r[1] - r[0]) ** 2).sum())
            dep_d.append(2 * a0 * np.sqrt(L0 / l0seg)); dep_t.append(t)
            if record and len(snaps) == 0:
                snaps.append(r[:n].copy())
            r[:n - 1] = r[1:n]; v[:n - 1] = v[1:n]; sig[:n - 2] = sig[1:n - 1]; ids[:n - 1] = ids[1:n]
            n -= 1
    dep_d = np.array(dep_d)
    use = dep_d[n_skip:] if len(dep_d) > n_skip else dep_d
    out = dict(d_jet=use.mean() if len(use) else np.nan, d_jet_sd=use.std() if len(use) else np.nan,
               n_dep=len(dep_d), t=t, steps=steps, n_beads=n, t_first=dep_t[0] if dep_t else np.nan,
               current=rho_q * Q, v0=v0, theta=theta)
    if record:
        out['first_path'] = np.array(first_path)
        out['snapshot'] = snaps[0] if snaps else r[:n].copy()
        out['final'] = r[:n].copy()
        seg = np.sqrt(((r[1:n] - r[:n - 1]) ** 2).sum(1))
        out['profile_s'] = np.concatenate([[0], np.cumsum(seg[::-1])])  # arc length from nozzle
        out['profile_x'] = r[:n, 0][::-1]
        out['profile_d'] = np.concatenate([[2 * a0], (2 * a0 * np.sqrt(L0 / seg))[::-1]])
        out['dep_all'] = dep_d
    return out

# CHANGELOG vs original program
# 1. alpha = 41e-1 N/m (4.1 N/m) corrected to 0.041 N/m.
# 2. nodes[i].sigma = G/viskositas (constant, units s^-1) replaced by the Maxwell law
#    dsigma/dt = G (dl/dt)/l - sigma/theta, theta = mu/G (exponential integrator).
# 3. Tension uses the current segment radius a = a0 sqrt(L0/l) (radius was never updated).
# 4. Surface-tension force along the segment replaced by the curvature form of Reneker (2000).
# 5. Random lateral perturbation applied to every new bead (only the first bead before).
# 6. Diameter = mean of deposited segments after n_skip start-up beads (was the 10 leading
#    segments of a single snapshot). Fibre (dry) diameter: d_f = d_jet * sqrt(phi_p).
# 7. O(N^2) Python loops replaced by a compiled kernel; adaptive time step.
