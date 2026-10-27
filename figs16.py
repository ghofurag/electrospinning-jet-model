import json, pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
OUT = '/home/claude/v3/figs/'
import os; os.makedirs(OUT, exist_ok=True)
PHI = 0.0673; SQ = np.sqrt(PHI)
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'axes.linewidth': 0.8,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'savefig.dpi': 300})
C1, C2, C3 = '#1f3b73', '#c0392b', '#7f8c8d'
def lab(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight='bold', fontsize=10, va='bottom')

rec = pickle.load(open('/home/claude/v3/runs/rec.pkl', 'rb'))
S = None

# ---------------- Figure 1: flowchart
fig, ax = plt.subplots(figsize=(3.4, 5.6)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 20)
steps = ['Start: read inputs (V, h, Q, μ, G, α, ρ, ρ$_q$, a$_0$, L$_0$)',
         'Insert bead at nozzle if last bead\nis ≥ L$_0$ away (random lateral offset ε)',
         'Forces: Coulomb + field + Maxwell\ntension + surface tension (curvature)',
         'Adaptive Δt; update v, r (semi-implicit\nEuler) and σ (exponential Maxwell step)',
         'Bead reached collector (x ≥ h)?\nRecord segment diameter, remove bead',
         'Deposited beads ≥ N$_{dep}$ ?',
         'Output: mean d$_j$, d$_f$ = d$_j$√φ$_p$,\njet shape and thinning profile']
ys = [18.6, 15.9, 13.2, 10.5, 7.8, 5.4, 2.6]
for i, (s, y) in enumerate(zip(steps, ys)):
    w = 9.0; hgt = 1.9 if '\n' in s else 1.3
    st = 'round,pad=0.12' if i not in (5,) else 'round,pad=0.12'
    ax.add_patch(FancyBboxPatch((5 - w / 2, y - hgt / 2), w, hgt, boxstyle=st,
                                fc='#eef2f8' if i not in (0, 6) else '#dfe7f3', ec=C1, lw=0.8))
    ax.text(5, y, s, ha='center', va='center', fontsize=7.3)
    if i < len(ys) - 1:
        ax.annotate('', xy=(5, ys[i + 1] + (0.95 if '\n' in steps[i + 1] else 0.65)),
                    xytext=(5, y - hgt / 2), arrowprops=dict(arrowstyle='-|>', color=C1, lw=0.8))
ax.plot([0.5, 0.15, 0.15], [5.4, 5.4, 15.9], color=C2, lw=0.8)
ax.annotate('', xy=(0.5, 15.9), xytext=(0.15, 15.9), arrowprops=dict(arrowstyle='-|>', color=C2, lw=0.8))
ax.text(-0.35, 10.6, 'no', color=C2, fontsize=7, rotation=90, va='center')
ax.text(5.25, 4.25, 'yes', color=C1, fontsize=7)
fig.savefig(OUT + 'fig1_flowchart.png', bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 2: baseline jet + thinning profile
from matplotlib.ticker import ScalarFormatter, FuncFormatter
b = rec['base']; f = b['final'] * 100
fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.7), gridspec_kw=dict(width_ratios=[0.8, 1, 1.7]))
axs[0].plot(f[:, 1], f[:, 0], color=C1, lw=0.7); axs[0].set_ylim(12.3, 0); axs[0].set_xlim(-2, 2)
axs[0].set_xlabel('y (cm)'); axs[0].set_ylabel('x (cm)')
axs[1].plot(f[:, 1], f[:, 2], color=C1, lw=0.7); axs[1].set_aspect('equal'); axs[1].set_xlim(-2, 2); axs[1].set_ylim(-2, 2)
axs[1].set_xlabel('y (cm)'); axs[1].set_ylabel('z (cm)')
s_ = b['profile_s'] * 100; dj = b['profile_d'] * 1e6
ax2 = axs[2]; ax2.semilogy(s_, dj, color=C1, lw=1)
fmt = FuncFormatter(lambda v, p: ('%g' % v)); ax2.yaxis.set_major_formatter(fmt); ax2.yaxis.set_minor_formatter(FuncFormatter(lambda v,p: ''))
ax2.set_yticks([0.3, 0.5, 1, 2, 5]); ax2.set_ylim(0.25, 8)
ax2.set_xlabel('Arc length from nozzle (cm)'); ax2.set_ylabel('Wet-jet diameter $d_j$ (\u00b5m)')
ax2r = ax2.twinx(); ax2r.set_yscale('log'); ax2r.set_ylim(0.25 * SQ * 1e3, 8 * SQ * 1e3)
ax2r.set_yticks([100, 200, 500, 1000]); ax2r.yaxis.set_major_formatter(fmt); ax2r.yaxis.set_minor_formatter(FuncFormatter(lambda v,p: ''))
ax2r.set_ylabel('Dry-fiber equivalent $d_f$ (nm)')
ax2.axhline(6, color=C3, ls=':', lw=0.8); ax2.text(s_.max() * 0.5, 6.4, 'initial jet, 6 \u00b5m', fontsize=7, color='#555')
for i, ax in enumerate(axs): lab(ax, '(%s)' % 'abc'[i], -0.3 if i < 2 else -0.15, 1.03)
fig.tight_layout(); fig.savefig(OUT + 'fig2_baseline.png', bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 3: jet shapes (side view)
fig, axs = plt.subplots(1, 5, figsize=(6.8, 3.0), sharey=False)
cases = [('V10', '10 kV, 12 cm'), ('base', '16 kV, 12 cm'), ('V20', '20 kV, 12 cm'), ('h7', '16 kV, 7 cm'), ('h20', '16 kV, 20 cm')]
for i, (ax, (k, t)) in enumerate(zip(axs, cases)):
    f = rec[k]['final'] * 100
    ax.plot(f[:, 1], f[:, 0], color=C1, lw=0.7)
    hh = f[:, 0].max()
    ax.set_ylim(max(hh, 7) * 1.02 if k != 'h20' else 20.4, 0)
    ax.set_xlim(-4, 4); ax.set_title(t, fontsize=8)
    ax.set_xlabel('y (cm)')
    if i == 0: ax.set_ylabel('Distance from nozzle, x (cm)')
    lab(ax, '(%s)' % 'abcde'[i], -0.25, 1.09)
fig.tight_layout(); fig.savefig(OUT + 'fig3_shapes.png', bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 4: parametric sensitivity
R = json.load(open('/home/claude/v3/runs/prod.json'))
def agg(g):
    d = {}
    for r in R:
        if r['grp'] == g: d.setdefault(r['val'], []).append(r['d_jet'] * SQ * 1e9)
    x = np.array(sorted(d)); m = np.array([np.mean(d[v]) for v in x]); s = np.array([np.std(d[v]) for v in x])
    return x, m, s
fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.5))
x, m, s = agg('V'); axs[0].errorbar(x / 1e3, m, s, fmt='o-', color=C1, ms=4, capsize=2, lw=1, label='fixed $\\rho_q$')
x, m, s = agg('V_IV'); axs[0].errorbar(x / 1e3, m, s, fmt='s--', color=C2, ms=4, capsize=2, lw=1, mfc='white', label='$\\rho_q \\propto V$')
axs[0].set_xlabel('Applied voltage V (kV)'); axs[0].set_ylabel('Fiber diameter $d_f$ (nm)'); axs[0].legend(frameon=False, fontsize=7)
x, m, s = agg('mu'); axs[1].errorbar(x * 1e3, m, s, fmt='o-', color=C1, ms=4, capsize=2, lw=1)
axs[1].set_xscale('log'); axs[1].set_xlabel('Viscosity μ (mPa s)')
axs[1].axvspan(1.5, 60, color=C3, alpha=0.12, lw=0); axs[1].text(2, 300, 'viscous stress\nnegligible', fontsize=6.5, color='#555')
x, m, s = agg('h'); axs[2].errorbar(x * 100, m, s, fmt='o-', color=C1, ms=4, capsize=2, lw=1)
axs[2].set_xlabel('Nozzle–collector distance h (cm)')
for i, ax in enumerate(axs):
    ax.axhline(142, color=C3, ls=':', lw=0.8); lab(ax, '(%s)' % 'abc'[i], -0.2, 1.03)
axs[2].text(15, 150, 'SEM, 142 nm', fontsize=6.5, color='#555')
fig.tight_layout(); fig.savefig(OUT + 'fig4_sensitivity.png', bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 5: experiment vs simulation
fw = np.load('/home/claude/v2/sem/fwhm.npy'); sim = np.load('/home/claude/v3/runs/base_dep_nm.npy')
sem = Image.open('/home/claude/v2/sem/sem.jpg').convert('L')
opt = Image.open('/home/claude/work/media/media/image6.png').convert('RGB')
W, H = opt.size; opt = opt.crop((0, 0, int(W * 0.52), int(H * 0.89)))
fig = plt.figure(figsize=(6.8, 2.3))
ax = fig.add_axes([0.0, 0.05, 0.31, 0.9]); ax.imshow(sem, cmap='gray'); ax.axis('off'); lab(ax, '(a)', 0, 1.01)
ax = fig.add_axes([0.40, 0.2, 0.26, 0.72])
bins = np.arange(0, 420, 20)
ax.hist(fw, bins=bins, weights=np.full(len(fw), 100 / len(fw)), color=C1, alpha=0.75, label='SEM (n = %d)' % len(fw))
ax.hist(sim, bins=bins, weights=np.full(len(sim), 100 / len(sim)), histtype='step', color=C2, lw=1.4, label='Model (n = %d)' % len(sim))
ax.set_xlabel('Fiber diameter (nm)'); ax.set_ylabel('Frequency (%)'); ax.legend(frameon=False, fontsize=6.5)
lab(ax, '(b)', -0.3, 1.01)
ax = fig.add_axes([0.70, 0.05, 0.30, 0.9]); ax.imshow(opt); ax.axis('off'); lab(ax, '(c)', 0, 1.01)
fig.savefig(OUT + 'fig5_experiment.png', bbox_inches='tight'); plt.close(fig)
print('ok')
