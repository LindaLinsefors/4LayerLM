"""Plot the t-freqtest47 inv_freq trajectory against the fitted seed-45/46 spectrum.

Fetch first (works mid-run on a partial trajectory):
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal volume get -f vpd-4layer \
      /sink-models/runs/t-freqtest47/inv_freq_trajectory.npz own-pretrain/hide/cache/

Writes own-pretrain/inv_freq_trajectory.png:
  panel 1 — spectrum omega_p vs plane p (log y): init (textbook 1e4), current/final,
            and the fitted seed-45/46 average spectrum (well-constrained planes 0-26
            solid, the rest dashed);
  panel 2 — trajectory of log10(omega_p / omega_p^init) vs step for a subset of planes;
  panel 3 — distance-to-fitted vs step: RMS over planes 0-26 of
            log(omega_p(t)) - log(omega_p^fitted), plus the same for RMS
            log-distance to INIT (does it move away from init toward fitted?).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

traj = np.load(HERE / "cache" / "inv_freq_trajectory.npz")
steps, freqs, init = traj["steps"], traj["inv_freq"].astype(np.float64), traj["init"].astype(np.float64)
fitted = np.exp(np.load(ROOT / "sink-models" / "hide" / "cache" / "fitted_freqs_avg.npz")["log_inv_freq"])
GOOD = 27  # well-constrained fitted planes 0..26 (rope_report.md)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

ax = axes[0]
p = np.arange(64)
ax.semilogy(p, init, "k--", lw=1, label="init (textbook 1e4)")
ax.semilogy(p[:GOOD], fitted[:GOOD], "-", color="#d62728", lw=1.5, label="fitted seed-45/46 (planes 0-26)")
ax.semilogy(p[GOOD:], fitted[GOOD:], ":", color="#d62728", lw=1, label="fitted (poorly constrained)")
ax.semilogy(p, np.abs(freqs[-1]), "-", color="#2a78d6", lw=1.5, label=f"trained @ step {steps[-1]:,}")
ax.set_xlabel("plane p"), ax.set_ylabel(r"$\omega_p$"), ax.legend(fontsize=8)
ax.set_title("RoPE spectrum")

ax = axes[1]
for pp in [0, 1, 2, 5, 10, 20, 30, 40, 50, 63]:
    ax.plot(steps, np.log10(np.abs(freqs[:, pp]) / init[pp]), lw=1, label=f"p={pp}")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("step"), ax.set_ylabel(r"$\log_{10}(\omega_p/\omega_p^{init})$")
ax.legend(fontsize=7, ncol=2), ax.set_title("per-plane drift")

ax = axes[2]
dl_fit = np.sqrt(np.mean((np.log(np.abs(freqs[:, :GOOD])) - np.log(fitted[:GOOD])) ** 2, axis=1))
dl_init = np.sqrt(np.mean((np.log(np.abs(freqs[:, :GOOD])) - np.log(init[:GOOD])) ** 2, axis=1))
ax.plot(steps, dl_fit, color="#d62728", label="RMS log-dist to fitted (p<27)")
ax.plot(steps, dl_init, color="k", label="RMS log-dist to init (p<27)")
ax.set_xlabel("step"), ax.set_ylabel("RMS log distance"), ax.legend(fontsize=8)
ax.set_title("toward fitted, away from init?")

fig.suptitle("t-freqtest47: trainable RoPE inverse frequencies", y=1.02)
fig.tight_layout()
out = ROOT / "own-pretrain" / "inv_freq_trajectory.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}; latest step {steps[-1]:,}, "
      f"drift L2 {np.linalg.norm(freqs[-1] - init):.4g}, "
      f"dist-to-fitted {dl_fit[-1]:.3f} (init dist {dl_fit[0]:.3f})")
