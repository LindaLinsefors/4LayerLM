"""Distribution of subcomponent activations (paper definition 2) for every
component of the Pile 4-layer decomposition:

    a_c = ||U_c|| * (V_c . phi)

where phi is the pre-weight hidden activation of the decomposed matrix.
|a_c| is the norm of the component's rank-one contribution to the layer output,
so thresholds on |a_c| are comparable across components.

Runs the target model over Pile training rows, capturing phi for all 24
decomposed matrices with forward pre-hooks, and accumulates per-component
statistics: a log-spaced histogram of |a_c| (so any threshold can be applied
later), signed sum, sum of squares, and max. Cached to cache/activations.npz.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from load import PILE_4L, ParameterComponents, load_pile_4l, pile_samples

CACHE = Path(__file__).parent / "cache" / "activations.npz"

N_SEQS = 100  # x 512 tokens = 51,200 tokens
SEQS_PER_BATCH = 10
# |a_c| histogram: underflow bin (< 1e-6, incl. exact zeros), 10 bins per
# decade up to 1e3, overflow bin
BIN_EDGES = np.geomspace(1e-6, 1e3, 91)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model, pc, _ = load_pile_4l()
    model.to(device)

    print("fetching training rows...")
    tokens = torch.stack([s[:512] for s in pile_samples(N_SEQS)]).to(device)

    modules = list(pc.components)
    factors = {
        m: (pc.components[m].V.to(device), pc.components[m].U.norm(dim=1).to(device))
        for m in modules
    }

    # forward pre-hooks capture each decomposed nn.Linear's input (= phi)
    cache = {}
    hooks = [
        model.get_submodule(m).register_forward_pre_hook(
            lambda _mod, args, name=m: cache.__setitem__(name, args[0].detach())
        )
        for m in modules
    ]

    edges = torch.tensor(BIN_EDGES, dtype=torch.float32, device=device)
    n_bins = len(BIN_EDGES) + 1
    stats = {
        m: {
            "hist": torch.zeros(V.shape[1], n_bins, dtype=torch.long, device=device),
            "sum": torch.zeros(V.shape[1], dtype=torch.float64, device=device),
            "sum_sq": torch.zeros(V.shape[1], dtype=torch.float64, device=device),
            "max_abs": torch.zeros(V.shape[1], device=device),
        }
        for m, (V, _) in factors.items()
    }

    with torch.no_grad():
        for i in range(0, N_SEQS, SEQS_PER_BATCH):
            model(tokens[i : i + SEQS_PER_BATCH])
            for m in modules:
                V, u_norm = factors[m]
                phi = cache.pop(m).reshape(-1, V.shape[0])  # (T, d_in)
                a = (phi @ V) * u_norm  # (T, C) signed activations
                s = stats[m]
                s["sum"] += a.sum(0).double()
                s["sum_sq"] += (a * a).sum(0).double()
                a_abs = a.abs()
                s["max_abs"] = torch.maximum(s["max_abs"], a_abs.amax(0))
                bin_idx = torch.bucketize(a_abs, edges)  # (T, C) in 0..n_bins-1
                C = a.shape[1]
                flat = bin_idx + torch.arange(C, device=device) * n_bins
                s["hist"] += torch.bincount(flat.reshape(-1), minlength=C * n_bins).reshape(
                    C, n_bins
                )
            print(f"  {i + SEQS_PER_BATCH}/{N_SEQS} sequences")
    for h in hooks:
        h.remove()

    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        CACHE,
        module=np.concatenate([[m] * factors[m][0].shape[1] for m in modules]),
        index=np.concatenate([np.arange(factors[m][0].shape[1]) for m in modules]),
        hist=np.concatenate([stats[m]["hist"].cpu().numpy() for m in modules]),
        sum=np.concatenate([stats[m]["sum"].cpu().numpy() for m in modules]),
        sum_sq=np.concatenate([stats[m]["sum_sq"].cpu().numpy() for m in modules]),
        max_abs=np.concatenate([stats[m]["max_abs"].cpu().numpy() for m in modules]),
        bin_edges=BIN_EDGES,
        n_tokens=N_SEQS * 512,
    )

    rms = np.sqrt(
        np.concatenate([stats[m]["sum_sq"].cpu().numpy() for m in modules]) / (N_SEQS * 512)
    )
    print(f"cached {len(rms)} components -> {CACHE}")
    print(
        f"RMS(a_c) percentiles: 1%={np.percentile(rms, 1):.3g} "
        f"50%={np.percentile(rms, 50):.3g} 99%={np.percentile(rms, 99):.3g}"
    )


if __name__ == "__main__":
    main()
