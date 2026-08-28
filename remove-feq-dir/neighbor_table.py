"""Nearest-neighbor comparison table for report.md.

For each model, sample 10 random tokens from the frequent/alive set and list
their top-10 cosine neighbors (within the same set) in three geometries:
  raw   -- the raw embedding rows, as-is
  pc1   -- rows mean-centered over the set, top PC projected out
  vcov  -- rows mean-centered, covariance-with-log-count direction projected out

Writes neighbor_table.md (markdown, pasted into report.md).
Pile tokens are decoded; a leading space shows as an underscore prefix "_".
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from load import PILE_4L, SIMPLE_2L, load_tokenizer  # noqa: E402

N_QUERIES, N_NEIGHBORS = 10, 10


def neighbors(E: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Indices (into E's rows) of the k nearest rows by cosine, per query row."""
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    sims = En[queries] @ En.T
    for r, q in enumerate(queries):
        sims[r, q] = -np.inf
    return np.argsort(-sims, axis=1)[:, :k]


def build(name: str, emb: np.ndarray, counts: np.ndarray, thr: int,
          show) -> list[str]:
    keep = counts >= thr
    ids = np.where(keep)[0]
    A = emb[keep].astype(np.float64)
    Ac = A - A.mean(0)
    yc = np.log(counts[keep].astype(np.float64))
    yc -= yc.mean()

    _, _, vt = np.linalg.svd(Ac, full_matrices=False)
    pc1 = vt[0]
    vcov = Ac.T @ yc
    vcov /= np.linalg.norm(vcov)

    spaces = {"raw embeddings": A,
              "PC1 removed": Ac - np.outer(Ac @ pc1, pc1),
              "v_cov removed": Ac - np.outer(Ac @ vcov, vcov)}

    rng = np.random.default_rng(0)
    queries = rng.choice(len(ids), N_QUERIES, replace=False)
    nn = {label: neighbors(E, queries, N_NEIGHBORS) for label, E in spaces.items()}

    lines = [f"### {name}  ({keep.sum():,} {'frequent' if name == 'pile_4l' else 'alive'} "
             f"tokens; neighbors drawn from the same set)\n",
             "| token | " + " | ".join(f"top-{N_NEIGHBORS} ({lbl})" for lbl in spaces) + " |",
             "|---|" + "---|" * len(spaces)]
    for r, q in enumerate(queries):
        cells = ["<br>".join(show(ids[j]) for j in nn[lbl][r]) for lbl in spaces]
        lines.append(f"| {show(ids[q])} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def md_token(s: str) -> str:
    s = s.replace(" ", "_").replace("|", "\\|").replace("`", "'")
    return f"`{s}`" if s.strip("_") else f"`{repr(s)}`"


pile_emb = load_file(
    PILE_4L / "target_model_t-9d2b8f02" / "model_step_99999.safetensors"
)["wte.weight"].float().numpy()
pile_rows = torch.stack(torch.load(
    ROOT / "context-loss/hide/cache/pile_rows.pt", map_location="cpu", weights_only=True))
pile_counts = np.bincount(pile_rows.flatten().numpy(), minlength=len(pile_emb))
pile_tok = load_tokenizer("pile_4l")

simple_emb = torch.load(
    SIMPLE_2L / "target_model_gf6rbga0" / "model_step_99999.pt",
    map_location="cpu", weights_only=True,
)["wte.weight"].float().numpy()
table = pd.read_csv(ROOT / "simple-token-table/token_table.csv")
simple_counts = np.zeros(len(simple_emb), dtype=np.int64)
simple_counts[table["id"].to_numpy()] = table["freq"].to_numpy()
simple_tok = load_tokenizer("simple_2l")

out = ["Legend: `_` marks a leading space in pile tokens (GPT-NeoX tokenizer); "
       "`##` marks SimpleStories continuation pieces.\n"]
out += build("pile_4l", pile_emb, pile_counts, 7,
             lambda i: md_token(pile_tok.decode([int(i)])))
out += build("simple_2l", simple_emb, simple_counts, 10,
             lambda i: md_token(simple_tok.convert_ids_to_tokens([int(i)])[0]))

path = Path(__file__).parent / "neighbor_table.md"
path.write_text("\n".join(out), encoding="utf-8")
print(f"wrote {path}")
