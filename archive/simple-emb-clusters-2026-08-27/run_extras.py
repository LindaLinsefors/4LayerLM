"""Follow-up statistics for the report:

- what the silhouette-optimal KMeans k=2 split actually separates,
- the rare-token cone: norms + pairwise cosines of never-seen tokens,
- overall anisotropy of the embedding cloud.

Writes cache/extras.json.
"""

import json

import numpy as np

from common import CACHE, CLASS_NAMES, embeddings_and_tokens, frequencies, token_classes


def mean_pairwise_cos(Xn: np.ndarray) -> float:
    """mean_{i != j} cos(x_i, x_j), via |sum x|^2 = sum_ij cos."""
    s = Xn.sum(0)
    n = len(Xn)
    return float((s @ s - n) / (n * (n - 1)))


def main():
    emb, tokens = embeddings_and_tokens()
    classes = token_classes(tokens)
    freq = frequencies(len(tokens))
    norms = np.linalg.norm(emb, axis=1)
    Xn = emb / norms[:, None]

    labels2 = np.load(CACHE / "labels.npz")["kmeans_opt"]
    k2 = {}
    for c in (0, 1):
        m = labels2 == c
        k2[f"cluster {c}"] = dict(
            n=int(m.sum()),
            mean_norm=float(norms[m].mean()),
            median_freq=float(np.median(freq[m])),
            zero_freq_frac=float(np.mean(freq[m] == 0)),
            suffix_frac=float(np.mean(classes[m] == CLASS_NAMES.index("##suffix"))),
            examples_frequent=[tokens[i] for i in np.where(m)[0][np.argsort(-freq[m])][:10]],
        )

    rare = freq == 0
    seen = ~rare
    extras = dict(
        n_tokens_counted=int(freq.sum()),
        n_zero_freq=int(rare.sum()),
        kmeans_k2=k2,
        norms=dict(rare_mean=float(norms[rare].mean()), seen_mean=float(norms[seen].mean()),
                   rare_p90=float(np.percentile(norms[rare], 90))),
        mean_pairwise_cos=dict(all=mean_pairwise_cos(Xn),
                               rare=mean_pairwise_cos(Xn[rare]),
                               seen=mean_pairwise_cos(Xn[seen])),
        # correlation between log-frequency and norm
        corr_logfreq_norm=float(np.corrcoef(np.log10(1 + freq), norms)[0, 1]),
    )
    print(json.dumps(extras, indent=1, ensure_ascii=False))
    (CACHE / "extras.json").write_text(json.dumps(extras, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
