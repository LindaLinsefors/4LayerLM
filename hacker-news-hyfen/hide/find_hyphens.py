"""Find hacker-news title|username hyphens in the 4000 cached Pile rows.

A "hyphen" is exactly the token 428 (' -').  A hacker-news hyphen is a
' -' token on the *first line of a document* (right after <|endoftext|>,
possibly with intervening whitespace-only tokens, or at row start when the
chunk cut truncated the context) that is followed by a single username word
and a newline: the Pile-HN heading format "Title - username\n[url\n]======".

Writes:
  cache/hn_positions.npz  -- hyphen_rc (N,2) all token-428 (row, pos);
                             cand_rc (17,2) auto-detected heading candidates;
                             hn_rc (12,2) = candidates minus manual rejects
  cache/candidates.txt    -- every candidate with context, one block each,
                             for manual review

Rows are truncated to 512 tokens (model context) so positions match the
Modal CI run.
"""

import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from load import load_tokenizer  # noqa: E402

HYPHEN_ID, EOS_ID = 428, 0
USER_RE = re.compile(r"^ ?[A-Za-z0-9_.\-]+$")

# Manual review of cache/candidates.txt (2026-09-02): auto-detected heading
# candidates that are NOT hacker-news title|username hyphens.
REJECTS: dict[tuple[int, int], str] = {
    (392, 6): "math dataset, '0 - -14) + -2.' (row-start arithmetic)",
    (938, 8): "math dataset, ', w, -1/4, -5' (row-start list)",
    (2813, 6): "math dataset, '2*sqrt(2) - 2' (row-start arithmetic)",
    (2567, 272): "news headline quote attribution ('... - Coulthard', "
                 "article text follows; no url/====== HN structure)",
    (2573, 230): "JS code file, '// (c) ammap.com | SVG ... map of Uruguay - Low'",
}

tok = load_tokenizer("pile_4l")
rows = [r[:512] for r in torch.load(ROOT / "context-loss/hide/cache/pile_rows.pt")]
assert len(rows) == 4000

dec_cache: dict[int, str] = {}


def dec(i: int) -> str:
    if i not in dec_cache:
        dec_cache[i] = tok.decode([i])
    return dec_cache[i]


hyphen_rc, cand_rc, blocks = [], [], []
for r, row in enumerate(rows):
    ids = row.tolist()
    for p, t in enumerate(ids):
        if t != HYPHEN_ID:
            continue
        hyphen_rc.append((r, p))

        # forward: username = text up to the first newline after the hyphen
        buf, j = "", p + 1
        while j < len(ids) and "\n" not in buf and len(buf) < 40:
            buf += dec(ids[j])
            j += 1
        user = buf.split("\n")[0]
        user_ok = bool(USER_RE.match(user)) and user.strip() != ""
        truncated_fwd = "\n" not in buf

        # backward: walk to the start of the line the hyphen is on
        i = p - 1
        while i >= 0 and ids[i] != EOS_ID and "\n" not in dec(ids[i]):
            i -= 1
        title = tok.decode(ids[i + 1 : p])
        if i < 0:
            kind = "rowstart"
        elif ids[i] == EOS_ID:
            kind = "docstart"
        else:
            # newline token: only whitespace tokens may separate it from EOS
            k = i
            while k >= 0 and ids[k] != EOS_ID and dec(ids[k]).strip() == "":
                k -= 1
            kind = "docstart" if (k >= 0 and ids[k] == EOS_ID) else (
                "rowstart" if k < 0 else "middoc")

        is_cand = user_ok and kind in ("docstart", "rowstart") and title.strip() != ""
        if is_cand:
            cand_rc.append((r, p))
            hn = (r, p) not in REJECTS
            ctx = tok.decode(ids[max(0, i - 5) : min(len(ids), j + 25)])
            blocks.append(
                f"{'HN  ' if hn else 'REJ '} row {r} pos {p}  kind={kind}"
                f"{' truncated-user' if truncated_fwd else ''}\n"
                f"  title={title!r}\n  user={user!r}\n  ctx={ctx!r}\n"
                + (f"  reject: {REJECTS[(r, p)]}\n" if not hn else ""))

hn_rc = [rc for rc in cand_rc if rc not in REJECTS]
assert all(rc in cand_rc for rc in REJECTS), "stale REJECTS entry"

(HERE / "cache").mkdir(exist_ok=True)
np.savez(HERE / "cache/hn_positions.npz",
         hyphen_rc=np.array(hyphen_rc), cand_rc=np.array(cand_rc),
         hn_rc=np.array(hn_rc))
(HERE / "cache/candidates.txt").write_text("".join(blocks), encoding="utf-8")
print(f"token ' -' occurrences: {len(hyphen_rc)}")
print(f"heading candidates:     {len(cand_rc)}")
print(f"hacker-news hyphens:    {len(hn_rc)}")
