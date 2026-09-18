# `examples.bin` format — reverse-engineering notes (2026-09-17)

State of knowledge about the task-1423 harvest example storage, from a long
reverse-engineering session against
`sink-models/runs/p-d60af588/harvest/h-…-task1423-v1/published/shards/`.
The official reader (`param_decomp.harvest.reader.HarvestReader` per the
bundle README) is **not public**: not at PR #1002's commit `82a67f71c`, not on
latest `main`, not on any of the 2,446 remote branches, not anywhere in
history (`example_component_offset` appears in no public commit). The old
`param_decomp/harvest/` module in public history (last at `53965b5e7`) is the
previous JSON-in-SQLite pipeline, a different schema.

## Solid (verified) facts

- Each shard = `components.parquet` (per-component sums; sums / 10,002,432
  exactly equal the sqlite means) + `examples.bin` of exactly
  **37,901 bytes × n_components**.
- `harvest.sqlite` `components` table: state, firing_count, means, max,
  eligible_region_count, example_count (≤ 100), example_file,
  example_component_offset (== component_idx for every component checked).
- `examples.bin` global layout: `[n_comps × u8 example_count]`
  `[n_comps × 100 u8 "window length" slots]` `[n_comps × 37,800-byte blocks]`.
  - The first section matches sqlite example_count **768/768** exactly.
  - The second: slot c = bytes `[C + 100c, C + 100c + 100)` (C = n_comps);
    values ≤ 41 with exactly example_count nonzero entries for **all 768**
    q_proj components (brute-force stride search: 100 is the unique stride).
    Values are window lengths (41 = full; 21-22 for a pos-0/EOS-firing
    component whose windows are clipped at a document boundary).
- Window token arrays exist inside the per-component blocks as **u32 token
  ids** (< 50,277), 41-token windows zero-padded past document boundaries;
  decoded windows are coherent Pile text and (for stale ' for'-component
  content) all contain the expected trigger token.
- Fire-position arrays exist as **u8 values ≤ 40** (mode exactly 20 = window
  center; decaying tail toward 0) — h.0.attn.k_proj:0's block starts with
  ≈ erc (29,596) of them, matching eligible_region_count.
- 8-byte records exist (seen at one component's block+37,000) that look like
  per-example metadata `(u16, u32-ish, u16)` tables — semantics unresolved
  (region index / deterministic-hash priority candidates failed validation
  against the corpus).

## The blocker: stale scratch buffers

The writer evidently reuses a per-component buffer: bytes beyond a
component's real data retain the **previous component's** content. Observed
directly: h.0.attn.k_proj:28 has example_count = 1 (firing_count = 1) but its
37,800-byte block contains ~90 coherent 41-token windows all centered on
' for' — a different component's leftovers. Consequently:

- zero-example components have garbage-filled blocks (⇒ every "empty slots
  are zeroed" heuristic fails);
- per-component section boundaries cannot be inferred from content alone
  without risking **cross-component misattribution** of example windows;
- the per-component section order/sizes (fire-positions ~erc·u8, tokens
  4·Σwlen, CI trace location + dtype, loc records 8·ec, alignment/padding)
  are not yet pinned; CI traces were not conclusively located at all
  (candidate f16/f32/u8-scaled encodings all failed clean validation).

## What would resolve it

Ask the co-authors for `param_decomp/harvest/{reader,domain,storage}.py` at
the internal commit that wrote these harvests (same channel as the pending
RoPE-convention question — internal training commit `37519f99`, "what RoPE
did task-1423 use?"). One file of writer code makes the whole format exact.

## Validation gotcha for later

Our own C/D/E per-token CI caches are broken-RoPE (public-loader) values;
the harvest presumably is not. Do NOT treat disagreement between a parsed
example and our caches as evidence of a misparse — position-sensitive
components (e.g. pos-0-locked ones) may legitimately differ. Use token-level
invariants (trigger token at the fire position, window text = real corpus
text at a matching location) for validation instead.

The corpus is rows 0..19,536 of the pre-tokenized Pile dataset (512-token
rows; 10,002,432 tokens; same source as our `pile_rows.pt`, which covers
rows 0..3,999 = the first 20% — usable as ground truth for matching windows).
