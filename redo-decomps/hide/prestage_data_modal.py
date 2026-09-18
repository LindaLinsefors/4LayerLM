"""Prestage the decomposition-trainer datasets onto volume vpd-4layer.

The HF dataset danbraunai/pile-uncopyrighted-tok-shuffled is ALREADY in the
trainer's shard format (parquet, one `input_ids` column of int32 513-token
rows, 48,638-row row groups >= the 256 global batch) -- prestaging is just
downloading shards to flat filenames + writing meta.json:

  /data/sink-models/datasets/pile_neox_tok_512/      train-00000..00105  (106 shards,
      ~21.6 GB, 25.8M rows ~= one no-repeat epoch of the run's 25.6M-row consumption)
  /data/sink-models/datasets/pile_neox_tok_512_val/  val-00000            (231k rows)

data_root stays /data/sink-models (pretrain_cache/spd-t-87f91319 already there),
matching `<data_root>/datasets/<name>` resolution at the sink-loader commit.

Run:  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 modal run redo-decomps/hide/prestage_data_modal.py
"""

import modal

app = modal.App("ropefix-prestage-data")
vol = modal.Volume.from_name("vpd-4layer")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub", "requests", "pyarrow")

REPO = "danbraunai/pile-uncopyrighted-tok-shuffled"
N_TRAIN = 106
TRAIN_DIR = "/data/sink-models/datasets/pile_neox_tok_512"
VAL_DIR = "/data/sink-models/datasets/pile_neox_tok_512_val"
META = '{"seq_len":512,"tokenizer_name":"EleutherAI/gpt-neox-20b"}\n'


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def fetch(job: tuple) -> str:
    """Download one repo file to a flat local filename (idempotent by size)."""
    import shutil
    from pathlib import Path

    import requests
    from huggingface_hub import hf_hub_url, get_hf_file_metadata

    repo_path, dest = job
    dest = Path(dest)
    url = hf_hub_url(REPO, repo_path, repo_type="dataset")
    size = get_hf_file_metadata(url).size
    if dest.exists() and dest.stat().st_size == size:
        return f"skip {dest.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f, length=8 << 20)
    assert tmp.stat().st_size == size, (dest.name, tmp.stat().st_size, size)
    tmp.rename(dest)
    return f"got {dest.name} ({size / 1e6:.0f} MB)"


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def finalize() -> str:
    """Write meta.json files and verify the store with the trainer's own scan logic."""
    from pathlib import Path

    import pyarrow.parquet as pq

    vol.reload()  # pick up the fetch containers' commits
    out = []
    for d, expect in [(TRAIN_DIR, N_TRAIN), (VAL_DIR, 1)]:
        d = Path(d)
        files = sorted(d.glob("*.parquet"))
        assert len(files) == expect, (d, len(files), expect)
        total_rows = 0
        for f in files:
            md = pq.ParquetFile(f).metadata
            total_rows += md.num_rows
            assert all(md.row_group(i).num_rows >= 256 for i in range(md.num_row_groups)), f
        t = pq.ParquetFile(files[0]).read_row_group(0, columns=["input_ids"]).slice(0, 1)
        assert len(t.column(0)[0].as_py()) == 513, files[0]
        (d / "meta.json").write_text(META)
        out.append(f"{d.name}: {len(files)} shards, {total_rows:,} rows, meta.json written")
    vol.commit()
    return "\n".join(out)


@app.local_entrypoint()
def main() -> None:
    jobs = [(f"data/train-{i:05d}-of-02021.parquet", f"{TRAIN_DIR}/train-{i:05d}.parquet")
            for i in range(N_TRAIN)]
    jobs.append(("data/val-00000-of-00012.parquet", f"{VAL_DIR}/val-00000.parquet"))
    for msg in fetch.map(jobs):
        print(msg, flush=True)
    print(finalize.remote())
