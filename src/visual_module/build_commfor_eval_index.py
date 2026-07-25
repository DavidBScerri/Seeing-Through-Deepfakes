"""
Builds the Community Forensics eval-set shard index used by evaluation.py.

The eval set is 206 GB over 413 parquet shards, so choosing which shards to
score has to happen without downloading them. This scans only the small
metadata columns (generator, label, source) over HTTP range reads — a few
minutes total, no image bytes fetched — and caches the result as JSON.

The index is committed, so this normally never needs re-running; do so only if
the upstream dataset changes.

    python -m src.visual_module.build_commfor_eval_index
"""

import argparse
import collections
import concurrent.futures as cf
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from src.visual_module.evaluation import (
    COMMFOR_EVAL_INDEX_PATH,
    COMMFOR_EVAL_REPO,
    COMMFOR_EVAL_TOTAL_SHARDS,
)

# Metadata only — small, dictionary-encoded columns. Deliberately excludes
# image_data, which is the entire 206 GB.
METADATA_COLUMNS = ["model_name", "label", "real_source", "architecture"]


def shard_sizes(repo_id=COMMFOR_EVAL_REPO):
    """Byte size per shard, from the repo tree listing."""
    sizes = {}
    for entry in HfApi().list_repo_tree(
        repo_id, "data", repo_type="dataset", recursive=False
    ):
        if entry.path.endswith(".parquet"):
            sizes[int(entry.path.split("-")[1])] = entry.size
    return sizes


def scan_shard(shard, sizes, repo_id=COMMFOR_EVAL_REPO, attempts=3):
    """Reads one shard's metadata columns. Retries — this is a network read."""
    path = (
        f"datasets/{repo_id}/data/"
        f"CompEval-{shard:05d}-of-{COMMFOR_EVAL_TOTAL_SHARDS:05d}.parquet"
    )
    for attempt in range(attempts):
        try:
            rows = pq.ParquetFile(HfFileSystem().open(path, "rb")).read(
                columns=METADATA_COLUMNS
            ).to_pydict()
            counts = collections.Counter(rows["label"])
            return shard, {
                "generator": collections.Counter(rows["model_name"]).most_common(1)[0][0],
                "architecture": collections.Counter(rows["architecture"]).most_common(1)[0][0],
                "real_source": collections.Counter(rows["real_source"]).most_common(1)[0][0],
                "n_real": counts[0],
                "n_ai": counts[1],
                "bytes": sizes[shard],
            }
        except Exception as exc:
            if attempt == attempts - 1:
                return shard, {"error": str(exc)[:200], "bytes": sizes.get(shard)}
    return shard, {"error": "unreachable"}


def build_index(output_path=COMMFOR_EVAL_INDEX_PATH, workers=16):
    sizes = shard_sizes()
    print(f"Listed {len(sizes)} shards, {sum(sizes.values()) / 1024**3:.0f} GB total.")

    index, done = {}, 0
    with cf.ThreadPoolExecutor(workers) as pool:
        futures = [
            pool.submit(scan_shard, shard, sizes)
            for shard in range(COMMFOR_EVAL_TOTAL_SHARDS)
        ]
        for future in cf.as_completed(futures):
            shard, record = future.result()
            index[shard] = record
            done += 1
            if done % 25 == 0:
                print(f"  scanned {done}/{COMMFOR_EVAL_TOTAL_SHARDS}", flush=True)

    errors = {s: r for s, r in index.items() if "error" in r}
    if errors:
        print(f"WARNING: {len(errors)} shards failed to scan: {sorted(errors)[:10]}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump({str(s): index[s] for s in sorted(index)}, fh, indent=1)
    print(f"Index written to {output_path}")

    generators = collections.Counter(
        r["generator"] for r in index.values() if "error" not in r
    )
    print(f"\n{len(generators)} generators:")
    for generator in sorted(generators):
        records = [r for r in index.values() if r.get("generator") == generator]
        print(
            f"  {generator:32s} shards={len(records):3d} "
            f"real={sum(r['n_real'] for r in records):5d} "
            f"ai={sum(r['n_ai'] for r in records):5d} "
            f"{sum(r['bytes'] for r in records) / 1024**3:6.1f} GB"
        )
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=COMMFOR_EVAL_INDEX_PATH)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    build_index(args.output, args.workers)
