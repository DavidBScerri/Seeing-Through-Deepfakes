"""
Evaluation helpers shared by visual_module_eval.ipynb.

Scores any classifier exposing the module's predict()/ai_probabilities()
contract over the sample images and the combined_dataset test split, and
renders the report's confusion-matrix style so figures stay comparable to
training.py:evaluate_model() output.
"""

import json
import os
from io import BytesIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

LABEL_NAMES = ["Real", "AI-Generated"]

# Comprehensive evaluation set of the Community Forensics dataset: 51,836
# images over 21 generators, each paired with reals drawn from RAISE, COCO,
# FFHQ or LAION. External to combined_dataset, so it is the one test set here
# that neither backend was fitted to.
COMMFOR_EVAL_REPO = "OwensLab/CommunityForensics-Eval"
COMMFOR_EVAL_TOTAL_SHARDS = 413
COMMFOR_EVAL_INDEX_PATH = os.path.join(
    os.path.dirname(__file__), "outputs", "commfor_eval_shard_index.json"
)


def iter_split_images(split_dir):
    """
    Yields (PIL.Image, label) from a datasets-on-disk split, reading the arrow
    shards directly. Avoids load_from_disk(), which is slow to materialise the
    whole 36 GB corpus from iCloud when we only ever stream it once.

    Shards are streamed batch-by-batch: a single shard holds ~470 MB of image
    blobs, so materialising a whole one with to_pylist() stalls on I/O and
    memory before any GPU work starts.
    """
    # Take the shard list from state.json rather than globbing *.arrow: a split
    # directory can also hold HuggingFace cache-*.arrow files, which carry a
    # different schema and would silently corrupt the iteration.
    state_path = os.path.join(split_dir, "state.json")
    if os.path.exists(state_path):
        with open(state_path) as fh:
            shards = [entry["filename"] for entry in json.load(fh)["_data_files"]]
    else:
        shards = sorted(
            f for f in os.listdir(split_dir)
            if f.endswith(".arrow") and not f.startswith("cache-")
        )
    for shard in shards:
        with pa.memory_map(os.path.join(split_dir, shard)) as src:
            reader = pa.ipc.open_stream(src)
            for batch in reader:
                images = batch.column("image").to_pylist()
                labels = batch.column("label").to_pylist()
                for record, label in zip(images, labels):
                    yield Image.open(BytesIO(record["bytes"])), label


def commfor_eval_index(path=COMMFOR_EVAL_INDEX_PATH, rebuild=False):
    """
    Shard index -> {generator, architecture, real_source, n_real, n_ai, bytes}.

    Built by scanning every shard's small metadata columns over HTTP range
    reads, which never touches the image bytes. Cached to JSON because the scan
    takes a couple of minutes; the committed copy under outputs/ means the
    notebook normally never rebuilds it.
    """
    if os.path.exists(path) and not rebuild:
        with open(path) as fh:
            return {int(k): v for k, v in json.load(fh).items()}
    raise FileNotFoundError(
        f"No shard index at '{path}'. Rebuild it with "
        f"scripts/build_commfor_eval_index.py — it takes a few minutes."
    )


def select_commfor_eval_shards(
    index,
    min_per_class=24,
    max_shard_bytes=256 * 1024**2,
    generators=None,
):
    """
    Chooses shards giving every generator both real and AI images, cheapest
    first, and reports what the size cap excluded.

    Shards are NOT one-generator-with-paired-reals: a generator spans many
    shards, and an individual shard is often entirely real or entirely AI
    (`model_name` on a real row records the generator it is paired against).
    Evenly spaced sampling therefore yields generators with only one class
    present, whose AP is undefined — hence selecting per generator per class.

    Size matters as much as count. Shards run from 4.7 MB to 2.9 GB depending
    on the generator's output resolution, so a naive pick can spend gigabytes
    on one generator; `max_shard_bytes` caps per-shard cost, and anything a cap
    excludes entirely is returned in `skipped` rather than silently dropped.

    Returns:
        (shard_ids, plan) where plan maps generator -> selection detail,
        including a "skipped" list of generators the cap excluded.
    """
    by_generator = {}
    for shard, rec in index.items():
        if "error" in rec:
            continue
        if generators and rec["generator"] not in generators:
            continue
        by_generator.setdefault(rec["generator"], []).append((shard, rec))

    chosen, plan, skipped = set(), {}, []
    for generator, records in sorted(by_generator.items()):
        affordable = [(s, r) for s, r in records if r["bytes"] <= max_shard_bytes]
        detail = {"shards": [], "n_real": 0, "n_ai": 0, "bytes": 0}

        for key, counter in (("n_ai", "n_ai"), ("n_real", "n_real")):
            # Cheapest-first, richest-per-byte: covers the class in as few
            # shards (and megabytes) as possible.
            candidates = sorted(
                (r for r in affordable if r[1][counter] > 0),
                key=lambda r: (r[1]["bytes"] / max(r[1][counter], 1), r[1]["bytes"]),
            )
            for shard, rec in candidates:
                if detail[key] >= min_per_class:
                    break
                if shard not in detail["shards"]:
                    detail["shards"].append(shard)
                    detail["n_real"] += rec["n_real"]
                    detail["n_ai"] += rec["n_ai"]
                    detail["bytes"] += rec["bytes"]

        if detail["n_real"] == 0 or detail["n_ai"] == 0:
            skipped.append({
                "generator": generator,
                "reason": "no affordable shard for one class",
                "smallest_bytes": min(r["bytes"] for _, r in records),
            })
            continue

        chosen.update(detail["shards"])
        plan[generator] = detail

    return sorted(chosen), {"generators": plan, "skipped": skipped}


def iter_commfor_eval(
    shard_ids,
    repo_id=COMMFOR_EVAL_REPO,
    max_images_per_shard=None,
    index=None,
    progress=True,
):
    """
    Yields (PIL.Image, label, model_name) from the Community Forensics eval set,
    fetching each shard through the HuggingFace cache on first use.

    label 1 == AI-generated. Reals carry the model_name of the generator they
    are paired against, which is what makes the per-generator grouping work.

    Streams row batches rather than reading the file whole: the largest shards
    hold 25 images per ~550 MB row group, so materialising one costs gigabytes
    of RAM and looks like a hang once the machine starts swapping.

    `max_images_per_shard` selects an even split of both classes rather than a
    prefix. Rows within a shard are grouped by class, so taking the first N
    would hand back an all-real or all-AI sample and undo the class balance
    select_commfor_eval_shards() arranged.

    Progress is logged per shard, before the fetch, because a multi-gigabyte
    download is otherwise indistinguishable from a stall.
    """
    import time

    from huggingface_hub import hf_hub_download

    for n, shard in enumerate(shard_ids, 1):
        rec = (index or {}).get(shard, {})
        if progress:
            size = f"{rec['bytes'] / 1024**2:.0f} MB" if "bytes" in rec else "size unknown"
            label = f" {rec['generator']}" if "generator" in rec else ""
            print(f"  [{n}/{len(shard_ids)}] shard {shard}{label} ({size})...",
                  end="", flush=True)

        started = time.time()
        path = hf_hub_download(
            repo_id,
            f"data/CompEval-{shard:05d}-of-{COMMFOR_EVAL_TOTAL_SHARDS:05d}.parquet",
            repo_type="dataset",
        )
        parquet = pq.ParquetFile(path)

        wanted = None
        if max_images_per_shard:
            # The label column alone is tiny, so reading it up front to plan a
            # balanced subset costs far less than decoding images we discard.
            labels = parquet.read(columns=["label"]).to_pydict()["label"]
            per_class = max(max_images_per_shard // 2, 1)
            wanted = set()
            for target in (0, 1):
                rows = [i for i, lab in enumerate(labels) if lab == target]
                # Even stride across the shard, so the sample is not biased
                # toward whichever images happen to sit at the start.
                if len(rows) > per_class:
                    step = len(rows) / per_class
                    rows = [rows[int(k * step)] for k in range(per_class)]
                wanted.update(rows)

        emitted, row = 0, 0
        for batch in parquet.iter_batches(
            batch_size=8, columns=["image_data", "label", "model_name"]
        ):
            rows_batch = batch.to_pydict()
            for blob, lab, generator in zip(
                rows_batch["image_data"], rows_batch["label"], rows_batch["model_name"]
            ):
                if wanted is None or row in wanted:
                    emitted += 1
                    yield Image.open(BytesIO(blob)), lab, generator
                row += 1

        if progress:
            print(f" {emitted} images in {time.time() - started:.0f}s", flush=True)


def score_commfor_eval(
    classifier,
    shard_ids,
    batch_size=16,
    max_images_per_shard=None,
    index=None,
    progress=True,
):
    """
    Scores a classifier over the selected eval shards.

    Returns:
        (probs, labels, generators) as numpy arrays.
    """
    probs, labels, generators = [], [], []
    batch_imgs, batch_meta = [], []

    def flush():
        if batch_imgs:
            probs.extend(ai_probabilities(classifier, batch_imgs))
            labels.extend(m[0] for m in batch_meta)
            generators.extend(m[1] for m in batch_meta)
            batch_imgs.clear()
            batch_meta.clear()

    for image, label, generator in iter_commfor_eval(
        shard_ids,
        max_images_per_shard=max_images_per_shard,
        index=index,
        progress=progress,
    ):
        batch_imgs.append(image)
        batch_meta.append((label, generator))
        if len(batch_imgs) == batch_size:
            flush()
    flush()

    return (
        np.asarray(probs, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(generators, dtype=object),
    )


def per_generator_metrics(probs, labels, generators, threshold=0.55, min_per_class=24):
    """
    Paper-style breakdown: AP and accuracy per generator, averaged over
    generators to give mAP/mAcc.

    The paper averages per generator rather than pooling, so a generator with
    few images counts the same as one with many. Reported alongside the pooled
    AUC, which is what the other test sets in this notebook use.

    Generators lacking both classes (possible when few shards are sampled) are
    skipped for AP/AUC — those are undefined on a single-class group.
    """
    rows = {}
    for generator in sorted(set(generators.tolist())):
        mask = generators == generator
        g_probs, g_labels = probs[mask], labels[mask]
        n_real = int((g_labels == 0).sum())
        n_ai = int((g_labels == 1).sum())
        both_classes = n_real > 0 and n_ai > 0
        preds = (g_probs >= threshold).astype(int)
        rows[generator] = {
            "n": int(mask.sum()),
            "n_real": n_real,
            "n_ai": n_ai,
            # AP over a badly skewed group is dominated by the base rate — with
            # one real against 250 AI it is ~1.0 for any ranking. Flagged so a
            # weak estimate is never read as a strong result.
            "balanced": min(n_real, n_ai) >= min_per_class,
            "ap": float(average_precision_score(g_labels, g_probs)) if both_classes else None,
            "auc": float(roc_auc_score(g_labels, g_probs)) if both_classes else None,
            "accuracy": float(accuracy_score(g_labels, preds)),
        }

    scored = [r for r in rows.values() if r["ap"] is not None]
    balanced = [r for r in rows.values() if r["balanced"]]
    return {
        "n": int(len(labels)),
        "n_generators": len(rows),
        # mAP/AUC can only average over generators that have both classes in
        # the sample. Sample few shards and several generators arrive with no
        # paired reals, so report the backing count rather than let mAP look
        # like it covers every generator listed.
        "n_generators_scored": len(scored),
        "n_generators_balanced": len(balanced),
        "threshold": threshold,
        "mAP": float(np.mean([r["ap"] for r in scored])) if scored else None,
        "mAcc": float(np.mean([r["accuracy"] for r in rows.values()])),
        # Restricted to generators with min_per_class of BOTH classes. Prefer
        # this over mAP when quoting a figure: it drops the groups whose AP is
        # propped up by a lopsided base rate.
        "mAP_balanced": float(np.mean([r["ap"] for r in balanced])) if balanced else None,
        "mAcc_balanced": float(np.mean([r["accuracy"] for r in balanced])) if balanced else None,
        "pooled": metrics_at(probs, labels, threshold),
        "per_generator": rows,
    }


def ai_probabilities(classifier, images):
    """
    P(AI) for a list of PIL images, using the classifier's batched path when it
    has one (CommunityForensicsClassifier) and predict() otherwise.
    """
    if hasattr(classifier, "ai_probabilities"):
        return classifier.ai_probabilities(images)

    from src.integration_pipeline.fusion import extract_visual_ai_probability

    return [extract_visual_ai_probability(classifier.predict(img)) for img in images]


def score_split(classifier, split_dir, batch_size=16, limit=None, progress=True):
    """
    Streams a combined_dataset split through a classifier.

    Returns:
        (probs, labels) as numpy arrays, where label 1 == AI-generated.
    """
    probs, labels = [], []
    batch_imgs, batch_labels = [], []

    def flush():
        if batch_imgs:
            probs.extend(ai_probabilities(classifier, batch_imgs))
            labels.extend(batch_labels)
            batch_imgs.clear()
            batch_labels.clear()
            if progress and len(probs) % (batch_size * 20) == 0:
                print(f"  scored {len(probs)}...", flush=True)

    for image, label in iter_split_images(split_dir):
        batch_imgs.append(image)
        batch_labels.append(label)
        if len(batch_imgs) == batch_size:
            flush()
        if limit is not None and len(probs) >= limit:
            break
    flush()

    return np.asarray(probs, dtype=float), np.asarray(labels, dtype=int)


def metrics_at(probs, labels, threshold=0.55):
    """
    Threshold-free (AUC) and thresholded metrics for one classifier.

    The default threshold is the canonical fused decision threshold so the
    per-stream numbers line up with what fusion actually sees.
    """
    preds = (np.asarray(probs) >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "n": int(len(labels)),
        "threshold": threshold,
        "auc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else None,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }


def plot_confusion_matrices(results, output_path=None):
    """
    Side-by-side confusion matrices, one per classifier, in the report's style.

    Args:
        results: {display_name: metrics dict from metrics_at()}
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    axes = np.atleast_1d(axes)

    for ax, (name, metrics) in zip(axes, results.items()):
        cm = np.asarray(metrics["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(LABEL_NAMES); ax.set_yticklabels(LABEL_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{name}\nacc {metrics['accuracy']:.3f} @ {metrics['threshold']}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Figure saved to {output_path}")
    return fig


def plot_roc(panels, output_path=None):
    """
    ROC curves, one panel per test set with every classifier overlaid.

    Args:
        panels: {test_set_name: {classifier_name: (probs, labels)}}, or a bare
                {classifier_name: (probs, labels)} for a single unnamed panel.
    """
    import matplotlib.pyplot as plt

    first = next(iter(panels.values()))
    if isinstance(first, tuple):
        panels = {"Real vs AI-Generated": panels}

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5))
    axes = np.atleast_1d(axes)

    for ax, (panel_name, curves) in zip(axes, panels.items()):
        for name, (probs, labels) in curves.items():
            fpr, tpr, _ = roc_curve(labels, probs)
            ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(labels, probs):.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(panel_name)
        ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Figure saved to {output_path}")
    return fig


def save_results(results, output_path):
    """Writes a run's metrics dict to JSON under the module's outputs/ dir."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Metrics saved to {output_path}")
