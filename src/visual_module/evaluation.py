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
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

LABEL_NAMES = ["Real", "AI-Generated"]


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


def plot_roc(curves, output_path=None):
    """
    Overlaid ROC curves.

    Args:
        curves: {display_name: (probs, labels)}
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (probs, labels) in curves.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(labels, probs):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Real vs AI-Generated — ROC")
    ax.legend(loc="lower right")
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
