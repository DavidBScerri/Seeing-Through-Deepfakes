"""
two_stage_finetuning.py
========================
Helper utilities for the fine-tuning notebook.

Provides dataset loading (Zitacron/real-vs-ai-corpus), preprocessing,
augmentation, training, and evaluation helpers for training a
ConvNeXt V2-based AI-vs-real image classifier.

This module is imported by the notebook; all heavy lifting lives here so
the notebook stays beginner-friendly.
"""

import os, json, io, torch, numpy as np
from PIL import Image
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ── helpers ────────────────────────────────────────────────────────────────────

def get_device():
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(eval_pred):
    """Metric function used by HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# ── Model loader ───────────────────────────────────────────────────────────────

BASE_MODEL_ID = "facebook/convnextv2-large-22k-224"


def load_model_from(source="base", device=None):
    """
    Load a model + processor from any source, ready for training or evaluation.

    Parameters
    ----------
    source : str
        One of:
        - ``"base"`` → load the facebook/convnextv2-large-22k-224 HuggingFace model
        - A local directory path (e.g. ``"outputs/models/run_01_zitacron"``)
        - A HuggingFace model ID
    device : torch.device or None
        Device to place the model on.  Auto-detected if None.

    Returns
    -------
    (model, processor)
    """
    if device is None:
        device = get_device()

    model_id = BASE_MODEL_ID if source == "base" else source

    print(f"📦  Loading model from: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    
    if source == "base" or model_id == BASE_MODEL_ID:
        model = AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=2,
            id2label={0: "human", 1: "AI-generated"},
            label2id={"human": 0, "AI-generated": 1},
            ignore_mismatched_sizes=True
        ).to(device)
    else:
        model = AutoModelForImageClassification.from_pretrained(
            model_id, ignore_mismatched_sizes=True
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅  Model loaded  –  {n_params:,} parameters  (device: {device})")
    return model, processor


# ── Zitacron dataset builder ──────────────────────────────────────────────────

def build_zitacron_subset(
    total_train=10000,
    total_val=2000,
    total_test=2000,
    seed=42,
    hf_token=None,
):
    """
    Load Zitacron/real-vs-ai-corpus and build a balanced subset.

    Balancing strategy:
      - Group by (source_dataset, label)
      - Allocate equal quota per group per split
      - If a group has fewer samples than its quota, take all and
        redistribute the remainder proportionally

    Only keeps columns: image, label, source_dataset.

    Returns
    -------
    DatasetDict  with splits 'train', 'validation', 'test'
    """
    local_path = os.path.join("data", "zitacron_subset")
    if os.path.exists(local_path):
        print(f"📦  Loading cached Zitacron subset from {local_path}…")
        from datasets import load_from_disk
        return load_from_disk(local_path)

    print("📡  Streaming Zitacron/real-vs-ai-corpus to build a balanced subset…")
    from huggingface_hub import list_repo_files
    import random
    from datasets import Dataset

    # 1. Get the list of all parquet files
    try:
        files = list_repo_files("Zitacron/real-vs-ai-corpus", repo_type="dataset", token=hf_token)
        parquet_files = [f for f in files if f.endswith(".parquet")]
    except Exception as e:
        print(f"❌ Error listing repository files: {e}")
        print("Falling back to full dataset download (could be very slow)...")
        # Fallback to the original full load behavior
        kwargs = {}
        if hf_token:
            kwargs["token"] = hf_token
        ds = load_dataset("Zitacron/real-vs-ai-corpus", **kwargs)
        full_ds = ds["train"]
        parquet_files = []

    if parquet_files:
        # Shuffle the files list so we stream from a diverse set of shards
        random.Random(seed).shuffle(parquet_files)

        collected_samples = []
        # We need total_train + total_val + total_test. Let's make sure we collect enough.
        target_pool_size = (total_train + total_val + total_test) * 1.5
        samples_per_shard = max(50, int(target_pool_size / len(parquet_files)) + 10)

        print(f"   Found {len(parquet_files)} parquet shards. Collecting up to {samples_per_shard} samples per shard...")

        from tqdm.auto import tqdm
        
        # Read from each parquet file in streaming mode
        for file in tqdm(parquet_files, desc="Streaming Zitacron shards"):
            try:
                ds_shard = load_dataset(
                    "Zitacron/real-vs-ai-corpus",
                    data_files=file,
                    split="train",
                    streaming=True,
                    token=hf_token
                )
                count = 0
                for row in ds_shard:
                    item = {
                        "image": row["image"],
                        "label": row["label"],
                        "source_dataset": row["source_dataset"]
                    }
                    collected_samples.append(item)
                    count += 1
                    if count >= samples_per_shard:
                        break
            except Exception as e:
                # If a shard fails to load, skip it
                continue

        print(f"   Collected {len(collected_samples)} raw samples from streaming.")
        full_ds = Dataset.from_list(collected_samples)

    # Keep only relevant columns
    keep_cols = {"image", "label", "source_dataset"}
    drop_cols = [c for c in full_ds.column_names if c not in keep_cols]
    if drop_cols:
        full_ds = full_ds.remove_columns(drop_cols)

    # ── Discover source datasets ──
    # Sample the source_dataset column to get unique values
    source_datasets = sorted(set(full_ds["source_dataset"]))
    n_sources = len(source_datasets)
    print(f"   Found {n_sources} source datasets: {source_datasets}")

    # ── Build indices grouped by label, and then by source_dataset ──
    print("   Building class-balanced group indices…")
    label_group_indices = {0: {}, 1: {}}
    for idx in range(len(full_ds)):
        src = full_ds[idx]["source_dataset"]
        lbl = full_ds[idx]["label"]
        if src not in label_group_indices[lbl]:
            label_group_indices[lbl][src] = []
        label_group_indices[lbl][src].append(idx)

    # Shuffle each group
    rng = np.random.RandomState(seed)
    for lbl in (0, 1):
        for src in label_group_indices[lbl]:
            rng.shuffle(label_group_indices[lbl][src])

    # ── Allocate samples per split enforcing strict class balance ──
    splits_config = {
        "train": total_train,
        "validation": total_val,
        "test": total_test,
    }

    split_indices = {"train": [], "validation": [], "test": []}
    # Track consumed indices: {label: {source: index_offset}}
    group_consumed = {
        0: {src: 0 for src in label_group_indices[0]},
        1: {src: 0 for src in label_group_indices[1]}
    }

    for split_name, split_total in splits_config.items():
        half_total = split_total // 2  # target per class (real vs AI)
        split_collected = []

        for lbl in (0, 1):
            sources_for_lbl = sorted(label_group_indices[lbl].keys())
            n_sources_lbl = len(sources_for_lbl)
            if n_sources_lbl == 0:
                continue

            quota_per_source = half_total // n_sources_lbl
            remainder = half_total - (quota_per_source * n_sources_lbl)

            collected_lbl = []
            deficit = 0
            sources_with_room = []

            # First pass: take up to quota from each source under this label
            for src in sources_for_lbl:
                available = label_group_indices[lbl][src][group_consumed[lbl][src]:]
                take = min(quota_per_source, len(available))
                start = group_consumed[lbl][src]
                collected_lbl.extend(label_group_indices[lbl][src][start:start + take])
                group_consumed[lbl][src] += take
                if take < quota_per_source:
                    deficit += (quota_per_source - take)
                else:
                    sources_with_room.append(src)

            # Second pass: distribute remainder + deficit to sources with room under this label
            extra_needed = remainder + deficit
            if extra_needed > 0 and sources_with_room:
                extra_per = extra_needed // len(sources_with_room)
                extra_rem = extra_needed % len(sources_with_room)
                for i, src in enumerate(sources_with_room):
                    take_extra = extra_per + (1 if i < extra_rem else 0)
                    available = label_group_indices[lbl][src][group_consumed[lbl][src]:]
                    take = min(take_extra, len(available))
                    start = group_consumed[lbl][src]
                    collected_lbl.extend(label_group_indices[lbl][src][start:start + take])
                    group_consumed[lbl][src] += take

            split_collected.extend(collected_lbl)

        rng.shuffle(split_collected)
        split_indices[split_name] = split_collected

    # ── Build Dataset objects ──
    result = {}
    for split_name in ("train", "validation", "test"):
        indices = split_indices[split_name]
        split_ds = full_ds.select(indices)

        # Count label distribution
        labels = split_ds["label"]
        n_real = sum(1 for l in labels if l == 0)
        n_ai = sum(1 for l in labels if l == 1)

        # Count source distribution
        sources = split_ds["source_dataset"]
        source_counts = {}
        for s in sources:
            source_counts[s] = source_counts.get(s, 0) + 1

        print(f"   {split_name}: {len(split_ds)} samples "
              f"(real={n_real}, ai={n_ai})")
        print(f"     Sources: { {k: v for k, v in sorted(source_counts.items())} }")

        result[split_name] = split_ds

    # Save the resulting DatasetDict to disk
    result_dict = DatasetDict(result)
    try:
        os.makedirs("data", exist_ok=True)
        print(f"💾  Saving balanced subset to disk at {local_path}…")
        result_dict.save_to_disk(local_path)
    except Exception as e:
        print(f"⚠️ Warning: failed to save subset to disk: {e}")

    return result_dict


# ── Preprocessing (shared) ────────────────────────────────────────────────────

def make_transform(processor):
    """
    Return a batched transform function compatible with HF Trainer.
    Converts images to RGB and applies the model's image processor.
    """
    def _transform(examples):
        images = examples["image"]
        if not isinstance(images, list):
            images = [images]
        # Handle raw bytes as well as PIL Image objects
        converted = []
        for img in images:
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
            img = img.convert("RGB")
            converted.append(img)
        inputs = processor(converted, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs
    return _transform


def make_augmented_transform(processor):
    """
    Return a batched transform with data augmentation applied *before*
    the model's image processor.  Used for training only.

    Augmentations (all stochastic, applied per-image):
      - Random horizontal flip (50 %)
      - Random JPEG re-compression at quality 50-95 (30 %)
      - Random brightness / contrast jitter ±20 % (30 %)
      - Random rotation ±15° (30 %)
      - Random crop + resize (30 %)
      - Random Gaussian blur (20 %)
      - Random color/saturation jitter (30 %)
    """
    import random
    from PIL import ImageEnhance

    def _augment_single(img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # JPEG compression simulation — injects realistic artifacts
        if random.random() < 0.3:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(50, 95))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        # Brightness / contrast jitter
        if random.random() < 0.3:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        
        # Random rotation (small angles — real photos are rarely perfectly level)
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=(128, 128, 128))
       
        # Random crop + resize (simulates different crops/shares of the same image)
        if random.random() < 0.3:
            w, h = img.size
            crop_ratio = random.uniform(0.8, 1.0)
            new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            img = img.crop((left, top, left + new_w, top + new_h))
            img = img.resize((w, h), Image.BILINEAR)
        
        # Random Gaussian blur (simulates real camera blur/social media compression)
        if random.random() < 0.2:
            from PIL import ImageFilter
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # Color jitter (saturation + hue)
        if random.random() < 0.3:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
        
        return img

    def _transform(examples):
        images = examples["image"]
        if not isinstance(images, list):
            images = [images]
        converted = []
        for img in images:
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
            img = img.convert("RGB")
            img = _augment_single(img)
            converted.append(img)
        inputs = processor(converted, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs

    return _transform


def collate_fn(examples):
    """Custom collator – stacks pixel_values & labels into tensors."""
    pixel_values = []
    for ex in examples:
        pv = ex["pixel_values"]
        if isinstance(pv, list):
            pv = torch.tensor(pv)
        pixel_values.append(pv)
    return {
        "pixel_values": torch.stack(pixel_values),
        "labels": torch.tensor([ex["labels"] for ex in examples]),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def run_training_stage(
    model,
    processor,
    train_ds,
    eval_ds,
    output_dir,
    epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    stage_name="stage",
    fp16=False,
    replay_ds=None,
    replay_ratio=0.25,
    augment=False,
    weight_decay=0.05,
    early_stopping_patience=2,
):
    """
    Train *model* on *train_ds* / *eval_ds* and save to *output_dir*.

    If *replay_ds* is provided, a random subset (sized as *replay_ratio* ×
    len(train_ds)) is drawn from it and concatenated with *train_ds* before
    training.  This implements **experience replay** to mitigate catastrophic
    forgetting when training across sequential stages.

    If *augment* is True, training images receive random flips, colour
    jitter, and JPEG re-compression to improve generalisation.

    Returns (trained_model, trainer).
    """
    print(f"\n{'='*60}")
    print(f"  🚀  {stage_name}")
    print(f"{'='*60}")

    # ── Experience replay: mix previous-stage data into current stage ──
    if replay_ds is not None:
        n_replay = max(1, int(len(train_ds) * replay_ratio))

        # Reset any transform set by a previous training stage so the
        # original column names (e.g. "label") are accessible for filtering.
        replay_ds.reset_format()

        # Balanced replay: 50/50 real/AI to prevent class bias
        real_replay = replay_ds.filter(lambda x: x["label"] == 0).shuffle(seed=42)
        ai_replay = replay_ds.filter(lambda x: x["label"] == 1).shuffle(seed=42)
        n_each = n_replay // 2
        replay_subset = concatenate_datasets([
            real_replay.select(range(min(n_each, len(real_replay)))),
            ai_replay.select(range(min(n_each, len(ai_replay)))),
        ]).shuffle(seed=42)

        # Align schemas — keep only columns common to both datasets
        keep = set(train_ds.column_names) & set(replay_subset.column_names)
        drop_primary = [c for c in train_ds.column_names if c not in keep]
        drop_replay  = [c for c in replay_subset.column_names if c not in keep]
        if drop_primary:
            train_ds = train_ds.remove_columns(drop_primary)
        if drop_replay:
            replay_subset = replay_subset.remove_columns(drop_replay)

        # Cast replay features to match train_ds features (e.g. binary → Image)
        # so that concatenate_datasets doesn't choke on type mismatches.
        replay_subset = replay_subset.cast(train_ds.features)

        train_ds = concatenate_datasets([train_ds, replay_subset]).shuffle(seed=42)
        print(f"  🔁  Experience replay: added {len(replay_subset)} samples "
              f"from previous stage ({replay_ratio:.0%} ratio) [balanced 50/50]")
        print(f"  📊  Combined training set: {len(train_ds)} samples")

    # Apply transforms (with optional augmentation for training)
    if augment:
        train_ds.set_transform(make_augmented_transform(processor))
    else:
        train_ds.set_transform(make_transform(processor))
    eval_ds.set_transform(make_transform(processor))

    args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=fp16,
        push_to_hub=False,
        report_to="none",
        label_smoothing_factor=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=weight_decay,
    )

    # ── Early stopping ──
    callbacks = []
    if early_stopping_patience is not None:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        print(f"  ⏱️  Early stopping enabled (patience={early_stopping_patience})")

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=callbacks or None,
    )

    results = trainer.train()
    trainer.save_model(output_dir)
    trainer.log_metrics("train", results.metrics)
    trainer.save_metrics("train", results.metrics)
    print(f"  ✅  Model saved → {output_dir}")
    return model, trainer


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    processor,
    test_ds,
    output_prefix,
    output_dir="outputs",
    batch_size=8,
    fp16=False,
    label_names=None,
):
    """
    Evaluate *model* on *test_ds* and save JSON metrics + confusion-matrix PNG.

    Parameters
    ----------
    output_prefix : str   e.g. 'zitacron_stage1' — used for file naming.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if label_names is None:
        label_names = ["Real", "AI-Generated"]

    os.makedirs(output_dir, exist_ok=True)

    transform = make_transform(processor)
    test_ds.set_transform(transform)

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, "tmp_eval"),
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,
        fp16=fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # ── metrics ──
    report = classification_report(
        labels, preds, target_names=label_names, output_dict=True
    )
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    json_path = os.path.join(output_dir, f"{output_prefix}_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  📄  Metrics saved → {json_path}")

    # ── confusion matrix plot ──
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(label_names); ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {output_prefix}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    png_path = os.path.join(output_dir, f"{output_prefix}_confusion_matrix.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  🖼️   Confusion matrix saved → {png_path}")

    # ── print summary ──
    print(f"\n  {'─'*40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  {'─'*40}\n")
    print(classification_report(labels, preds, target_names=label_names))
    return metrics
