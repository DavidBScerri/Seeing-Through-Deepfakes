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

import os
import random
import uuid
import collections
import glob
from datasets import load_dataset, Dataset, DatasetDict, Image as HFImage
from tqdm.auto import tqdm

# ── Data Puller Mini-Functions ────────────────────────────────────────────────

def pull_from_repo(repo, config, limit, output_dir, fixed_label=None, label_col="label"):
    """
    Generic mini-function to pull `limit` images from a HuggingFace repository.
    Saves them locally to `output_dir` as JPEGs.
    """
    samples = []
    print(f"📡 Pulling up to {limit} images from {repo}...")
    try:
        if config:
            ds = load_dataset(repo, config, streaming=True, split="train")
        else:
            ds = load_dataset(repo, streaming=True, split="train")
    except Exception as e:
        print(f"   ⚠️ Failed to load {repo}: {e}")
        return samples
        
    for i, row in enumerate(ds):
        if len(samples) >= limit:
            break
            
        if i > 0 and i % 500 == 0:
            print(f"   ... streamed {i} rows ...")
        # Determine label (0 = Real, 1 = AI)
        if fixed_label is not None:
            lbl = fixed_label
        else:
            lbl = row.get(label_col)
            
        if lbl not in [0, 1]:
            continue
            
        # Find the image column
        img_obj = None
        for col in ["image", "img", "photo"]:
            if col in row:
                img_obj = row[col]
                break
                
        if not img_obj:
            continue
            
        # Save to disk
        repo_safe = repo.replace("/", "_")
        src_dir = os.path.join(output_dir, repo_safe)
        os.makedirs(src_dir, exist_ok=True)
        file_path = os.path.join(src_dir, f"{uuid.uuid4().hex}.jpg")
        
        try:
            if hasattr(img_obj, "save"):
                if img_obj.mode in ("RGBA", "P"):
                    img_obj = img_obj.convert("RGB")
                img_obj.save(file_path, format="JPEG", quality=90)
            elif isinstance(img_obj, dict) and "bytes" in img_obj:
                with open(file_path, "wb") as f:
                    f.write(img_obj["bytes"])
            elif isinstance(img_obj, bytes):
                with open(file_path, "wb") as f:
                    f.write(img_obj)
            else:
                continue # Unknown format
                
            samples.append({
                "image": file_path,
                "label": lbl,
                "source_dataset": repo
            })
        except Exception:
            continue
            
    print(f"   ✅ Collected {len(samples)} images.")
    return samples

# Mini-functions for each dataset
def pull_ronantakizawa_webui(limit, out_dir): return pull_from_repo("ronantakizawa/webui", None, limit, out_dir, fixed_label=0)
def pull_derek_thomas_scienceqa(limit, out_dir): return pull_from_repo("derek-thomas/ScienceQA", None, limit, out_dir, fixed_label=0)
def pull_skylenage_deepvision(limit, out_dir): return pull_from_repo("skylenage/DeepVision-103K", "visual_logic", limit, out_dir, fixed_label=0)
def pull_mbzuai_openearthagent(limit, out_dir): return pull_from_repo("MBZUAI/OpenEarthAgent", None, limit, out_dir, fixed_label=0)
def pull_epfl_eceo_coralscapes(limit, out_dir): return pull_from_repo("EPFL-ECEO/coralscapes", None, limit, out_dir, fixed_label=0)
def pull_opendatalab_omnidocbench(limit, out_dir): return pull_from_repo("opendatalab/OmniDocBench", None, limit, out_dir, fixed_label=0)
def pull_sigurdur_isl_finepdfs(limit, out_dir): return pull_from_repo("Sigurdur/isl-finepdfs-images", None, limit, out_dir, fixed_label=0)

def pull_svjack_diffusiondb(limit, out_dir): return pull_from_repo("svjack/diffusiondb_random_10k", None, limit, out_dir, fixed_label=1)
def pull_bitmind_nanobanana(limit, out_dir): return pull_from_repo("bitmind/nano-banana", None, limit, out_dir, fixed_label=1)
def pull_ash12321_nanobananapro(limit, out_dir): return pull_from_repo("ash12321/nano-banana-pro-generated-1k", None, limit, out_dir, fixed_label=1)
def pull_abstractphil_synthetic(limit, out_dir): return pull_from_repo("AbstractPhil/synthetic-characters", "SDXL", limit, out_dir, fixed_label=1)
def pull_lucasfang_fluxreason(limit, out_dir): return pull_from_repo("LucasFang/FLUX-Reason-6M", None, limit, out_dir, fixed_label=1)

def pull_parveshiiii_aivsreal(limit, out_dir): return pull_from_repo("Parveshiiii/AI-vs-Real", None, limit, out_dir, fixed_label=None, label_col="binary_label")
def pull_julienlucas_midjourney(limit, out_dir): return pull_from_repo("julienlucas/midjourney-dalle-sd-nanobananapro-dataset", None, limit, out_dir, fixed_label=None, label_col="label")


# ── Main Builder ──────────────────────────────────────────────────────────────

def build_custom_dataset(
    total_train=10000,
    total_val=2000,
    total_test=2000,
    seed=42,
    hf_token=None,  # kept for signature compatibility
):
    """
    Builds the dataset directly from the source repositories, entirely bypassing the Zitacron mega-corpus.
    Downloads locally to data/custom_dataset to ensure blistering fast speeds.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(base_dir, "..", "..", "data", "custom_subset")
    images_dir = os.path.join(base_dir, "..", "..", "data", "custom_dataset")
    
    if os.path.exists(local_path):
        print(f"📦 Loading cached Custom subset from {local_path}…")
        from datasets import load_from_disk
        return load_from_disk(local_path)

    os.makedirs(images_dir, exist_ok=True)
    
    total_images = total_train + total_val + total_test
    target_real = total_images // 2
    target_ai = total_images - target_real
    
    # We ask each dataset for ~1500 images to ensure we have a massive pool to select our 50/50 mix from.
    limit_per_source = 1500 
    
    all_pullers = [
        pull_ronantakizawa_webui,
        pull_derek_thomas_scienceqa,
        pull_mbzuai_openearthagent,
        pull_epfl_eceo_coralscapes,
        pull_opendatalab_omnidocbench,
        pull_sigurdur_isl_finepdfs,
        pull_svjack_diffusiondb,
        pull_bitmind_nanobanana,
        pull_ash12321_nanobananapro,
        pull_abstractphil_synthetic,
        pull_lucasfang_fluxreason,
        pull_parveshiiii_aivsreal,
        pull_julienlucas_midjourney
    ]
    
    collected_real = []
    collected_ai = []
    
    for puller in all_pullers:
        samples = puller(limit_per_source, images_dir)
        for s in samples:
            if s["label"] == 0:
                collected_real.append(s)
            else:
                collected_ai.append(s)
                
    print(f"\n📊 Total Downloaded: {len(collected_real)} Real, {len(collected_ai)} AI")
    
    if len(collected_real) < target_real or len(collected_ai) < target_ai:
        print(f"⚠️ WARNING: Did not collect enough images! Requested {target_real}/{target_ai}, got {len(collected_real)}/{len(collected_ai)}")
        target_real = min(target_real, len(collected_real))
        target_ai = min(target_ai, len(collected_ai))
        
    rng = random.Random(seed)
    rng.shuffle(collected_real)
    rng.shuffle(collected_ai)
    
    # Slice exact requested totals
    final_real = collected_real[:target_real]
    final_ai = collected_ai[:target_ai]
    
    print("\n🔀 Distributing into Train, Validation, and Test (Exactly 50% Real / 50% AI)...")
    
    val_real_count = total_val // 2
    val_ai_count = total_val - val_real_count
    test_real_count = total_test // 2
    test_ai_count = total_test - test_real_count
    
    val_list = final_real[:val_real_count] + final_ai[:val_ai_count]
    test_list = final_real[val_real_count : val_real_count+test_real_count] + final_ai[val_ai_count : val_ai_count+test_ai_count]
    train_list = final_real[val_real_count+test_real_count:] + final_ai[val_ai_count+test_ai_count:]
    
    rng.shuffle(train_list)
    rng.shuffle(val_list)
    rng.shuffle(test_list)
    
    splits_data = {
        "train": train_list,
        "validation": val_list,
        "test": test_list
    }
    
    result = {}
    for split_name, samples in splits_data.items():
        if not samples: continue
        split_ds = Dataset.from_list(samples)
        try:
            split_ds = split_ds.cast_column("image", HFImage())
        except Exception:
            pass
            
        labels = split_ds["label"]
        n_real = sum(1 for l in labels if l == 0)
        n_ai = sum(1 for l in labels if l == 1)
        
        print(f"   {split_name}: {len(split_ds)} samples (real={n_real}, ai={n_ai})")
        result[split_name] = split_ds

    result_dict = DatasetDict(result)
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"💾 Saving final subset to disk at {local_path}…")
        result_dict.save_to_disk(local_path)
    except Exception as e:
        print(f"⚠️ Warning: failed to save subset to disk: {e}")

    return result_dict


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

BASE_MODEL_ID = "facebook/convnextv2-base-22k-384"


def load_model_from(source="base", device=None):
    """
    Load a model + processor from any source, ready for training or evaluation.

    Parameters
    ----------
    source : str
        One of:
        - ``"base"`` → load the facebook/convnextv2-base-22k-384 HuggingFace model
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
            if isinstance(img, dict) and "bytes" in img:
                img = img["bytes"]
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
            if isinstance(img, dict) and "bytes" in img:
                img = img["bytes"]
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
