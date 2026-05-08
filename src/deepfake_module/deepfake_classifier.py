import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset
import birder
import numpy as np


class DeepfakeClassifier:
    def __init__(self,
                 face_model_name="prithivMLmods/deepfake-detector-model-v1",
                 scene_model_name="birder-project/rope_vit_reg4_b14_capi-places365",
                 landmark_model_path="./fine_tuned_model",
                 landmark_delta_path=None):
        """
        Initializes the Deepfake Classifier with multiple sub-models:
          - A FaceForensics model for face manipulation detection.
          - A Places365 scene model via the birder library.
          - A fine-tuned DINOv2 model for landmark-based analysis.

        Args:
            face_model_name:      HuggingFace model ID for the face forensics detector.
            scene_model_name:     Birder model name for scene/Places365 classification.
            landmark_model_path:  Local path (or HF model ID) for the fine-tuned DINOv2
                                  landmark model. Defaults to ``./fine_tuned_model``.
                                  Pass ``None`` to skip loading the landmark model.
            landmark_delta_path:  Optional path to a .pt delta file produced by
                                  save_weight_delta(). When provided, the base DINOv2
                                  model is loaded and the stored weight differences are
                                  applied on top, avoiding the need to store a full
                                  model file in the repository.
        """
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        # ── 1. Face Forensics Model ──────────────────────────────────────────
        print(f"Loading Face Forensics model: {face_model_name}")
        try:
            self.face_processor = AutoImageProcessor.from_pretrained(face_model_name)
        except Exception:
            print(
                f"Warning: Could not load processor for {face_model_name}. "
                "Falling back to google/vit-base-patch16-224."
            )
            self.face_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

        self.face_model = AutoModelForImageClassification.from_pretrained(face_model_name).to(self.device)
        self.face_model.eval()

        # ── 2. Scene Classification Model (Places365 via birder) ─────────────
        print(f"Loading Scene model: {scene_model_name}")
        try:
            (self.scene_model, self.scene_info) = birder.load_pretrained_model(scene_model_name, inference=True)
            self.scene_model.to(self.device)
            self.scene_model.eval()
            size = birder.get_size_from_signature(self.scene_info.signature)
            self.scene_transform = birder.classification_transform(size, self.scene_info.rgb_stats)
        except Exception as e:
            print(f"Warning: Could not load {scene_model_name} via birder: {e}. Falling back to google/vit-base-patch16-224.")
            self.scene_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.scene_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to(self.device)
            self.scene_model.eval()
            self.scene_info = None

        # ── 3. Landmark Detection Model (fine-tuned DINOv2) ──────────────────
        self.landmark_model_path = landmark_model_path
        self.landmark_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

        if landmark_delta_path is not None:
            # Delta mode: load base model + apply weight delta
            print(f"Loading DINOv2 base model and applying landmark delta from: {landmark_delta_path}")
            self.landmark_model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-base").to(self.device)
            load_weight_delta(self.landmark_model, landmark_delta_path, device=self.device)
            print("Landmark delta applied successfully.")
            self.landmark_model.eval()
        elif landmark_model_path:
            print(f"Loading fine-tuned Landmark model from: {landmark_model_path}")
            self.landmark_model = AutoModelForImageClassification.from_pretrained(landmark_model_path).to(self.device)
            self.landmark_model.eval()
        else:
            print("Landmark model path not provided. Skipping landmark model loading.")
            self.landmark_model = None

    # ── Inference helpers ────────────────────────────────────────────────────

    def predict_face(self, image):
        """
        Detects whether the face in the image has been manipulated.

        Args:
            image: A PIL Image object.

        Returns:
            dict with keys ``label``, ``confidence``, and ``probs``.
        """
        inputs = self.face_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.face_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        max_prob, idx = torch.max(probs, dim=-1)
        label = self.face_model.config.id2label[idx.item()]
        return {"label": label, "confidence": round(max_prob.item(), 4), "probs": probs[0].tolist()}

    def predict_scene(self, image):
        """
        Classifies the generic scene in the image (Places365).

        Args:
            image: A PIL Image object.

        Returns:
            dict with keys ``label`` and ``confidence``.
        """
        if self.scene_info:
            with torch.no_grad():
                input_tensor = self.scene_transform(image).unsqueeze(0).to(self.device)
                outputs = self.scene_model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=-1)

            max_prob, idx = torch.max(probs, dim=-1)
            label = self.scene_info.labels[idx.item()]
        else:
            inputs = self.scene_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.scene_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            max_prob, idx = torch.max(probs, dim=-1)
            label = self.scene_model.config.id2label[idx.item()]

        return {"label": label, "confidence": round(max_prob.item(), 4)}

    def predict_landmark(self, image):
        """
        Identifies landmarks using the fine-tuned DINOv2 model.

        Args:
            image: A PIL Image object.

        Returns:
            dict with keys ``label`` and ``confidence``
            (or a ``message`` key if the model is not loaded).
        """
        if not self.landmark_model:
            return {"label": "N/A", "confidence": 0.0, "message": "Landmark model not loaded."}

        inputs = self.landmark_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.landmark_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        max_prob, idx = torch.max(probs, dim=-1)
        label = self.landmark_model.config.id2label[idx.item()]
        return {"label": label, "confidence": round(max_prob.item(), 4)}

    def predict(self, image, visual_classifier=None, threshold=0.5):
        """
        Full integrated pipeline:
          1. Optionally run the visual classifier for AI-generated confidence.
          2. If the AI confidence ≥ ``threshold``, run all deepfake sub-models.
          3. Combine results into a final decision.

        Args:
            image:             A PIL Image object.
            visual_classifier: An optional VisualClassifier instance.  When
                               provided, deepfake analysis is only triggered if
                               the AI-generated confidence exceeds ``threshold``.
            threshold:         Confidence threshold above which deepfake analysis
                               is triggered (default: 0.5).

        Returns:
            dict with keys ``visual_classification``, ``deepfake_analysis``,
            and ``final_decision``.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        results = {
            "visual_classification": None,
            "deepfake_analysis": None,
            "final_decision": "Inconclusive"
        }

        # 1. Visual Classifier (optional gate)
        ai_score = threshold  # default: always run deepfake analysis
        if visual_classifier:
            vis_res = visual_classifier.predict(image)
            results["visual_classification"] = vis_res
            ai_score = vis_res["confidence"] if vis_res["prediction"] == "AI Generated" else (1 - vis_res["confidence"])

        # 2. Deepfake sub-models (conditional on AI score)
        if ai_score >= threshold:
            face_res = self.predict_face(image)
            scene_res = self.predict_scene(image)
            landmark_res = self.predict_landmark(image)

            deepfake_score = (
                face_res["confidence"]
                if face_res["label"].lower() == "fake"
                else (1 - face_res["confidence"])
            )

            results["deepfake_analysis"] = {
                "deepfake_confidence": round(deepfake_score, 4),
                "face_analysis": face_res,
                "scene_analysis": scene_res,
                "landmark_analysis": landmark_res
            }
            results["final_decision"] = (
                "Potential Deepfake"
                if deepfake_score > 0.5
                else "Likely AI Generated (Non-Deepfake)"
            )
        else:
            results["final_decision"] = "Likely Real / Low AI Confidence"

        return results


# ---------------------------------------------------------------------------
# Weight-delta helpers
# ---------------------------------------------------------------------------

def save_weight_delta(
    fine_tuned_model,
    base_model_name="facebook/dinov2-base",
    output_path="./fine_tuned_model_delta/weight_delta.pt",
    threshold: float = 1e-9,
):
    """
    Saves weight differences between the fine-tuned model and the base model
    using per-tensor int8 quantisation.

    Encoding:
        For each parameter tensor whose L∞ change exceeds `threshold`:
            scale  = max_abs_diff / 127.0
            stored = round(diff / scale).clamp(-127, 127)  [int8]
        Reconstruction (done in load_weight_delta):
            diff   ≈ stored.float() * scale

    Args:
        fine_tuned_model: The trained model object (in memory after fine_tune_model()).
        base_model_name:  HuggingFace model ID of the base model used for training.
        output_path:      Where to write the .pt delta file.
        threshold:        Tensors whose L∞ change is below this are skipped (pure zeros).
    Returns:
        (output_path, size_mb)
    """
    print(f"Loading base model '{base_model_name}' to compute delta...")
    base_model = AutoModelForImageClassification.from_pretrained(base_model_name)
    base_state = base_model.state_dict()
    ft_state   = fine_tuned_model.state_dict()

    delta     = {}
    unchanged = []
    for key in ft_state:
        ft_param   = ft_state[key].float()
        base_param = base_state[key].float() if key in base_state else torch.zeros_like(ft_param)
        diff       = ft_param - base_param
        max_abs    = diff.abs().max().item()
        if max_abs < threshold:
            unchanged.append(key)
            continue
        # Per-tensor int8 quantisation — 4× smaller than float32, 2× smaller than float16
        scale = max_abs / 127.0
        quant = (diff / scale).round().clamp(-127, 127).to(torch.int8)
        delta[key] = {"q": quant, "s": scale}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(
        {"base_model": base_model_name, "dtype": "int8", "delta": delta},
        output_path,
    )

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Delta saved → '{output_path}'")
    print(f"  Changed parameters : {len(delta)}")
    print(f"  Unchanged (skipped): {len(unchanged)}")
    print(f"  File size          : {size_mb:.2f} MB")
    return output_path, size_mb


def load_weight_delta(model, delta_path, device=None):
    """
    Applies a weight delta (produced by save_weight_delta) to an already-loaded
    base model, modifying it in-place.

    Supports both the int8-quantised format ({"q": int8_tensor, "s": scale})
    and the legacy float16 format (raw half-precision tensor) for backwards
    compatibility.

    Args:
        model:      The base model instance — weights are updated in-place.
        delta_path: Path to the .pt file created by save_weight_delta().
        device:     torch.device to map tensors onto (defaults to CPU).
    """
    checkpoint = torch.load(delta_path, map_location=device or "cpu", weights_only=False)
    delta      = checkpoint["delta"]
    fmt        = checkpoint.get("dtype", "float16")
    state      = model.state_dict()

    for key, payload in delta.items():
        if key not in state:
            continue
        if fmt == "int8" and isinstance(payload, dict):
            # Dequantise: diff ≈ q * scale
            diff = payload["q"].float() * payload["s"]
        else:
            # Legacy float16 delta
            diff = payload.float()
        state[key] = (state[key].float() + diff).to(state[key].dtype)

    model.load_state_dict(state)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and F1 for the DINOv2 landmark
    fine-tuning task. Compatible with the HuggingFace Trainer
    ``compute_metrics`` API.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def fine_tune_model(
    base_model_name="facebook/dinov2-base",
    dataset_name="zguo0525/google-landmarks-v2-mini",
    output_dir="./fine_tuned_model",
    epochs=3,
    batch_size=8,
    learning_rate=5e-5
):
    """
    Fine-tunes DINOv2 on a landmark classification dataset and saves the
    best checkpoint to ``output_dir``.

    Args:
        base_model_name: HuggingFace model ID for the DINOv2 base model.
        dataset_name:    HuggingFace dataset ID for landmark images.
        output_dir:      Directory where the fine-tuned model is saved.
        epochs:          Number of training epochs (default: 3).
        batch_size:      Per-device training/eval batch size (default: 8).
        learning_rate:   AdamW learning rate (default: 5e-5).

    Returns:
        Tuple of (model, processor) after training.
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    num_classes = len(dataset['train'].features['label'].names)
    label_names = dataset['train'].features['label'].names
    print(f"Number of landmark classes: {num_classes}")

    processor = AutoImageProcessor.from_pretrained(base_model_name)

    def transforms(examples):
        inputs = processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs

    # Build train/eval splits
    if "test" in dataset:
        train_ds = dataset["train"]
        eval_ds = dataset["test"]
    else:
        split = dataset["train"].train_test_split(test_size=0.1)
        train_ds = split["train"]
        eval_ds = split["test"]

    print("Applying transformations...")
    train_ds.set_transform(transforms)
    eval_ds.set_transform(transforms)

    model = AutoModelForImageClassification.from_pretrained(
        base_model_name,
        num_labels=num_classes,
        id2label={str(i): name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_steps=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    train_results = trainer.train()

    print("Saving model...")
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    return model, processor
