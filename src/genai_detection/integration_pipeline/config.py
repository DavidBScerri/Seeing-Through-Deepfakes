"""
Live configuration for the integration pipeline (web app / demo).

One place for the fusion weights, accuracy scalars and decision thresholds
so the entry points stop drifting apart (GAPS.md #2).

Canonical values, confirmed by David on 2026-07-19:
  - fused decision threshold = 0.55 (the 0.25 previously in app.py was a
    temporary testing value from when the visual classifier was mispaired
    with its base model; the report's §3.4 prints 0.5 — report 0.55 in the
    thesis write-up)

Canonical visual backbone, changed by David on 2026-08-12:
  - visual model = OwensLab/commfor-model-384 (official Community Forensics
    weights, Park & Owens arXiv:2411.04125), used out of the box — no
    fine-tuning delta. Replaces the run_01_stage2A fine-tuned ViT on
    dima806/ai_vs_real_image_detection, which is now a documented
    alternative/comparison model, not the canonical one.
    Reason: generalisation to unseen generators, not in-distribution fit —
    see run_07_visual_backend_comparison_eval_results.json. On the
    21-generator external Community Forensics eval set, commfor-384 scores
    pooled AUC 0.994 / mAcc 0.931 vs the fine-tuned ViT's 0.761 / 0.676. The
    fine-tuned ViT only wins on the in-distribution combined_test split
    (AUC 0.921 vs 0.822), which is not representative of "any AI generator,
    long term" — the report's own §-numbers for the fine-tuned model still
    stand as historical results for that model, unaffected by this switch.
  - visual_accuracy (a_v) = 0.9311, the commfor-384 mean-per-generator
    accuracy (mAcc, averaged evenly over the 17 scorable generators in the
    same eval run) rather than pooled accuracy, so no single large generator
    dominates the reliability scalar. Replaces the report's Annex II(2)
    Stage-2 figure (0.84), which was the fine-tuned ViT's own accuracy and
    no longer describes the deployed model.

These are the thesis's empirical constants — any further change needs
David's sign-off and must be reported, not silently applied.
"""

# --- Fusion strategy selection -------------------------------------------
FUSION_STRATEGY = "weighted_average"  # "weighted_average" | "conservative_threshold" | "bayesian"

# --- Weighted-average fusion (the thesis formula, report §3.4) -----------
W_META = 0.30            # w_m — importance of the metadata stream
W_VISUAL = 0.70          # w_v — importance of the visual stream
META_ACCURACY = 0.70     # a_m — reliability scalar for metadata
VISUAL_ACCURACY = 0.9311  # a_v — commfor-384 mAcc, run_07 CF eval (David, 2026-08-12)
WA_DECISION_THRESHOLD = 0.55  # canonical (David, 2026-07-19)

# --- Conservative threshold fusion (experimental AND-gate) ---------------
CT_META_THRESHOLD = 0.70
CT_VISUAL_THRESHOLD = 0.65

# --- Bayesian fusion (experimental) --------------------------------------
BAYES_PRIOR = 0.50
BAYES_THRESHOLD = 0.55

# --- Face crop -----------------------------------------------------------
FACE_PADDING = 0.30  # relative padding around the YuNet bbox before reclassification

# --- Web server ----------------------------------------------------------
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # reject larger uploads instead of reading them into memory
