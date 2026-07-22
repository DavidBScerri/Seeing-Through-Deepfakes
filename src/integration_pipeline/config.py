"""
Live configuration for the integration pipeline (web app / demo).

One place for the fusion weights, accuracy scalars and decision thresholds
so the entry points stop drifting apart (GAPS.md #2).

Canonical values, confirmed by David on 2026-07-19:
  - fused decision threshold = 0.55 (the 0.25 previously in app.py was a
    temporary testing value from when the visual classifier was mispaired
    with its base model; the report's §3.4 prints 0.5 — report 0.55 in the
    thesis write-up)
  - base model = dima806/ai_vs_real_image_detection everywhere, delta
    run_01_stage2A (the report's Stage-2 model)
  - visual_accuracy = 0.84 (report Annex II(2) Stage 2 accuracy, 84.06%)

These are the thesis's empirical constants — any further change needs
David's sign-off and must be reported, not silently applied.
"""

# --- Fusion strategy selection -------------------------------------------
FUSION_STRATEGY = "weighted_average"  # "weighted_average" | "conservative_threshold" | "bayesian"

# --- Weighted-average fusion (the thesis formula, report §3.4) -----------
W_META = 0.30            # w_m — importance of the metadata stream
W_VISUAL = 0.70          # w_v — importance of the visual stream
META_ACCURACY = 0.70     # a_m — reliability scalar for metadata
VISUAL_ACCURACY = 0.84   # a_v — reliability scalar for the visual classifier
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
