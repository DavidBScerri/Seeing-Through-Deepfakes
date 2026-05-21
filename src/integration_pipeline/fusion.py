"""
Decision-Fusion Engine
======================
Combines outputs from the **Metadata Module** and the **Visual Classifier Module**
into a single AI-generated probability using one of several selectable strategies.

Strategies
----------
1. WeightedAverageFusion  — Weighted linear combination with model-accuracy scaling.
2. ConservativeThresholdFusion — Flags AI if **both** modules agree above their
   respective confidence thresholds (AND-gate — avoids false positives from
   metadata's tendency to produce 0.99 scores whenever any AI marker is present).
3. BayesianFusion — Treats each module as an independent evidence source and
   applies Bayes' rule with a configurable prior to produce a posterior.

Usage
-----
    from src.integration_pipeline.fusion import get_fusion_strategy, extract_visual_ai_probability

    strategy = get_fusion_strategy("weighted_average", w_meta=0.3, w_visual=0.7)
    result   = strategy.fuse(metadata_ai_prob=0.92, visual_ai_prob=0.78)
    print(result)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    """Output of every fusion strategy."""
    ai_probability: float           # Combined P(AI) in [0, 1]
    is_ai: bool                     # True if ai_probability >= decision_threshold
    formula_name: str               # Human-readable strategy name
    explanation: dict[str, Any] = field(default_factory=dict)  # Per-strategy breakdown


# ---------------------------------------------------------------------------
# Helper: extract a comparable [0, 1] AI probability from the visual classifier
# ---------------------------------------------------------------------------

def extract_visual_ai_probability(visual_result: dict) -> float:
    """
    Normalises the VisualClassifier.predict() output into a single float
    in [0, 1] representing P(AI-generated).

    The classifier returns::

        {
            "prediction": "AI Generated" | "Real",
            "confidence": float,
            "raw_label": str,
            "all_scores": {"human": float, "AI-generated": float}
        }

    We prefer `all_scores["AI-generated"]` for a direct probability;
    if that key is missing we derive it from the top-level fields.
    """
    all_scores = visual_result.get("all_scores", {})

    # Try the direct score first
    ai_score = all_scores.get("AI-generated")
    if ai_score is not None:
        return float(ai_score)

    # Fallback: interpret confidence relative to predicted class
    prediction = visual_result.get("prediction", "")
    confidence = float(visual_result.get("confidence", 0.5))

    if "ai" in prediction.lower() or "generated" in prediction.lower():
        return confidence
    else:
        return 1.0 - confidence


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class FusionStrategy(ABC):
    """Base class for all decision-fusion strategies."""

    @abstractmethod
    def fuse(self, metadata_ai_prob: float, visual_ai_prob: float) -> FusionResult:
        """
        Combine two AI-generated probabilities into one decision.

        Args:
            metadata_ai_prob: P(AI) from the metadata module (0–1).
            visual_ai_prob:   P(AI) from the visual classifier (0–1).

        Returns:
            FusionResult with the combined probability and decision.
        """


# ---------------------------------------------------------------------------
# 1. Weighted Average Fusion
# ---------------------------------------------------------------------------

class WeightedAverageFusion(FusionStrategy):
    """
    Weighted linear combination of the two module probabilities.

    Formula
    -------
        combined = w_meta × metadata_prob  +  w_visual × visual_prob

    The default weighting (0.3 / 0.7) down-weights metadata because:
    • Metadata can be trivially stripped or manipulated.
    • Placeholder slots (w=0.1 each) are reserved for future modules
      (watermark detection, perceptual hashing) so the weights across
      all modules will eventually total 1.0.

    The visual weight can optionally be scaled by the model's known
    test-set accuracy via `visual_accuracy`, providing a principled
    self-calibration signal.

    Parameters
    ----------
    w_meta : float
        Weight for the metadata module (default 0.3).
    w_visual : float
        Weight for the visual module (default 0.7).
    decision_threshold : float
        Combined score above which the image is classified AI (default 0.55).
    visual_accuracy : float or None
        If provided, the visual weight is scaled by this value
        (e.g. 0.92 for 92 % test accuracy), giving an accuracy-aware
        effective weight of ``w_visual × visual_accuracy``.
        The weights are then re-normalised to sum to 1.
    """

    def __init__(
        self,
        w_meta: float = 0.3,
        w_visual: float = 0.7,
        decision_threshold: float = 0.55,
        visual_accuracy: float | None = None,
    ):
        self.w_meta = w_meta
        self.w_visual = w_visual
        self.decision_threshold = decision_threshold
        self.visual_accuracy = visual_accuracy

    def fuse(self, metadata_ai_prob: float, visual_ai_prob: float) -> FusionResult:
        eff_w_meta = self.w_meta
        eff_w_visual = self.w_visual

        # Scale the visual weight by its known accuracy if provided
        if self.visual_accuracy is not None:
            eff_w_visual = self.w_visual * self.visual_accuracy
            # Re-normalise so the effective weights sum to 1
            total = eff_w_meta + eff_w_visual
            eff_w_meta /= total
            eff_w_visual /= total

        combined = eff_w_meta * metadata_ai_prob + eff_w_visual * visual_ai_prob
        combined = max(0.0, min(1.0, combined))

        return FusionResult(
            ai_probability=round(combined, 4),
            is_ai=combined >= self.decision_threshold,
            formula_name="Weighted Average",
            explanation={
                "w_meta_nominal": self.w_meta,
                "w_visual_nominal": self.w_visual,
                "w_meta_effective": round(eff_w_meta, 4),
                "w_visual_effective": round(eff_w_visual, 4),
                "visual_accuracy": self.visual_accuracy,
                "metadata_ai_prob": round(metadata_ai_prob, 4),
                "visual_ai_prob": round(visual_ai_prob, 4),
                "combined_score": round(combined, 4),
                "decision_threshold": self.decision_threshold,
            },
        )


# ---------------------------------------------------------------------------
# 2. Conservative (AND-gate) Threshold Fusion
# ---------------------------------------------------------------------------

class ConservativeThresholdFusion(FusionStrategy):
    """
    Flags the image as AI-generated only when **both** modules agree
    above their respective thresholds (a logical AND gate).

    Why AND instead of OR?
    ----------------------
    The metadata module scores 0.99 whenever *any* AI-related EXIF tag
    is present — even a single C2PA marker.  An OR gate would therefore
    trigger on virtually every image that carries provenance metadata,
    producing many false positives.  Requiring both modules to agree
    eliminates that noise.

    The combined probability is taken as the *minimum* of the two scores
    when both thresholds are exceeded; otherwise it is the lower of the
    two raw values (reflecting low confidence).

    Parameters
    ----------
    meta_threshold : float
        Metadata module must exceed this to count (default 0.70).
    visual_threshold : float
        Visual classifier must exceed this to count (default 0.65).
    """

    def __init__(
        self,
        meta_threshold: float = 0.70,
        visual_threshold: float = 0.65,
    ):
        self.meta_threshold = meta_threshold
        self.visual_threshold = visual_threshold

    def fuse(self, metadata_ai_prob: float, visual_ai_prob: float) -> FusionResult:
        meta_pass = metadata_ai_prob >= self.meta_threshold
        visual_pass = visual_ai_prob >= self.visual_threshold

        both_pass = meta_pass and visual_pass

        if both_pass:
            combined = min(metadata_ai_prob, visual_ai_prob)
        else:
            combined = min(metadata_ai_prob, visual_ai_prob) * 0.5

        combined = max(0.0, min(1.0, combined))

        return FusionResult(
            ai_probability=round(combined, 4),
            is_ai=both_pass,
            formula_name="Conservative Threshold (AND-gate)",
            explanation={
                "meta_threshold": self.meta_threshold,
                "visual_threshold": self.visual_threshold,
                "metadata_ai_prob": round(metadata_ai_prob, 4),
                "visual_ai_prob": round(visual_ai_prob, 4),
                "meta_exceeds_threshold": meta_pass,
                "visual_exceeds_threshold": visual_pass,
                "both_agree": both_pass,
                "combined_score": round(combined, 4),
            },
        )


# ---------------------------------------------------------------------------
# 3. Bayesian Fusion
# ---------------------------------------------------------------------------

class BayesianFusion(FusionStrategy):
    """
    Bayesian evidence fusion treating each module's output as an
    independent likelihood.

    Intuition (plain English)
    -------------------------
    Imagine you start with a "prior belief" that any random image has,
    say, a 50 % chance of being AI-generated.  Then each module
    provides its own evidence — the metadata score and the visual score.
    Bayes' rule lets us combine these two independent pieces of
    evidence into a single updated ("posterior") probability that
    accounts for both.

    Formula
    -------
    Let:
        p₁ = metadata AI probability
        p₂ = visual AI probability
        π  = prior P(AI) (default 0.5 — no prior bias)

    Likelihood of observing both scores under "AI" hypothesis:
        L_ai  = p₁ × p₂

    Likelihood under "Real" hypothesis:
        L_real = (1 − p₁) × (1 − p₂)

    Posterior P(AI | evidence) via Bayes' rule:
        P(AI | evidence) = (L_ai × π) / (L_ai × π + L_real × (1 − π))

    If both modules are confident the image is AI, the posterior
    shoots close to 1.  If they disagree, the posterior lands near 0.5.
    If both say "real", the posterior drops toward 0.

    Parameters
    ----------
    prior : float
        Prior probability that any image is AI-generated (default 0.5).
    decision_threshold : float
        Posterior above which we classify as AI (default 0.55).
    """

    def __init__(
        self,
        prior: float = 0.5,
        decision_threshold: float = 0.55,
    ):
        self.prior = prior
        self.decision_threshold = decision_threshold

    def fuse(self, metadata_ai_prob: float, visual_ai_prob: float) -> FusionResult:
        # Clamp to avoid division-by-zero / log(0) edge cases
        eps = 1e-9
        p1 = max(eps, min(1 - eps, metadata_ai_prob))
        p2 = max(eps, min(1 - eps, visual_ai_prob))

        likelihood_ai = p1 * p2
        likelihood_real = (1 - p1) * (1 - p2)

        numerator = likelihood_ai * self.prior
        denominator = numerator + likelihood_real * (1 - self.prior)

        posterior = numerator / denominator
        posterior = max(0.0, min(1.0, posterior))

        return FusionResult(
            ai_probability=round(posterior, 4),
            is_ai=posterior >= self.decision_threshold,
            formula_name="Bayesian Fusion",
            explanation={
                "prior": self.prior,
                "metadata_ai_prob": round(metadata_ai_prob, 4),
                "visual_ai_prob": round(visual_ai_prob, 4),
                "likelihood_ai": round(likelihood_ai, 6),
                "likelihood_real": round(likelihood_real, 6),
                "posterior": round(posterior, 4),
                "decision_threshold": self.decision_threshold,
                "intuition": (
                    "Both modules are combined as independent evidence sources. "
                    "A posterior near 1.0 means both strongly agree the image is AI; "
                    "near 0.0 means both agree it's real; near 0.5 means they disagree."
                ),
            },
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AVAILABLE_STRATEGIES: dict[str, type[FusionStrategy]] = {
    "weighted_average": WeightedAverageFusion,
    "conservative_threshold": ConservativeThresholdFusion,
    "bayesian": BayesianFusion,
}


def get_fusion_strategy(name: str, **kwargs) -> FusionStrategy:
    """
    Factory that returns a configured FusionStrategy by name.

    Args:
        name:   One of "weighted_average", "conservative_threshold", "bayesian".
        **kwargs: Forwarded to the strategy's ``__init__``.

    Returns:
        An initialised FusionStrategy instance.

    Raises:
        ValueError: If *name* is not recognised.
    """
    cls = AVAILABLE_STRATEGIES.get(name)
    if cls is None:
        valid = ", ".join(sorted(AVAILABLE_STRATEGIES))
        raise ValueError(
            f"Unknown fusion strategy '{name}'. Choose from: {valid}"
        )
    return cls(**kwargs)
