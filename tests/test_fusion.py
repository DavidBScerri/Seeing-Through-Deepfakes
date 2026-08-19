"""
Tests for the decision-fusion engine
(src/genai_detection/integration_pipeline/fusion.py).

The WeightedAverageFusion expectations are taken from the work placement
report, Annex II (4): w_m=0.3, w_v=0.7, a_m=0.70, a_v=0.84. If these tests
break, the thesis formula changed — that requires David's sign-off, not a
test update.
"""

import pytest

from src.genai_detection.integration_pipeline.fusion import (
    WeightedAverageFusion,
    ConservativeThresholdFusion,
    BayesianFusion,
    extract_visual_ai_probability,
    get_fusion_strategy,
)

REPORT_KWARGS = dict(w_meta=0.3, w_visual=0.7, meta_accuracy=0.70, visual_accuracy=0.84)


class TestWeightedAverageFusion:
    @pytest.mark.parametrize(
        "meta_prob, visual_prob, expected_fused",
        [
            # Rows from report Annex II (4)
            (0.99, 0.8264, 0.8695),  # ai+metadata-place-face
            (0.50, 0.8264, 0.7405),  # ai-metadata-place-face
            (0.99, 0.0625, 0.3066),  # deepfake+metadata+place-face
            (0.53, 0.0625, 0.1855),  # deepfake-metadata+place-face
            (0.50, 0.2168, 0.2913),  # real-metadata+place-face
            (0.35, 0.1147, 0.1766),  # real+metadata-place-face
        ],
    )
    def test_matches_report_annex_values(self, meta_prob, visual_prob, expected_fused):
        strategy = WeightedAverageFusion(**REPORT_KWARGS)
        result = strategy.fuse(metadata_ai_prob=meta_prob, visual_ai_prob=visual_prob)
        assert result.ai_probability == pytest.approx(expected_fused, abs=1e-4)

    def test_crop_max_rule_takes_stronger_face_signal(self):
        strategy = WeightedAverageFusion(**REPORT_KWARGS)
        without_crop = strategy.fuse(metadata_ai_prob=0.5, visual_ai_prob=0.2)
        with_crop = strategy.fuse(metadata_ai_prob=0.5, visual_ai_prob=0.2, cropped_visual_ai_prob=0.9)
        assert with_crop.explanation["visual_ai_prob_effective"] == 0.9
        assert with_crop.ai_probability > without_crop.ai_probability

    def test_crop_max_rule_ignores_weaker_crop(self):
        strategy = WeightedAverageFusion(**REPORT_KWARGS)
        result = strategy.fuse(metadata_ai_prob=0.5, visual_ai_prob=0.8, cropped_visual_ai_prob=0.3)
        assert result.explanation["visual_ai_prob_effective"] == 0.8

    def test_zero_weights_edge_case(self):
        strategy = WeightedAverageFusion(w_meta=0.0, w_visual=0.0)
        result = strategy.fuse(metadata_ai_prob=0.99, visual_ai_prob=0.99)
        assert result.ai_probability == 0.0
        assert result.is_ai is False

    def test_decision_threshold_boundary(self):
        strategy = WeightedAverageFusion(**REPORT_KWARGS, decision_threshold=0.5)
        # Equal inputs of p fuse to exactly p regardless of weights
        assert strategy.fuse(metadata_ai_prob=0.5, visual_ai_prob=0.5).is_ai is True
        assert strategy.fuse(metadata_ai_prob=0.49, visual_ai_prob=0.49).is_ai is False

    def test_explanation_exposes_reasoning(self):
        result = WeightedAverageFusion(**REPORT_KWARGS).fuse(metadata_ai_prob=0.9, visual_ai_prob=0.1)
        for key in ("w_meta_effective", "w_visual_effective", "metadata_ai_prob", "combined_score", "decision_threshold"):
            assert key in result.explanation


class TestConservativeThresholdFusion:
    def test_and_gate_requires_both(self):
        strategy = ConservativeThresholdFusion(meta_threshold=0.70, visual_threshold=0.65)
        assert strategy.fuse(metadata_ai_prob=0.9, visual_ai_prob=0.9).is_ai is True
        assert strategy.fuse(metadata_ai_prob=0.9, visual_ai_prob=0.1).is_ai is False
        assert strategy.fuse(metadata_ai_prob=0.1, visual_ai_prob=0.9).is_ai is False

    def test_disagreement_halves_min_score(self):
        strategy = ConservativeThresholdFusion()
        result = strategy.fuse(metadata_ai_prob=0.9, visual_ai_prob=0.2)
        assert result.ai_probability == pytest.approx(0.1)


class TestBayesianFusion:
    def test_agreeing_high_evidence_amplifies(self):
        result = BayesianFusion(prior=0.5).fuse(metadata_ai_prob=0.8, visual_ai_prob=0.8)
        # 0.64 / (0.64 + 0.04)
        assert result.ai_probability == pytest.approx(0.9412, abs=1e-4)
        assert result.is_ai is True

    def test_neutral_evidence_returns_prior(self):
        result = BayesianFusion(prior=0.5).fuse(metadata_ai_prob=0.5, visual_ai_prob=0.5)
        assert result.ai_probability == pytest.approx(0.5)

    def test_extreme_probabilities_do_not_crash(self):
        result = BayesianFusion().fuse(metadata_ai_prob=1.0, visual_ai_prob=0.0)
        assert 0.0 <= result.ai_probability <= 1.0


class TestExtractVisualAiProbability:
    def test_direct_ai_generated_score(self):
        assert extract_visual_ai_probability({"all_scores": {"AI-generated": 0.83}}) == pytest.approx(0.83)

    def test_fallback_ai_prediction_uses_confidence(self):
        result = {"all_scores": {"FAKE": 0.9, "REAL": 0.1}, "prediction": "AI Generated", "confidence": 0.9}
        assert extract_visual_ai_probability(result) == pytest.approx(0.9)

    def test_fallback_real_prediction_inverts_confidence(self):
        result = {"all_scores": {"FAKE": 0.2, "REAL": 0.8}, "prediction": "Real", "confidence": 0.8}
        assert extract_visual_ai_probability(result) == pytest.approx(0.2)


class TestGetFusionStrategy:
    def test_factory_returns_each_strategy(self):
        assert isinstance(get_fusion_strategy("weighted_average"), WeightedAverageFusion)
        assert isinstance(get_fusion_strategy("conservative_threshold"), ConservativeThresholdFusion)
        assert isinstance(get_fusion_strategy("bayesian"), BayesianFusion)

    def test_factory_rejects_unknown_name(self):
        with pytest.raises(ValueError):
            get_fusion_strategy("majority_vote")
