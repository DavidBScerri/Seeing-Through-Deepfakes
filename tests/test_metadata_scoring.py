"""
Tests for the metadata evidence stream
(src/genai_detection/metadata_module/metadata_extraction.py).

Covers the design rule that missing metadata scores ≈0.5 (uncertain, never
"real"), the report Annex II (1) score expectations, and regression cases
for the substring false positives fixed per GAPS.md #7.
"""

import pytest

from src.genai_detection.metadata_module.metadata_extraction import (
    FeatureSet,
    build_features,
    decide,
    find_keyword_hits,
    flatten_metadata,
    score_features,
)


class TestScoreFeatures:
    def test_empty_metadata_is_neutral_not_real(self):
        score, rationale = score_features(FeatureSet())
        assert score == pytest.approx(0.50)
        assert decide(score, FeatureSet()) == "uncertain"

    def test_ai_claim_rich_metadata_scores_high(self):
        # Mirrors report Annex II (1) ai+metadata rows: clamps at 0.99
        features = FeatureSet(
            has_ai_claim=True,
            has_c2pa=True,
            keyword_hits=["c2pa", "claim_generator", "openai"],
            binary_hits=["c2pa", "claim_generator", "openai"],
            suspicious_only_software_tags=True,
        )
        score, rationale = score_features(features)
        assert score == pytest.approx(0.99)
        assert decide(score, features) == "likely_ai_generated"

    def test_camera_rich_metadata_scores_low(self):
        # Mirrors report Annex II (1) real+metadata rows: 0.35
        features = FeatureSet(
            has_make=True,
            has_model=True,
            has_lens_model=True,
            has_makernote=True,
            suspicious_perfect_timestamp=True,
        )
        score, rationale = score_features(features)
        assert score == pytest.approx(0.35)
        assert decide(score, features) == "likely_camera_origin"

    def test_two_camera_tags_gives_smaller_reduction(self):
        features = FeatureSet(has_make=True, has_model=True)
        score, _ = score_features(features)
        assert score == pytest.approx(0.40)

    def test_score_is_clamped(self):
        features = FeatureSet(
            has_ai_claim=True,
            has_c2pa=True,
            keyword_hits=["a"] * 20,
            binary_hits=["b"] * 20,
            suspicious_only_software_tags=True,
            suspicious_perfect_timestamp=True,
        )
        score, _ = score_features(features)
        assert score <= 0.99

    def test_every_score_change_has_a_rationale(self):
        features = FeatureSet(has_ai_claim=True, has_c2pa=True)
        score, rationale = score_features(features)
        assert score != 0.50
        assert len(rationale) >= 2


class TestBuildFeaturesFalsePositives:
    """Regression cases for GAPS.md #7 substring false positives."""

    def test_influx_does_not_match_flux(self):
        flat = flatten_metadata({"XMP": {"Description": "photo of the InFlux festival"}})
        features = build_features(flat, binary_hits=[])
        assert features.has_ai_claim is False
        assert "flux" not in find_keyword_hits(flat)

    def test_photosynthetic_does_not_match_synthetic(self):
        flat = flatten_metadata({"XMP": {"Description": "photosynthetic algae bloom"}})
        assert "synthetic" not in find_keyword_hits(flat)

    def test_camera_raw_software_is_not_a_camera_claim(self):
        flat = flatten_metadata({"IFD0": {"Software": "Adobe Camera Raw 15.0"}})
        features = build_features(flat, binary_hits=[])
        assert features.has_camera_claim is False

    def test_real_camera_tags_are_a_camera_claim(self):
        flat = flatten_metadata({"IFD0": {"Make": "Canon", "Model": "EOS R5"}})
        features = build_features(flat, binary_hits=[])
        assert features.has_camera_claim is True
        assert features.has_make is True
        assert features.has_model is True

    def test_empty_gps_keys_do_not_count(self):
        flat = flatten_metadata({"GPS": {"GPSLatitude": "", "GPSLongitude": "0 deg 0' 0.00\""}})
        features = build_features(flat, binary_hits=[])
        assert features.has_gps is False

    def test_real_gps_coordinates_count(self):
        flat = flatten_metadata({"GPS": {"GPSLatitude": "35 deg 53' 52.44\" N"}})
        features = build_features(flat, binary_hits=[])
        assert features.has_gps is True

    def test_gps_version_id_alone_does_not_count(self):
        flat = flatten_metadata({"GPS": {"GPSVersionID": "2.3.0.0"}})
        features = build_features(flat, binary_hits=[])
        assert features.has_gps is False

    def test_genuine_ai_keywords_still_detected(self):
        flat = flatten_metadata({"XMP": {"CreatorTool": "Midjourney", "Description": "ai generated artwork"}})
        features = build_features(flat, binary_hits=[])
        assert features.has_ai_claim is True
        hits = find_keyword_hits(flat)
        assert "midjourney" in hits
        assert "ai generated" in hits


class TestFlattenMetadata:
    def test_nested_dicts_and_lists_flatten_lowercased(self):
        flat = flatten_metadata({"A": {"B": "Value"}, "C": ["x", "y"]})
        assert flat == {"a.b": "value", "c[0]": "x", "c[1]": "y"}


class TestDecide:
    def test_thresholds(self):
        assert decide(0.85, FeatureSet()) == "likely_ai_generated"
        assert decide(0.75, FeatureSet(has_ai_claim=True)) == "likely_ai_generated"
        assert decide(0.75, FeatureSet()) == "uncertain"
        assert decide(0.40, FeatureSet()) == "likely_camera_origin"
        assert decide(0.50, FeatureSet()) == "uncertain"
