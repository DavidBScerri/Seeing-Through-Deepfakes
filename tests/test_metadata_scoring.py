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


class TestC2paMarkerIsNotAnAiClaim:
    """
    Semantic correction from the provenance-validation task:
    ``c2pa``, ``claim_generator``, ``created_software_agent`` and
    ``content credentials`` identify provenance plumbing — the manifest
    system that recorded assertions about the file. They are NOT
    themselves proof that the image was AI-generated. A camera-signed
    image and an AI image both carry these strings.
    """

    def test_c2pa_marker_alone_does_not_set_ai_claim(self):
        flat = flatten_metadata({
            "C2PA": {"Manifest": {"claim_generator": "Adobe_Photoshop/25.0"}},
        })
        features = build_features(flat, binary_hits=[])
        assert features.has_c2pa_marker is True
        assert features.has_ai_claim is False, (
            "A C2PA marker is not itself an AI-generation claim."
        )

    def test_claim_generator_alone_does_not_set_ai_claim(self):
        flat = flatten_metadata({"XMP": {"claim_generator": "Some Camera 1.0"}})
        features = build_features(flat, binary_hits=[])
        assert features.has_ai_claim is False

    def test_created_software_agent_alone_does_not_set_ai_claim(self):
        flat = flatten_metadata({
            "XMP": {"created_software_agent": "Adobe_Photoshop/25.0"},
        })
        features = build_features(flat, binary_hits=[])
        assert features.has_ai_claim is False

    def test_content_credentials_string_alone_does_not_set_ai_claim(self):
        flat = flatten_metadata({"XMP": {"Description": "Content Credentials attached"}})
        features = build_features(flat, binary_hits=[])
        assert features.has_ai_claim is False

    def test_c2pa_marker_adds_zero_to_score(self):
        # Raw marker present, no AI claim: score must not creep upward.
        with_marker = FeatureSet(has_c2pa_marker=True)
        without = FeatureSet(has_c2pa_marker=False)
        s_with, _ = score_features(with_marker)
        s_without, _ = score_features(without)
        assert s_with == pytest.approx(s_without)

    def test_c2pa_marker_still_appears_in_rationale(self):
        # Descriptive-only rationale keeps users informed without
        # nudging the probability.
        features = FeatureSet(has_c2pa_marker=True)
        _, rationale = score_features(features)
        assert any("c2pa" in r.lower() or "content-credentials" in r.lower() for r in rationale)

    def test_binary_marker_only_provenance_hits_do_not_set_ai_claim(self):
        # Binary scan finds `c2pa` bytes (as happens in random compressed
        # image data — GAPS.md #7). That is not an AI claim.
        flat = flatten_metadata({})
        features = build_features(flat, binary_hits=["c2pa", "claim_generator"])
        assert features.has_ai_claim is False
        assert features.unverified_ai_provider_hints == []

    def test_provider_keyword_is_flagged_as_unverified(self):
        # Provider hits still contribute to `has_ai_claim` (the report's
        # existing behaviour), but must be surfaced as unverified so
        # consumers do not treat them as validated claims.
        flat = flatten_metadata({"XMP": {"Software": "Midjourney"}})
        features = build_features(flat, binary_hits=[])
        assert features.has_ai_claim is True
        assert "midjourney" in features.unverified_ai_provider_hints


class TestProvenanceOnlyStaysNeutralEndToEnd:
    """Task 1 regression: raw provenance-plumbing text and binary hits
    must not — via keyword_hits, binary_hits, software_tag_count, or
    suspicious_only_software_tags — indirectly push the metadata score
    away from the neutral 0.50. Descriptive fields keep the value
    visible; scoring does not."""

    def _score(self, raw, binary_hits=None):
        flat = flatten_metadata(raw)
        features = build_features(flat, binary_hits=binary_hits or [])
        score, rationale = score_features(features)
        return features, score, rationale

    def test_bare_c2pa_string_stays_neutral(self):
        features, score, _ = self._score({"XMP": {"Description": "c2pa"}})
        assert features.has_ai_claim is False
        assert score == pytest.approx(0.50)

    def test_claim_generator_alone_stays_neutral(self):
        features, score, _ = self._score({"XMP": {"claim_generator": "Adobe Photoshop 25.0"}})
        assert features.has_ai_claim is False
        assert features.suspicious_only_software_tags is False, (
            "claim_generator is provenance plumbing, not an editor tag — "
            "it must not trigger suspicious_only_software_tags"
        )
        assert score == pytest.approx(0.50)

    def test_created_software_agent_alone_stays_neutral(self):
        features, score, _ = self._score(
            {"XMP": {"created_software_agent": "Adobe Photoshop 25.0"}}
        )
        assert features.has_ai_claim is False
        assert features.suspicious_only_software_tags is False
        assert score == pytest.approx(0.50)

    def test_content_credentials_string_stays_neutral(self):
        features, score, _ = self._score(
            {"XMP": {"Description": "Content Credentials attached"}}
        )
        assert features.has_ai_claim is False
        assert score == pytest.approx(0.50)

    def test_generic_manifest_field_stays_neutral(self):
        features, score, _ = self._score({"C2PA": {"manifest": "some-store-uri"}})
        assert features.has_ai_claim is False
        assert score == pytest.approx(0.50)

    def test_provenance_only_binary_hits_stay_neutral(self):
        features, score, _ = self._score({}, binary_hits=["c2pa", "claim_generator"])
        assert features.has_ai_claim is False
        assert score == pytest.approx(0.50)

    def test_ambiguous_synthetic_word_stays_neutral(self):
        features, score, _ = self._score(
            {"XMP": {"Description": "synthetic aperture radar sample"}}
        )
        assert features.has_ai_claim is False
        assert score == pytest.approx(0.50), (
            "The word 'synthetic' alone is ambiguous — it must not push "
            "the score above 0.50 without explicit AI evidence."
        )

    def test_combination_of_provenance_only_signals_stays_neutral(self):
        features, score, _ = self._score(
            {
                "C2PA": {"Manifest": {"claim_generator": "Adobe Photoshop 25.0"}},
                "XMP": {
                    "created_software_agent": "Adobe Photoshop 25.0",
                    "Description": "Content Credentials attached — synthetic scene",
                },
            },
            binary_hits=["c2pa", "claim_generator"],
        )
        assert features.has_c2pa_marker is True
        assert features.has_ai_claim is False
        assert features.suspicious_only_software_tags is False
        assert score == pytest.approx(0.50)

    def test_explicit_ai_provider_still_scores(self):
        """The negative regressions above must not accidentally silence
        the genuinely explicit signals — a Midjourney credit line must
        still push the score up."""
        features, score, _ = self._score(
            {"XMP": {"CreatorTool": "Midjourney", "Description": "ai generated"}}
        )
        assert features.has_ai_claim is True
        assert score > 0.50

    def test_provenance_hits_still_appear_in_rationale(self):
        _, _, rationale = self._score(
            {"XMP": {"claim_generator": "Adobe Photoshop 25.0"}},
            binary_hits=["c2pa"],
        )
        joined = " ".join(rationale).lower()
        assert "descriptive" in joined or "c2pa" in joined or "provenance" in joined


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
