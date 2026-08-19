"""
Import-level smoke tests for the reorganised package tree.

Ensures every package moved by the src/ reshuffle is importable under its
new name (src.genai_detection.*, src.deepfake_detection), the stub
subpackages that were added for later work exist, and the PROJECT_ROOT
helper points at the repository root.
"""

import importlib

import pytest


NEW_PACKAGES = [
    "src",
    "src.genai_detection",
    "src.genai_detection.metadata_module",
    "src.genai_detection.visual_module",
    "src.genai_detection.integration_pipeline",
    "src.genai_detection.watermark_module",
    "src.genai_detection.hash_module",
    "src.genai_detection.evaluation",
    "src.deepfake_detection",
]

OLD_PACKAGES = [
    "src.metadata_module",
    "src.visual_module",
    "src.integration_pipeline",
    "src.deepfake_module",
]


@pytest.mark.parametrize("modname", NEW_PACKAGES)
def test_new_package_imports(modname):
    mod = importlib.import_module(modname)
    assert mod is not None


@pytest.mark.parametrize("modname", OLD_PACKAGES)
def test_old_package_names_are_gone(modname):
    """A stray old-name package would silently accept old imports again."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(modname)


def test_project_root_points_at_repo():
    from src import PROJECT_ROOT

    # The helper should resolve to the directory containing src/ and web/.
    assert (PROJECT_ROOT / "src").is_dir()
    assert (PROJECT_ROOT / "web" / "index.html").is_file()


def test_public_api_reexports_survived_move():
    """__init__.py re-exports the modules ship should still resolve."""
    from src.genai_detection.metadata_module import analyse_image, AnalysisResult
    from src.genai_detection.integration_pipeline import get_fusion_strategy
    from src.genai_detection.visual_module import CommunityForensicsClassifier
    from src.deepfake_detection import DeepfakeClassifier

    assert callable(analyse_image)
    assert AnalysisResult is not None
    assert callable(get_fusion_strategy)
    assert CommunityForensicsClassifier is not None
    assert DeepfakeClassifier is not None
