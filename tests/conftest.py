import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Register custom marks — currently just ``integration`` for the
    optional real-library end-to-end provenance test."""
    config.addinivalue_line(
        "markers",
        "integration: end-to-end tests that need external assets or "
        "binaries (opt in via -m integration).",
    )
