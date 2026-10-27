"""Smoke tests for Indexora Phase 0."""

from pathlib import Path
import numpy as np
import indexora


def test_import_and_version():
    """Verify package imports correctly and exposes a valid version string."""
    assert hasattr(indexora, "__version__")
    assert isinstance(indexora.__version__, str)
    assert len(indexora.__version__) > 0


def test_source_layout_resolution():
    """Verify package is imported from the expected source directory layout (src/indexora)."""
    pkg_file = Path(indexora.__file__).resolve()
    assert pkg_file.parent.name == "indexora"
    assert pkg_file.parent.parent.name == "src"


def test_numpy_environment():
    """Verify NumPy dependency is available and operates properly in the test environment."""
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert vec.dtype == np.float32
    assert vec.shape == (3,)


def test_all_public_exports_accessible():
    """Verify that every symbol listed in __all__ exists and is accessible from indexora."""
    assert hasattr(indexora, "__all__")
    for name in indexora.__all__:
        assert hasattr(indexora, name), f"Missing public export: {name}"
        obj = getattr(indexora, name)
        assert obj is not None, f"Export {name} is None"
