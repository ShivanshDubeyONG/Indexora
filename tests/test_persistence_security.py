"""Comprehensive security and sandboxing tests for index persistence.

Verifies:
1. Public / sandboxed persistence operates strictly within the controlled storage root.
2. Path traversal attempts (../, ..\\, nested ../) are rejected with ValueError.
3. Absolute paths (/etc, C:\\) are rejected in sandboxed operations.
4. Invalid index names (empty, whitespace, special characters) are rejected.
5. Symlink escape attempts pointing outside storage root are detected and blocked.
6. Legitimate save_named, load_named, and list_saved roundtrips succeed with checksum validation.
7. Manifest-level path traversal attempts (ADR-0009) remain strictly blocked across deserializers.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from indexora.exact import ExactBlasIndex, ExactFlatIndex
from indexora.lab import (
    DEFAULT_LAB_STORAGE_ROOT,
    LabSession,
    resolve_safe_storage_path,
    sanitize_index_name,
)
from indexora.persistence import load_index, save_index


@pytest.fixture
def sample_session(tmp_path) -> LabSession:
    """Fixture providing a LabSession bound to a temporary sandbox storage root."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((20, 8)).astype(np.float32)
    ids = np.arange(20, dtype=np.int64)
    idx = ExactBlasIndex(database=vectors, ids=ids)
    root = tmp_path / "sandbox_root"
    root.mkdir()
    return LabSession(index=idx, index_name="ExactBlasIndex", d=8, storage_root=root, strict_sandbox=True)


# ==============================================================================
# Identifier Sanitization & Safe Path Resolution Tests
# ==============================================================================


@pytest.mark.parametrize(
    "valid_name",
    [
        "my_index",
        "index-1",
        "hnsw_m8_ef32",
        "INDEX_2026",
        "a",
        "0",
        "session-abc_123",
    ],
)
def test_sanitize_index_name_valid(valid_name):
    """Permitted alphanumeric, hyphen, and underscore identifiers pass sanitization."""
    assert sanitize_index_name(valid_name) == valid_name


@pytest.mark.parametrize(
    "invalid_name,error_match",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("../escape", "Path traversal"),
        ("..\\escape", "Path traversal"),
        ("nested/dir", "Path traversal"),
        ("nested\\dir", "Path traversal"),
        ("drive:index", "Path traversal"),
        ("index name with spaces", "Invalid index name"),
        ("index@name!", "Invalid index name"),
        ("index$var", "Invalid index name"),
        ("a" * 65, "exceeds maximum length"),
    ],
)
def test_sanitize_index_name_invalid_rejected(invalid_name, error_match):
    """Malformed or malicious index identifiers raise ValueError with descriptive reason."""
    with pytest.raises(ValueError, match=error_match):
        sanitize_index_name(invalid_name)


def test_sanitize_index_name_type_error():
    """Non-string inputs raise TypeError."""
    with pytest.raises(TypeError, match="must be a string"):
        sanitize_index_name(123)  # type: ignore


def test_resolve_safe_storage_path_root_containment(tmp_path):
    """Resolved path must be strictly contained within storage_root."""
    root = tmp_path / "storage"
    root.mkdir()

    safe_path = resolve_safe_storage_path("test_index", storage_root=root)
    assert safe_path.is_relative_to(root.resolve())
    assert safe_path.name == "test_index"


@pytest.mark.parametrize(
    "attack_vector",
    [
        "../outside",
        "..\\outside",
        "../../etc/passwd",
        "/etc/shadow",
        "C:\\Windows\\System32",
        "nested/../../escape",
    ],
)
def test_resolve_safe_storage_path_rejects_traversal_and_absolute(attack_vector, tmp_path):
    """Path traversal and absolute path attacks are blocked."""
    root = tmp_path / "storage"
    root.mkdir()

    with pytest.raises(ValueError, match="Path traversal|Invalid index name|escapes storage root"):
        resolve_safe_storage_path(attack_vector, storage_root=root)


def test_resolve_safe_storage_path_symlink_escape(tmp_path):
    """Symlink inside storage root pointing outside is detected and rejected."""
    root = tmp_path / "storage"
    root.mkdir()
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()

    symlink_path = root / "symlink_escape"
    try:
        symlink_path.symlink_to(outside_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not supported in current environment/permissions")

    with pytest.raises(ValueError, match="Symlink escape detected|Path traversal"):
        resolve_safe_storage_path("symlink_escape", storage_root=root)


# ==============================================================================
# LabSession Sandboxed Persistence Tests
# ==============================================================================


def test_lab_session_save_named_and_load_named(sample_session):
    """Valid index names save and restore successfully within the sandbox root."""
    manifest = sample_session.save_named("verified_session")
    assert manifest["format"] == "indexora_index"

    target_dir = sample_session.storage_root / "verified_session"
    assert target_dir.exists()
    assert (target_dir / "manifest.json").exists()

    # List saved
    saved_list = sample_session.list_saved()
    assert "verified_session" in saved_list

    # Restore
    restored = sample_session.load_named("verified_session", verify_checksums=True)
    assert isinstance(restored, ExactBlasIndex)
    assert len(restored) == 20


@pytest.mark.parametrize("attack_name", ["../outside", "../../escape", "bad name with spaces", "/absolute"])
def test_lab_session_save_named_rejects_malicious_names(sample_session, attack_name):
    """LabSession.save_named strictly rejects invalid or traversal names."""
    with pytest.raises(ValueError):
        sample_session.save_named(attack_name)


@pytest.mark.parametrize("attack_name", ["../outside", "../../escape", "bad name with spaces", "/absolute"])
def test_lab_session_load_named_rejects_malicious_names(sample_session, attack_name):
    """LabSession.load_named strictly rejects invalid or traversal names."""
    with pytest.raises(ValueError):
        sample_session.load_named(attack_name)


def test_lab_session_strict_sandbox_blocks_unauthorized_paths(sample_session, tmp_path):
    """When strict_sandbox is enabled, save() and load() reject external paths."""
    external_dir = tmp_path / "external_directory"
    with pytest.raises(ValueError):
        sample_session.save(external_dir)

    with pytest.raises(ValueError):
        sample_session.load(external_dir)


# ==============================================================================
# Manifest-Level Path Traversal Defense (ADR-0009 Continuity)
# ==============================================================================


def test_manifest_array_path_traversal_detection(tmp_path):
    """Manifest referencing array filenames outside index directory triggers ValueError."""
    index_dir = tmp_path / "tampered_index"
    index_dir.mkdir()

    manifest_data = {
        "format": "indexora_index",
        "schema_version": "1.0",
        "index_type": "ExactFlatIndex",
        "d": 8,
        "ntotal": 1,
        "arrays": {
            "../../escape.npy": {"shape": [1, 8], "dtype": "float32", "sha256": "fake"}
        },
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Path traversal detected"):
        load_index(index_dir)
