"""
Pytest configuration for Meta Quantum Field Agent tests
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests"""
    temp = tempfile.mkdtemp(prefix="mqa_test_")
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def db_path(temp_dir):
    """Create temporary database path"""
    return temp_dir / "test_evolution.db"
