import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Provide a workspace-local temporary directory on Windows."""
    temp_root = PROJECT_ROOT / ".test-runtime"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / uuid4().hex
    temp_dir.mkdir()

    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
