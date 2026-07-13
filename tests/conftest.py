import shutil
import sys
import tempfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_path() -> Path:
    temp_root = PROJECT_ROOT / "artifacts" / "pytest_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(dir=temp_root))

    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
