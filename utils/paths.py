"""Repository paths and optional data location overrides."""

import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("CAIR_DATA_DIR", REPO_ROOT / "data"))
CACHE_ROOT = Path(tempfile.gettempdir()) / "cair-matplotlib"
