"""Point the package at a throwaway project root before anything imports it.

The end-to-end tests build a tiny synthetic Datashop export there, so the
real 19.8M-row data is never needed to run the suite (CI has no data).
"""

import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(tempfile.mkdtemp(prefix="so_ell_test_"))
os.environ["SO_ELL_ROOT"] = str(ROOT)
shutil.copytree(Path(__file__).resolve().parents[1] / "sql", ROOT / "sql")
