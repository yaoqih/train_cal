from __future__ import annotations

from pathlib import Path


_SRC_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src" / "fzed_shunting"
__path__ = [str(_SRC_PACKAGE_DIR)]
