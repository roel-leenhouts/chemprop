from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR / "model_core"

__path__ = [str(_PKG_DIR)] if _PKG_DIR.is_dir() else [str(_THIS_DIR)]
