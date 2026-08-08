from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    viewer_path = project_root / "viewer.py"
    if not viewer_path.exists():
        raise FileNotFoundError(f"Expected viewer at {viewer_path}")

    spec = importlib.util.spec_from_file_location("gopro360_viewer", viewer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {viewer_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    run_func = getattr(module, "run", None)
    if run_func is None:
        raise RuntimeError("viewer.py does not define run()")

    return int(run_func())
