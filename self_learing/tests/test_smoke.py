import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_package_imports():
    import self_learing

    assert self_learing is not None
