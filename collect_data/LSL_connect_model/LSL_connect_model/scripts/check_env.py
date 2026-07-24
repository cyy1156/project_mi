"""第 1 课环境检查：在项目根目录执行 python scripts/check_env.py"""


def main() -> None:
    errors = []

    for name in ("brainflow", "numpy", "pylsl"):
        try:
            __import__(name)
            print(f"[OK] import {name}")
        except ImportError as e:
            errors.append(name)
            print(f"[FAIL] import {name}: {e}")

    if errors:
        print("\n请先：pip install -r requirements.txt  （或 requirement.txt）")
        raise SystemExit(1)

    import brainflow
    import numpy
    import pylsl

    print(f"brainflow {getattr(brainflow, '__version__', 'unknown')}")
    print(f"numpy {getattr(numpy, '__version__')}")
    print(f"pylsl {getattr(pylsl, '__version__')}")


if __name__ == "__main__":
    main()
