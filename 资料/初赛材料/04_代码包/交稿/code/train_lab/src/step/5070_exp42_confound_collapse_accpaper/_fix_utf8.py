# -*- coding: utf-8 -*-
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def w(name: str, text: str) -> None:
    (ROOT / name).write_text(text, encoding="utf-8", newline="\n")
    print("wrote", name)


w("__init__.py", '"""Exp42 confound collapse diagnosis (offline)."""\n')

# Verify remaining modules decode as utf-8; rewrite broken ones later.
for p in sorted(ROOT.glob("*.py")):
    if p.name.startswith("_"):
        continue
    try:
        p.read_text(encoding="utf-8")
        print("ok", p.name)
    except UnicodeDecodeError as e:
        print("BAD", p.name, e)
