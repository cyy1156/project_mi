# -*- coding: utf-8 -*-
from pathlib import Path

p = Path(__file__).resolve().parent / "summary_42.py"
t = p.read_text(encoding="utf-8")
if "A_full" in t and "A-full jackknife" in t:
    print("already patched")
else:
    marker = 'lines.append(f"## {t[\'e\']}")'
    insert = '''
    Afull = arms.get("A_full")
    if Afull:
        lines.append("")
        lines.append("## A-full jackknife")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| R_vol_mean | {_fmt(Afull.get('R_vol_mean'), 4)} |")
        lines.append(f"| frac_pos | {_fmt(Afull.get('frac_positive_R_vol'), 3)} |")
        lines.append(f"| verdict | **{Afull.get('verdict')}** |")
        lines.append(f"| dA5_mean_slope | {_fmt(Afull.get('dA5_mean_slope'), 4)} |")
        lines.append("")

    '''
    if marker not in t:
        raise SystemExit(f"marker not found: {marker!r}")
    t = t.replace(marker, insert + marker, 1)
    # also harden E sim field
    old_e = (
        'f"| collapse frac | {E.get(\'sim_collapse_frac\') or \'TBD\'} | "'
    )
    new_e = (
        'f"| collapse frac | {_fmt(E.get(\'sim_collapse_frac\')) or (E.get(\'sim_collapse_frac\') or \'TBD\')} | "'
    )
    if old_e in t:
        t = t.replace(old_e, new_e, 1)
    p.write_text(t, encoding="utf-8")
    print("patched ok")
