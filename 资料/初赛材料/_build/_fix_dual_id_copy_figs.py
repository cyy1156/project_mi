# -*- coding: utf-8 -*-
from pathlib import Path
import shutil

files = [
    Path(r"D:/MI/资料/初赛材料/04_代码包/交稿/code/train_lab/src/step/5070_exp42_confound_collapse_accpaper/io_sessions.py"),
    Path(r"D:/MI/资料/初赛材料/04_代码包/交稿/experiment_game/tools/run_leave_next_e1f_task_ramp.py"),
    Path(r"D:/MI/资料/初赛材料/04_代码包/交稿/experiment_game/tools/run_real_subject_all4_vs_so.py"),
    Path(r"D:/MI/资料/初赛材料/04_代码包/交稿/experiment_game/tools/archive/_audit_real_sessions_6runs.py"),
]
for p in files:
    if not p.exists():
        print("missing", p)
        continue
    t = p.read_text(encoding="utf-8")
    nt = t
    nt = nt.replace('member_id == "xjh0828"', 'member_id in ("xjh0828", "fnz0828")')
    nt = nt.replace('subject_id == "xjh0828"', 'subject_id in ("xjh0828", "fnz0828")')
    nt = nt.replace('sid == "xjh0828"', 'sid in ("xjh0828", "fnz0828")')
    if nt != t:
        p.write_text(nt, encoding="utf-8")
        print("patched", p.name)
    else:
        print("no change", p.name)

src_fig = Path(r"D:/MI/资料/初赛材料/01_技术报告/figures")
dst_fig = Path(r"D:/MI/资料/初赛材料/01_技术报告/交稿/figures")
for name in [
    "图10_真人LeaveNext逐轮窗级.png",
    "图10_真人LeaveNext逐轮窗级.pdf",
    "图11_仿真与真人增益对照.png",
    "图11_仿真与真人增益对照.pdf",
    "图11b_仿真与真人增益对照_各轮最高.png",
    "图11b_仿真与真人增益对照_各轮最高.pdf",
    "fig10_cohort_final_mi.png",
    "fig11_sim_vs_human_gain.png",
    "fig11b_sim_vs_human_peak_gain.png",
]:
    s = src_fig / name
    if s.exists():
        shutil.copy2(s, dst_fig / name)
        print("copied", name)
