"""标记 OpenBMI Acc_paper 实验记录与权重为 RTX 5090 训练产物。"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "资料" / "模型训练" / "runs"
SRC_OUT = REPO / "code" / "train_lab" / "out" / "5060_baseline_openbmi_2s_hop100_accpaper"
DST_OUT = REPO / "code" / "train_lab" / "out" / "5090_baseline_openbmi_2s_hop100_accpaper"

DEVICE_LINE = (
    "- 训练设备：**NVIDIA RTX 5090**（32GB · sm_120 · conda `cyy` · PyTorch 2.11+cu128）"
)
DEVICE_JSON = {
    "gpu": "NVIDIA RTX 5090",
    "vram_gb": 32,
    "compute_capability": "sm_120",
    "conda_env": "cyy",
    "pytorch": "2.11.0+cu128",
    "marked_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "note": "OpenBMI 2s/hop100 Acc_paper 五折训练权重",
}


def _patch_md(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "RTX 5090" in text:
        return
    text = text.replace(
        "5060_baseline_openbmi_2s_hop100_accpaper",
        "5090_baseline_openbmi_2s_hop100_accpaper",
    )
    if DEVICE_LINE not in text:
        text = re.sub(
            r"(- device：`[^`]+`)\n",
            rf"\1\n{DEVICE_LINE}\n",
            text,
            count=1,
        )
    path.write_text(text, encoding="utf-8")


def _copy_weights() -> list[str]:
    lines: list[str] = []
    if not SRC_OUT.is_dir():
        print(f"skip weights: {SRC_OUT} missing")
        return lines
    if DST_OUT.exists():
        shutil.rmtree(DST_OUT)
    shutil.copytree(SRC_OUT, DST_OUT)
    (DST_OUT / "DEVICE.json").write_text(
        json.dumps(DEVICE_JSON, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for run_dir in sorted(DST_OUT.rglob("run_*")):
        if not run_dir.is_dir():
            continue
        (run_dir / "DEVICE.json").write_text(
            json.dumps({**DEVICE_JSON, "run_dir": run_dir.name}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        rel = run_dir.relative_to(REPO).as_posix()
        model = run_dir.parent.parent.name.replace("_openbmi_2s_hop100_balbatch_accpaper", "")
        lines.append(f"| {model} | `{rel}` |")
    return lines


def main() -> None:
    for md in sorted(RUNS.glob("*openbmi*/*五折实验记录.md")):
        _patch_md(md)
        print("md", md.name)

    latest = REPO / "资料" / "模型训练" / "五折实验记录_最新.md"
    if latest.is_file():
        _patch_md(latest)

    weight_rows = _copy_weights()
    manifest = REPO / "资料" / "模型训练" / "5090_openbmi_accpaper_实验与权重清单.md"
    manifest.write_text(
        "\n".join(
            [
                "# RTX 5090 · OpenBMI Acc_paper 实验与权重清单",
                "",
                f"- 标记时间：`{DEVICE_JSON['marked_at']}`",
                f"- {DEVICE_LINE.lstrip('- ')}",
                "- 代码包：`code/train_lab/src/step/5090_baselines_openbmi_2s_hop100_accpaper/`",
                "- 权重根目录：`code/train_lab/out/5090_baseline_openbmi_2s_hop100_accpaper/`",
                "- 实验记录：`资料/模型训练/runs/*openbmi*`",
                "",
                "## 权重 run 目录",
                "",
                "| 模型 | run 路径 |",
                "|------|----------|",
                *weight_rows,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("manifest", manifest)


if __name__ == "__main__":
    main()
