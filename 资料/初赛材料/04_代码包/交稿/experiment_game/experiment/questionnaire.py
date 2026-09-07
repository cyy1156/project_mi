"""采集结束问卷（操作者手动打开，诱导页作答）。

内容：MIQ 改编（三视角 × 左右手）+ 策略 + 诱导有效性 + 质控 + 负荷。
结果落盘到会话目录 99_summary/questionnaire_post_<stamp>.json。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 5 点量表的默认锚点
_ANCHOR_DIFF = ("1 = 非常困难", "5 = 非常容易")
_ANCHOR_AGREE = ("1 = 完全不符合", "5 = 非常符合")
_ANCHOR_TIRED = ("1 = 非常清醒", "5 = 非常疲劳")

POST_FORM_TITLE = "采集结束问卷（约 3 分钟）"

POST_FORM_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "miq_iv_l",
        "group": "运动想象能力（左手）",
        "text": "以第一人称视角（通过你自己的眼睛）清晰想象左手完成抓握的难易程度。",
        "kind": "scale5",
        "anchors": _ANCHOR_DIFF,
    },
    {
        "id": "miq_ev_l",
        "group": "运动想象能力（左手）",
        "text": "以第三人称视角（如同观看自己的录像）清晰想象左手完成抓握的难易程度。",
        "kind": "scale5",
        "anchors": _ANCHOR_DIFF,
    },
    {
        "id": "miq_k_l",
        "group": "运动想象能力（左手）",
        "text": "不实际动手，在心里感受左手抓握时手指收紧、肌肉发力的躯体感觉的难易程度。",
        "kind": "scale5",
        "anchors": _ANCHOR_DIFF,
    },
    {
        "id": "miq_iv_r",
        "group": "运动想象能力（右手）",
        "text": "以第一人称视角（通过你自己的眼睛）清晰想象右手完成抓握的难易程度。",
        "kind": "scale5",
        "anchors": _ANCHOR_DIFF,
    },
    {
        "id": "miq_ev_r",
        "group": "运动想象能力（右手）",
        "text": "以第三人称视角（如同观看自己的录像）清晰想象右手完成抓握的难易程度。",
        "kind": "scale5",
        "anchors": _ANCHOR_DIFF,
    },
    {
        "id": "miq_k_r",
        "group": "运动想象能力（右手）",
        "text": "不实际动手，在心里感受右手抓握时手指收紧、肌肉发力的躯体感觉的难易程度。",
        "kind": "scale5",
        "anchors": _ANCHOR_DIFF,
    },
    {
        "id": "strategy",
        "group": "想象策略",
        "text": "实验过程中你想象动作时，主要依赖哪种方式？",
        "kind": "choice",
        "options": ["身体感觉为主（动觉）", "画面为主（视觉）", "两者混合"],
    },
    {
        "id": "visual_help",
        "group": "诱导有效性",
        "text": "第一人称画面（桌面、物品、双手）对我的想象有帮助。",
        "kind": "scale5",
        "anchors": _ANCHOR_AGREE,
    },
    {
        "id": "lift_help",
        "group": "诱导有效性",
        "text": "中场休息时操作者抬起我的双手对照画面之后，接下来的想象更清晰了（若没有经历过请选 3）。",
        "kind": "scale5",
        "anchors": _ANCHOR_AGREE,
    },
    {
        "id": "involuntary",
        "group": "质控",
        "text": "实验过程中，我不自主地真的动了手。",
        "kind": "choice",
        "options": ["从不", "偶尔", "经常"],
    },
    {
        "id": "fatigue",
        "group": "负荷",
        "text": "当前的疲劳程度。",
        "kind": "scale5",
        "anchors": _ANCHOR_TIRED,
    },
    {
        "id": "effort",
        "group": "负荷",
        "text": "保持注意力集中在想象任务上的费力程度。",
        "kind": "scale5",
        "anchors": _ANCHOR_TIRED,
    },
    {
        "id": "discomfort",
        "group": "负荷",
        "text": "其他不适或想说的话（可空）。",
        "kind": "text",
        "optional": True,
    },
]

_QUESTION_IDS = {q["id"] for q in POST_FORM_QUESTIONS}
_REQUIRED_IDS = {q["id"] for q in POST_FORM_QUESTIONS if not q.get("optional")}
_CHOICES = {
    q["id"]: list(q["options"]) for q in POST_FORM_QUESTIONS if q["kind"] == "choice"
}
_SCALES = {q["id"] for q in POST_FORM_QUESTIONS if q["kind"] == "scale5"}


def post_form_payload() -> Dict[str, Any]:
    """发给诱导页的问卷消息体。"""
    return {
        "type": "questionnaire",
        "form": "post",
        "title": POST_FORM_TITLE,
        "questions": POST_FORM_QUESTIONS,
    }


def validate_post_answers(answers: Any) -> List[str]:
    """返回错误列表；空列表 = 合法。"""
    if not isinstance(answers, dict):
        return ["问卷答案格式错误"]
    errors: List[str] = []
    missing = sorted(_REQUIRED_IDS - set(answers.keys()))
    if missing:
        errors.append(f"未作答：{', '.join(missing)}")
    for qid, val in answers.items():
        if qid not in _QUESTION_IDS:
            continue
        if qid in _SCALES:
            try:
                v = int(val)
                if not 1 <= v <= 5:
                    errors.append(f"{qid} 须为 1–5")
            except (TypeError, ValueError):
                errors.append(f"{qid} 须为 1–5 的整数")
        elif qid in _CHOICES:
            if str(val) not in _CHOICES[qid]:
                errors.append(f"{qid} 选项非法")
        else:
            s = str(val).strip()
            if len(s) > 500:
                errors.append(f"{qid} 过长（>500 字）")
    return errors


def summarize_post_answers(answers: Dict[str, Any]) -> Dict[str, Any]:
    """给操作台的一行摘要。"""
    kin = [answers.get(k) for k in ("miq_k_l", "miq_k_r") if str(answers.get(k, "")).isdigit()]
    return {
        "kinesthetic_mean": (
            round(sum(int(x) for x in kin) / len(kin), 2) if kin else None
        ),
        "involuntary": answers.get("involuntary"),
        "fatigue": answers.get("fatigue"),
        "visual_help": answers.get("visual_help"),
    }


def save_post_result(
    answers: Dict[str, Any],
    *,
    session_root: Path,
    subject_id: str = "",
    session_id: str = "",
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(session_root) / "99_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "form": "post",
        "subject_id": subject_id,
        "session_id": session_id,
        "session_root": str(session_root),
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "answers": answers,
        "summary": summarize_post_answers(answers),
    }
    path = out_dir / f"questionnaire_post_{stamp}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def latest_session_dir(save_root: Path, subject_id: str = "") -> Optional[Path]:
    """save_root 下最新的会话目录（可按 subject 前缀过滤）。"""
    root = Path(save_root)
    if not root.is_dir():
        return None
    pat = re.compile(r"^[A-Za-z0-9_]+_[A-Za-z0-9_]+_\d{8}_\d{6}$")
    dirs = [d for d in root.iterdir() if d.is_dir() and pat.match(d.name)]
    if subject_id:
        dirs = [d for d in dirs if d.name.startswith(f"{subject_id}_")]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.name)
