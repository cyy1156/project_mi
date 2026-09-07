"""LSL OpenBCI_Markers 发布。"""

from __future__ import annotations

from typing import Optional

from pylsl import StreamInfo, StreamOutlet, local_clock


MARKER_STREAM_NAME = "OpenBCI_Markers"
MARKER_STREAM_TYPE = "Markers"
MARKER_SOURCE_ID = "experiment_game_markers"


def format_payload(
    event: str,
    *,
    trial_id: Optional[int] = None,
    label: Optional[int] = None,
    phase: Optional[str] = None,
    learn_step: Optional[int] = None,
    object_name: Optional[str] = None,
    scene: Optional[str] = None,
) -> str:
    """
    Marker / events.payload 统一字符串：
    {event}|trial=…|label=…|phase=…|step=…|obj=…|scene=…
    无值字段省略。
    """
    parts = [event]
    if trial_id is not None:
        parts.append(f"trial={trial_id}")
    if label is not None:
        parts.append(f"label={label}")
    if phase is not None:
        parts.append(f"phase={phase}")
    if learn_step is not None:
        parts.append(f"step={learn_step}")
    if object_name:
        parts.append(f"obj={object_name}")
    if scene:
        parts.append(f"scene={scene}")
    return "|".join(parts)


class MarkerPublisher:
    def __init__(self, enabled: bool = True) -> None:
        self._outlet: Optional[StreamOutlet] = None
        if enabled:
            info = StreamInfo(
                MARKER_STREAM_NAME,
                MARKER_STREAM_TYPE,
                1,
                0.0,
                "string",
                MARKER_SOURCE_ID,
            )
            self._outlet = StreamOutlet(info)

    def push(self, payload: str, t_lsl: Optional[float] = None) -> float:
        ts = float(local_clock() if t_lsl is None else t_lsl)
        if self._outlet is not None:
            self._outlet.push_sample([payload], timestamp=ts)
        return ts

    def close(self) -> None:
        self._outlet = None
