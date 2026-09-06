from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class TrialTiming:
    rest_s:float
    prep_s:float
    cue_s:float
    imagine_s:float
    iti_s:float

def load_timing(path:Path)->TrialTiming:
    cfg=yaml.safe_load(path.read_text(encoding="utf-8"))
    t=cfg["timing_v3"]
    return TrialTiming(
        rest_s=float(t["inter_trial_rest_s"]),
        prep_s=float(t["prep_s"]),
        cue_s=float(t["cue_s"]),
        imagine_s=float(t["imagine_s"]),
        iti_s=float(t["iti_s"]),
    )