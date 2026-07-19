import pandas as pd
import numpy as np
from pathlib import Path
from src.types import ContinuousEEG, sanity_check

def load_openbci_csv(csv_path: Path, fs: float = 250.0) -> ContinuousEEG:
    df = pd.read_csv(csv_path)
    ch_names = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]
    x = df[ch_names].to_numpy(dtype=np.float64)

    if "event" in df.columns:
        idx = np.where(df["event"].to_numpy() != 0)[0]
        codes = df["event"].to_numpy()[idx]
        events = np.column_stack([idx, codes])
    else:
        events = np.zeros((0, 2), dtype=int)

    return ContinuousEEG(
        subject=csv_path.stem,
        session="session0",
        x=x,
        fs=fs,
        ch_names=ch_names,
        events=events,
        labels=None,
        artifacts=None,
    )