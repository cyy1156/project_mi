import mne
from pathlib import Path
from src.eeg_types import ContinuousEEG, sanity_check

def load_gdf(gdf_path: Path) -> ContinuousEEG:
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    return ContinuousEEG(
        subject=gdf_path.stem[:3],
        session=gdf_path.stem,
        x=raw.get_data().T,          # → (n_times, n_ch)
        fs=float(raw.info["sfreq"]),
        ch_names=list(raw.ch_names),
        events=events[:, [0, 2]],    # sample, event_code
        labels=None,
        artifacts=None,
    )