"""Re-export perf_loader from official package."""
from _official_load import load_official

_m = load_official("perf_loader")
globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith("_")})
