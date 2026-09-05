"""Re-export official helpers under this package name."""
from _official_load import load_official

_m = load_official("md_fold_detail")
globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith("_")})
