"""Local smoke for scheme16 5090 — no training, no squeeze materialize."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from braindecode.models import ShallowFBCSPNet

import task_runner as tr  # noqa: F401
from data_paths import resolve_data
from hier_loss import build_criterion
from shared_hparams import OUT_ROOT_TAG, SHARED

data_dir, prefix = resolve_data("openbmi_2s_hop100")
x = data_dir / f"{prefix}_X.npy"
assert x.is_file(), x
X = np.load(x, mmap_mode="r")
assert X.shape[-1] == 500 and len(X) == 340200
from raw_time_openbmi import squeeze_raw_2s_openbmi  # noqa: F401

print("data_dir", data_dir)
print("OUT_ROOT_TAG", OUT_ROOT_TAG)
print("batch", SHARED.batch_train, SHARED.batch_eval, "workers", SHARED.num_workers)
if torch.cuda.is_available():
    print("cuda", torch.cuda.get_device_name(0))
else:
    print("cuda", False)
ShallowFBCSPNet(n_chans=8, n_outputs=3, n_times=500, drop_prob=0.5)
print("criterion", type(build_criterion("H1", 3)).__name__)
print("SMOKE OK")
