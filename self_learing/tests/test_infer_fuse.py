from __future__ import annotations

import numpy as np
from torch.cuda import temperature

from self_learing.infer_fake import FakeModel
from self_learing.infer_fuse import fuse_member_probs

from self_learing.inference_service import InferenceService

def test_fuse_handcalc()->None:
    p1=np.array([0.6,0.2,0.2])
    p2=np.array([0.2,0.6,0.2])
    out = fuse_member_probs([p1,p2],temperature=[1.0,1.0],weights=[0.5,0.5])
    assert np.allclose(out,[0.4,0.4,0.2])

def test_fuse_judge()->None:
    model=FakeModel((0.1,0.7,0.2))
    svc=InferenceService(model)

    out =svc.judge_window(np.zeros((8,750)))
    assert out["pred"]==1


