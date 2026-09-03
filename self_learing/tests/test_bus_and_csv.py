import  sys
from pathlib import Path

import numpy as np
from sympy.testing.matrices import allclose

from self_learing.channles import CHANNEL_LABELS
from self_learing.csv_recoder import CsvRecoder
from self_learing.live_capture import LiveCapture
from self_learing.ring_buffer import RingBuffer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from self_learing.eeg_bus import EegBus
from self_learing.source_synthetic import SyntheticSource


def test_faout_same_value():
    bus=EegBus()
    a,b=[],[]
    bus.subscribe(lambda t,x:a.append((t,x.copy())))
    bus.subscribe(lambda t,x:b.append((t,x.copy())))
    x=np.arange(8)
    bus.publish(0.0,x)
    assert len(a)==1 and len(b)==1
    assert allclose(a[0][1],b[0][1])

def test_csv_500_rows(tmp_path):
    src=SyntheticSource(seed=0)
    buf=RingBuffer()
    bus=EegBus()
    rec=CsvRecoder(tmp_path/"eeg.csv",CHANNEL_LABELS)
    bus.subscribe(rec.on_sample)
    LiveCapture(src,buf,bus).run_n_sample(500)
    rec.close()
    lines=(tmp_path/"eeg.csv").read_text(encoding="utf-8").strip().split("\n")
    assert lines[0].startswith("t,FC3")
    assert len(lines)==501
    assert len(buf)==500


