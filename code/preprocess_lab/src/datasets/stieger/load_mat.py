"""读取单个 Stieger 会话 .mat → list[StiegerTrial]。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from scipy.io import loadmat

@dataclass
class StiegerTrial:
    """单试次中间表示（尚未选导/滤波）。"""
    subject: str
    session: str
    trial_index: int
    x:np.ndarray
    time_ms: np.ndarray
    fs:float
    ch_names: list[str]
    tasknumber:int
    targetnumber:int
    artifact:int
    triallength:float
    resultind:int

def _parse_subject_session(path:Path)->tuple[str, str]:
    # S12_Session_3.mat → ("S12", "Session_3")
    m=re.match(r"(S\d+)_Session_(\d+)",path.stem,flags=re.IGNORECASE)
    if not m:
        return path.stem,"unknown"
    return f"S{int(m.group(1)[1:])}",f"Session_{int(m.group(2))}"
def _as_str_list(labels_obj) ->list[str]:
    """兼容 MATLAB cellstr / 嵌套 object。"""
    out:list[str] = []
    arr =np.array(labels_obj,dtype=object).reshape(-1)
    for item in arr:
        if isinstance(item,str):
            out.append(item)
        elif isinstance(item,bytes):
            out.append(item.decode("utf-8",errors="ignore"))
        else:
             #常见: array(['C3'], dtype='<U2') 或再包一层
             s=np.asarray(item).reshape(-1)
             out.append(str(s[0]) if len(s) else str(item))
    return out
def _get_field(trialdata_i,name:str,default=None):
    """TrialData 可能是 struct 数组或 mat_struct。"""
    if isinstance(trialdata_i, np.void) or hasattr(trialdata_i, "_fieldnames"):
        try:
            return trialdata_i[name]
        except Exception:
            return getattr(trialdata_i, name, default)
    if isinstance(trialdata_i, dict):
        return trialdata_i.get(name, default)
    return default

def load_stieger_mat(mat_path:Path|str)->list[StiegerTrial]:
    """
       读一个会话文件，返回全部试次（未过滤）。
       注意: scipy 对嵌套 struct 较挑；若字段取不到，用 mat73 / hdf5storage 再包一层。
    """
    mat_path = Path(mat_path)
    subject,session=_parse_subject_session(mat_path)

    raw =loadmat(mat_path,squeeze_me=True,struct_as_record=False)
    if "BCI" not in raw:
        raise KeyError(f"{mat_path.name} 中无 BCI 变量，键={list(raw)}")
    bci = raw["BCI"]

    fs=float(np.asarray(bci.SRATE).reshape(-1)[0])
    _chan = bci.chaninfo
    _fields = list(getattr(_chan, "_fieldnames", []))
    if "label" in _fields:
        _labels = _chan.__dict__["label"]
    elif "labels" in _fields:
        _labels = _chan.__dict__["labels"]
    else:
        raise AttributeError(f"chaninfo 无通道名字段，字段={_fields}")

    if _labels is None:
        raise AttributeError("chaninfo.label 存在但值为 None，请检查 .mat 是否完整")

    ch_names = _as_str_list(_labels)
    print("n_ch_names:", len(ch_names), "head:", ch_names[:5])  # 调试用，确认后可删

    data_cells = np.asarray(bci.data,dtype=object).reshape(-1)
    time_cells = np.asarray(bci.time,dtype=object).reshape(-1)
    trialdata =  np.asarray(bci.TrialData).reshape(-1)

    n =len(data_cells)
    trials:list[StiegerTrial] = []
    for i in np.arange(n):
        x=np.asarray(data_cells[i],dtype=np.float64)

        if x.ndim !=2:
            continue
        if x.shape[0] ==len(ch_names) or x.shape[0]<x.shape[1]:
            x=x.T
        t_ms = np.asarray(time_cells[i], dtype=np.float64).reshape(-1)
        td = trialdata[i]

        def _scalar(name, default=0):
            v = _get_field(td, name, default)
            if v is None:
                return default
            a = np.asarray(v).reshape(-1)
            return a[0] if len(a) else default

        trials.append(
            StiegerTrial(
                subject=subject,
                session=session,
                trial_index=i,
                x=x,
                time_ms=t_ms,
                fs=fs,
                ch_names=ch_names,
                tasknumber=int(_scalar("tasknumber", -1)),
                targetnumber=int(_scalar("targetnumber", -1)),
                artifact=int(_scalar("artifact", 0)),
                triallength=float(_scalar("triallength", 0.0)),
                resultind=int(_scalar("resultind", x.shape[0])),
            )
        )
    return trials























