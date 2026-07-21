from __future__ import annotations

import sys
from pathlib import Path



import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNet
from torch.utils.data import DataLoader

from dataset import TaskHeadDataset
from metrics import binary_task_metrics, format_task_metrics

# 本文件: MI/code/train_lab/src/step/train_task.py
# parents[0]=step → [1]=src → [2]=train_lab → [3]=code
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "preprocess_lab" / "out"/"bci2a"
OUT_DIR = ROOT / "train_lab" / "out"

@torch.no_grad()
def collect_preds(model,loader,device):
    model.eval()
    ys,ps=[],[]
    for x,y in loader:
        logits = model(x.to(device))#前向得到 (B, 2) 的两类分数
        pred =logits.argmax(dim=1).cpu().numpy()#取分数更大的那一类 → 0 或 1；再转到 CPU 成 nump
        ys.append(y.numpy())#本批真实标签（一般已在 CPU）
        ps.append(pred)#本批预测

    """
    某个 batch：logits 为
[[1.2, -0.3],   → argmax → 0（静息） [-0.5, 2.1]]   → argmax → 1（任务）
最终返回例如：
y_true = [0, 1, 1, 0, ...]   # 真实y_pred = [0, 1, 0, 0, ...]   # 模型猜的
    """
    return  np.concatenate(ys),np.concatenate(ps)




def run_epoch(model,loader,criterion,optimizer,device,train:bool):
    model.train(train)
    total_loss,n=0.0,0
    ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            if train:
                #上一批留下的梯度清掉，避免累加到当前 batch。验证阶段不更新参数，所以跳过
                optimizer.zero_grad()
            logits = model(x)  # (B, 2)：每类一个分数
            loss = criterion(logits, y)  # CrossEntropy，和 y_task 比
            if train:
                loss.backward()
                optimizer.step()
            total_loss+=loss.item()*x.size(0)
            n+=x.size(0)
    return total_loss/max(n,1)





def main() ->None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("DATA_DIR:", DATA_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("基线: braindecode.EEGNet | 阶段 A: n_outputs=2 | 不训第二头")

    if not(DATA_DIR/"train_X.npy").exists():
        raise FileNotFoundError(
            f"找不到预处理数据: {DATA_DIR}\n请先运行 preprocess_lab 的 pipeline。"
        )
    train_loader = DataLoader(
        TaskHeadDataset(DATA_DIR,"train"),
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TaskHeadDataset(DATA_DIR,"val"),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    model =EEGNet(
        n_chans=8,
        n_outputs=2,
        n_times=1000,
        F1=8,
        D=2,
        F2=16,
        drop_prob=0.5 ,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)

    best_score=-1.0
    best_state=None
    epochs=50
    for ep in range(1, epochs+1):
        tr_loss=run_epoch(model,train_loader,criterion,optimizer,device,train=True)
        va_loss=run_epoch(model,val_loader,criterion,optimizer,device,train=False)

        y_true,y_pred=collect_preds(model,val_loader,device)
        m = binary_task_metrics(y_true,y_pred)

        print(f"\nep {ep:03d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        print(format_task_metrics("val", m))

        if m["f1"]>best_score:
            best_score=m["f1"]
            best_state={
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(
                {
                    "stage": "A_braindecode_eegnet_task2",
                    "backend": "braindecode.models.EEGNet",
                    "n_outputs": 2,
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": m,
                },
                OUT_DIR / "best_task.pt",
            )
            print("  ↑ saved", OUT_DIR / "best_task.pt")

    if best_state is not None:
        model.load_state_dict(best_state)

    y_va, p_va = collect_preds(model, val_loader, device)
    print(format_task_metrics("val(best)", binary_task_metrics(y_va, p_va)))
    y_tr, p_tr = collect_preds(model, train_loader, device)
    print(format_task_metrics("train(best)", binary_task_metrics(y_tr, p_tr)))
    print("done. best val F1 =", best_score)

if __name__ == "__main__":
    main()

