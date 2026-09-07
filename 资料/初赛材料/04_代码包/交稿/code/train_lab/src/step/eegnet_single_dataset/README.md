# EEGNet 单库 Task 训练

与 `baselines_single/baseline_eegnet.py` 同设定（被试独立五折），当前默认臂：  
- **加权 CE** 固定 `w0=2.2`（静息）/ `w1=1.0`（任务）  
- 早停/选模 **Val Balanced Acc**  
- **无** batch balance / Focal  

**仅 Task**；**不用 merged 混合库**。

共用实现：`train_eegnet_task.py`（与 `run_all_three.py` 同级）。  
输出模型名带 `_wce2p2_balacc` 后缀，与此前 F1 / 仅 BalAcc / balbatch 结果并列留存。

## 子目录

| 目录 | 数据 | 切窗 |
|------|------|------|
| `bci2a_2s/` | `preprocess_lab/out/bci2a_2s` | Cue+2~4s / Cue前2s → 500 |
| `bci2a_4s/` | `preprocess_lab/out/bci2a_4s` | Cue+0~4s / Cue前4s → 1000 |
| `stieger_2s/` | `preprocess_lab/out/stieger_2s` | 反馈末 2s → 500 |
| `stieger_4s/` | `preprocess_lab/out/stieger_4s` | 反馈末 4s → 1000 |

## 生成 4s 数据（首次需要）

```bash
cd code/preprocess_lab
python -m src.datasets.bci2a.batch --cfg config/bci2a_4s.yaml

# Stieger 4s（本机 DATA/stieger）
python -m src.datasets.stieger.batch --glob "D:/cyy/MI/DATA/stieger/S*_Session_*.mat" --out out/stieger_4s --win-sec 4 --rebuild-split
```

## 训练

```bash
cd code/train_lab/src/step/eegnet_single_dataset/bci2a_2s
python run_eegnet_task.py

cd ../bci2a_4s
python run_eegnet_task.py

cd ../stieger_2s
python run_eegnet_task.py

cd ../stieger_4s
python run_eegnet_task.py
```

输出：`train_lab/out/eegnet_single_dataset/<model>/<data>/run_<stamp>/`  
MD：`资料/模型训练/runs/<stamp>_<model>/`  
汇总含 Acc / Spec / Rec / Precision / F1 / BalAcc。
