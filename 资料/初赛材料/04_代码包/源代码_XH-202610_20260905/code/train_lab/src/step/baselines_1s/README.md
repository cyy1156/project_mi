# baselines_1s

离线 **1 s / 40 ms** 协议下的基线选型（**原结构、不做 RAP**）。

文档（现行）：

- [`资料/模型训练/00_当前主线_1s伪在线/实验操作手册.md`](../../../../../资料/模型训练/00_当前主线_1s伪在线/实验操作手册.md)
- [`资料/模型训练/00_当前主线_1s伪在线/分析_baselines_1s_离线选型.md`](../../../../../资料/模型训练/00_当前主线_1s伪在线/分析_baselines_1s_离线选型.md)

- 数据：`bci2a_1s` / `stieger_1s`（**分开训，不合并**；Stieger 为 DATA 全库）
- 早停：Val **Balanced Accuracy**
- 训练采样：**batch balance**
- 任务：Task-only；无 RAP / OTTA

模型：

| 名称 | 输入 | 说明 |
|------|------|------|
| eegnet / shallow / deep / eegtcnet / conformer | `(B,8,250)` | 时域 CNN |
| dbn / gcbnet / dgcnn | `(B,8,2)` | 1s μ/β bandpower |
| **dbn_raw / gcbnet_raw / dgcnn_raw** | `(B,8,250)` | TemporalEncoder + 图/DBN（来自 `Self_development_model`） |

## 预处理

```bash
cd code/preprocess_lab
python -m src.datasets.bci2a.batch --cfg config/bci2a_1s.yaml
python -m src.datasets.stieger.batch_1s --out out/stieger_1s --no-merge --subjects all
python -m src.datasets.stieger.batch_1s --out out/stieger_1s --merge-only
```

## 训练

```bash
cd code/train_lab/src/step/baselines_1s
python smoke_models.py
python run_all.py --data bci2a_1s
python run_all.py --data stieger_1s --continue-on-error
```

权重：`train_lab/out/baseline_1s/<model>_1s_balbatch_balacc/<data>/run_<stamp>/`  
MD：`资料/模型训练/runs/`（旧整段结果在 `资料/模型训练/归档/03_实验结果_整段2s4s与merged/`）
