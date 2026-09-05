# baselines_fixed_2s

BCI2a **固定窗**选型：Cue 后 2–4 s / Rest=Cue 前 2 s（`n_times=500`）。  
配方同旁路：Val BalAcc + batch balance、Task→Three、无 RAP。

文档：[`资料/模型训练/02_固定窗_bci2a_cue2to4s/方案.md`](../../../../../资料/模型训练/02_固定窗_bci2a_cue2to4s/方案.md)

- 数据 tag：`bci2a_2s`（复用 `preprocess_lab/out/bci2a_2s`）
- **仅 BCI2a**；不训 Stieger；不 merged
- 与 `baselines_1s` / `baselines_2s_hop100` / `baselines_single` **隔离**
- 输入：`time`（braindecode）/ `feat`（bandpower）/ `raw`（原始时域图模型）；Deep 默认 Deep4（塌缩则 compat 消融）

## 预处理（已有可跳过）

```bash
cd code/preprocess_lab
python -m src.datasets.bci2a.batch --cfg config/bci2a_2s.yaml
```

核对：`X=(N,1,8,500)`，Task=Cue+2~4s，Rest=Cue前2s。

## 训练

```bash
cd code/train_lab/src/step/baselines_fixed_2s
python smoke_models.py
python run_all.py
python run_all.py --models eegnet,shallow --continue-on-error
# 调试可跳过 Three：
python baseline_eegnet.py --data bci2a_2s --skip-three
```

权重：`train_lab/out/baseline_fixed_2s/<model>_fixed2s_balbatch_balacc/bci2a_2s/run_<stamp>/`  
记录：`资料/模型训练/runs/<stamp>_<model>_fixed2s_balbatch_balacc/`
