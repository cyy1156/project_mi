# 02_固定窗_bci2a_cue2to4s

BCI2a 全量 · **固定窗 Cue 后 2–4 s** · Val **BalAcc** · **batch balance** · 选型无 RAP。  
定位：**对照实验**（非滑窗主训协议）。主线：[`../00_当前主线_2s滑窗100ms/`](../00_当前主线_2s滑窗100ms/)

- [`方案.md`](./方案.md) — 协议冻结与落点
- [`实验结果汇总_baselines_fixed_2s_bci2a.md`](./实验结果汇总_baselines_fixed_2s_bci2a.md) — **十一模型完整汇总**
- [`对照_固定窗_vs_hop100.md`](./对照_固定窗_vs_hop100.md) — 与 hop100 窗级对照
- 训练代码：`code/train_lab/src/step/baselines_fixed_2s/`

**主结论**：Task / Three 冠军均为 **shallow**（Task BalAcc **0.6539**，Three **0.5349**）；EEGNet Task **0.6395**。

```bash
cd code/train_lab/src/step/baselines_fixed_2s
python smoke_models.py
python run_all.py --continue-on-error
```

对照：[`../01_旁路_2s滑窗100ms/`](../01_旁路_2s滑窗100ms/)
