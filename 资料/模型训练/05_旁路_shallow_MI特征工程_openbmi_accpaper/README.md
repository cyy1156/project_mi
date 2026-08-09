# 05 · Shallow · MI 特征工程旁路（OpenBMI Acc_paper）

> 旁路深研 · **非正式夺冠表** · 2026-08-08  
> 正式十一模型结果仍以 [`../04_5060_旁路_2s滑窗100ms_openbmi_accpaper/`](../04_5060_旁路_2s滑窗100ms_openbmi_accpaper/) 为准。

| 文档 | 说明 |
|------|------|
| [`方案.md`](./方案.md) | 协议冻结 · A0–A3 消融 · 决策门槛 |
| 特征依据 | [`../../../find_best_trail/`](../../../find_best_trail/) |
| 训练代码 | `code/train_lab/src/step/5060_shallow_mi_feat_openbmi_accpaper/` |
| 五折记录 | `资料/模型训练/runs/5060_shallow_mi_feat/` |
| 权重 out | `code/train_lab/out/5060_shallow_mi_feat_openbmi_accpaper/` |

### 与正式臂差异

| 项 | 正式 04_5060 shallow | 本旁路 |
|----|----------------------|--------|
| 模型 | ShallowFBCSPNet | 同 |
| 协议 | Acc_paper · balbatch · hop100 | 同 |
| 输入 | 8 导 raw | A1+ 增加差模/包络等 |
| 结果表 | 正式实验结果 | 独立目录，不覆盖正式 |

### 怎么跑

```bash
cd code/train_lab/src/step/5060_shallow_mi_feat_openbmi_accpaper
python baseline_shallow_a0.py --max-folds 1          # 冒烟
python baseline_shallow_a1.py                        # 偏侧通道全五折
python export_trial_quality.py                       # A2 依赖
python baseline_shallow_a2.py
python baseline_shallow_a3.py
```

抽检复现：`python baseline_shallow_a1.py --repro --max-folds 1`
