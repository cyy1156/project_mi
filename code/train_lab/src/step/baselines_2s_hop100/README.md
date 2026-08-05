# baselines_2s_hop100

旁路选型：2 s 滑窗 / 100 ms 步长（`n_times=500`）。配方同 `baselines_1s`（Val BalAcc + batch balance、无 RAP）；**不与**论文 1 s/40 ms 对齐。

每个模型脚本会依次跑：
1. **Task**（静息/任务，二分类）
2. **Three**（空闲/左/右，三分类；**独立重训，不迁移 Task 权重**）

文档：[`资料/模型训练/01_旁路_2s滑窗100ms/方案.md`](../../../../../资料/模型训练/01_旁路_2s滑窗100ms/方案.md)

- 数据：`bci2a_2s_hop100` / `stieger_2s_hop100`（分开训，不合并）
- 模型 11 个：时域五 + bandpower 三 + `*_raw` 三
- 记录：`资料/模型训练/runs/<stamp>_<model>_2s_hop100_balbatch_balacc/`（含 Task/Three 五折明细与完整指标）
- 权重：`out/baseline_2s_hop100/.../task|three/fold*/best_*.pt` + `summary.json` + `final_meta.json`

## 预处理

```bash
cd code/preprocess_lab
python -m src.datasets.bci2a.batch --cfg config/bci2a_2s_hop100.yaml
```

## 训练

```bash
cd code/train_lab/src/step/baselines_2s_hop100
python smoke_models.py
python run_all.py --data bci2a_2s_hop100
# 单模型调试可跳过 Three：
python baseline_shallow.py --data bci2a_2s_hop100 --skip-three
```
