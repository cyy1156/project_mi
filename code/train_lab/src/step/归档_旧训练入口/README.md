# 归档：旧训练入口（注册表 / 多模型串跑）

本目录由「单模型单入口」重构时**整文件搬入**，**未改代码内容**。

## 内含

| 项 | 原用途 |
|----|--------|
| `models/` | 模型注册表 + 自研占位 |
| `run_five_baselines.py` | 五基线串跑 |
| `run_baseline_matrix.py` | 多模型矩阵过夜 |
| `run_three_grid_backfill.py` | 三分类网格补跑 |
| `run_overnight_kfold.py` | 依赖 `--model` + registry 的过夜编排 |
| `train_task_kfold.py` / `train_three_kfold.py` | 走 registry 的五折训练壳 |

## 说明

- **不删除**；仅归类，重写时可对照。
- 上级 `step/` 仍保留可复用：`dataset.py` / `metrics.py` / `data_paths.py`。
- 新计划代码目录：`../baselines_single/`（本步不写训练函数）。

若需临时跑旧脚本，把工作目录设到本归档目录，或把本目录加回 `sys.path`（自备依赖）。
