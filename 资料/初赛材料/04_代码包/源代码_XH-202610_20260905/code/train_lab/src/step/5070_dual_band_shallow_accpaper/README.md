# 5070 · 方案 19 · μ/β 双频带 Shallow + Gate · OpenBMI Acc_paper

> 机位：RTX 5070 · HP 对齐方案 18 `shared_hparams`（batch 256/512）  
> **V1 = 方案 18 S0（引用，不重跑）** · 本包只训 **V2**

## 结构

- 每支：`TemporalConv(20) → Spatial(20) → Square/Pool/Log`
- 默认：`Gate([z_μ,z_β])` → `p_final = α p_μ + (1-α) p_β` + `λ_aux=0.5` 辅助 CE

## 数据（须先预处理）

```powershell
cd D:\MI\code\preprocess_lab
conda activate cyy
# 必须用 cyy；系统/base python 常缺 numpy，会“秒退”
# 冒烟 1 文件
python -m src.datasets.openbmi.batch_2s_hop100 --band mu813 --limit 1 --reset
python -m src.datasets.openbmi.batch_2s_hop100 --band beta1330 --limit 1 --reset
# 全量（约 108 个 mat，耗时长）
python -m src.datasets.openbmi.batch_2s_hop100 --band mu813 --reset
python -m src.datasets.openbmi.batch_2s_hop100 --band beta1330 --reset
```

数据根目录：`D:\MI\DATA\openbmi\sess*_subj*_EEG_MI.mat`（已修默认 glob）。  
输出：`preprocess_lab/out/openbmi_2s_hop100_mu813/` 与 `..._beta1330/`。

## 训练

```powershell
cd D:\MI\code\train_lab\src\step\5070_dual_band_shallow_accpaper
conda activate cyy
python _smoke_local.py
# fold0 冒烟
python train_kfold.py --arm V2 --max-folds 1 --num-workers 0
# 正式五折（默认仅 Three）
python train_kfold.py --arm V2
# 消融
python train_kfold.py --arm V2_a05
python train_kfold.py --arm V2_cat
```

> 主表 **只报 Three Acc_paper**；对照 V1=S0 的 Three **0.5427±0.0243**。需要 Task 时显式加 `--run-task`（附跑）。

out：`code/train_lab/out/5070_dual_band_shallow_accpaper/`  
runs：`资料/模型训练/runs/5070_dual_band_shallow/`  
方案：`资料/模型训练/19_旁路_μβ双带通_双Shallow门控_openbmi_accpaper/方案.md`

## V1 对照数字（S0 · 仅 Three）

Three Acc_paper **0.5427±0.0243** · `run_20260819_162152`
