# 5070_shallow_impl_audit_accpaper · 方案 18

OpenBMI · 2s/hop100 · Acc_paper · **手写 ShallowFBCSP vs braindecode 库模型** 审计旁路。

| 项 | 路径 |
|----|------|
| 方案 | `资料/模型训练/18_旁路_手写vs库_ShallowFBCSP_openbmi_accpaper/方案.md` |
| 数据 | `preprocess_lab/out/openbmi_2s_hop100/`（仅 EEG_MI_train） |
| out | `train_lab/out/5070_shallow_impl_audit_accpaper/` |
| 锚点训练环 | 同 `5060_baselines_openbmi_2s_hop100_accpaper/task_runner` |

## 臂

| 臂 | 脚本 | 模型 |
|----|------|------|
| **L0** | `baseline_shallow_lib.py` | `braindecode.models.ShallowFBCSPNet` |
| **S0** | `baseline_shallow_self.py` | `self_model/shallowfbcsp.py` · `attn=None` |

## 推荐顺序

```powershell
conda activate cyy
cd D:\MI\code\train_lab\src\step\5070_shallow_impl_audit_accpaper

# A/B：结构 + 前向（不写权重）
python compare_shallow_impl.py

# C：fold0 冒烟（默认 batch 256/512 · workers=2）
python baseline_shallow_lib.py  --max-folds 1
python baseline_shallow_self.py --max-folds 1

# C：五折
python baseline_shallow_lib.py
python baseline_shallow_self.py

# 或一键
powershell -File .\run_pair.ps1
powershell -File .\run_pair.ps1 -MaxFolds 1   # 冒烟

# 内存紧 / OOM 时降档
# python baseline_shallow_lib.py --num-workers 0 --batch-train 128 --batch-eval 256
```

## 禁止

- 不得覆盖 `out/5060_baseline_openbmi_2s_hop100_accpaper/` 正式 shallow 权重
- 不得把本旁路数字写入十一模型正式表
