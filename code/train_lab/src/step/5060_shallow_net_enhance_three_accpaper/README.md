# 5060_shallow_net_enhance_three_accpaper

ShallowFBCSPNet 网络结构增强 · **主攻 Three**（OpenBMI · Acc_paper）。

旁路 09：协议与正式 5060 完全一致（`openbmi_2s_hop100` / z-score / balbatch / Acc_paper），
仅对 ShallowFBCSPNet 做网络结构 / 读出头 / 训练目标消融。

- **正式出数** = Fast 默认（本机 RTX 5060）
- 代码 / out / runs 目录独立，不修改正式表

## 目录结构

```
5060_shallow_net_enhance_three_accpaper/
├── __init__.py
├── _hop100_path.py              # 直接拷贝
├── shared_hparams.py             # 改 OUT_ROOT_TAG 和 runs 路径
├── task_runner.py                # 改 runs 目录名
├── perf_loader.py                # 直接拷贝
├── load_external.py              # 直接拷贝
├── raw_time_openbmi.py           # 直接拷贝
├── trial_metrics.py              # 直接拷贝
├── md_fold_detail.py             # 直接拷贝
├── model_registry.py             # 直接拷贝
├── baseline_shallow_s0.py        # S0：默认结构（复现锚点）
├── shallow_variants.py           # S1~S5 所有变体定义（严格按方案）
├── run_arm.py                    # 统一入口，--arm 参数选择实验臂
├── run_s1_grid.py                # S1 分阶段顺序搜索调度器
└── README.md
```

## 实验变体（严格按方案）

### S0 · 复现锚点（必做）

| 臂 | 说明 | 入口 |
|----|------|------|
| `s0` | braindecode 默认结构 | `--arm s0` |

### S1 · 结构超参网格（顺序搜索 S1a→S1b→S1c→S1d）

| 阶段 | 因子 | 候选 | 入口示例 |
|------|------|------|----------|
| S1a | filter_time_length | 13, 25, 50 | `--arm s1a_t13` |
| S1b | n_filters (time=spat) | 20, 40, 64 | `--arm s1b_f20 --s1a-best-t 13` |
| S1c | pool_time_stride | 10, 15, 25 | `--arm s1c_ps10 --s1a-best-t 13 --s1b-best-f 64` |
| S1d | drop_prob | 0.25, 0.5 | `--arm s1d_d025 --s1a-best-t 13 --s1b-best-f 64 --s1c-best-ps 25` |

### S2 · 多尺度时间核（中风险）

| 臂 | 说明 |
|----|------|
| `s2_ms_concat` | 三分支 TimeConv(13/25/50) 通道拼接，共享 SpatConv→Square→Pool→Log→头 |
| `s2_ms_sum` | 三分支各自 log-power 后相加 → 分类 |

### S3 · 读出头增强（中低风险 · Three 专攻）

| 臂 | 说明 |
|----|------|
| `s3_mlp` | AdaptiveAvgPool→Flatten→MLP(n→64→out)+Dropout |
| `s3_stats` | mean/std/max 拼接→浅 MLP |
| `s3_hier` | Task 二类头 + left/right 头分层推理（多任务训练 + 分层推理已集成 task_runner） |
| `s3_three_only_tune` | 冻结骨干，只训 MLP Three 头 |

### S4 · 训练目标与聚合对齐（中风险）

| 臂 | 说明 |
|----|------|
| `s4_softvote_loss` | 同 trial 窗 logits 均值后 CE（简化实现=标准CE；完整需 trial-level 分组） |
| `s4_focal` | Focal loss (γ 扫 1-2)，`--focal-gamma 1.0` |
| `s4_class_weight` | 逆频类权 CE（与 balbatch 勿双重过猛） |
| `s4_conf_agg` | 评测用置信度加权众数（仅评测不改训练） |

### S5 · 轻量混合骨干（较高风险 · 最后）

| 臂 | 说明 |
|----|------|
| `s5_res_pre` | Square 前加残差时序块（深度可分离 Conv） |
| `s5_dual` | 主路 Shallow-log + 旁路浅 Conv，特征拼接 |
| `s5_se` | SpatConv 后 SE 通道注意力 |

## 运行

```bash
cd code/train_lab/src/step/5060_shallow_net_enhance_three_accpaper

# S0 复现锚点（正式出数）
python run_arm.py --arm s0 --num-workers 0

# S1 分阶段顺序搜索
python run_s1_grid.py --stage s1a --num-workers 0
# → 查看 S1a 结果，选出最优 t（如 13）
python run_s1_grid.py --stage s1b --s1a-best-t 13 --num-workers 0
# → 查看 S1b 结果，选出最优 f（如 64）
python run_s1_grid.py --stage s1c --s1a-best-t 13 --s1b-best-f 64 --num-workers 0
# → 查看 S1c 结果，选出最优 ps（如 25）
python run_s1_grid.py --stage s1d --s1a-best-t 13 --s1b-best-f 64 --s1c-best-ps 25 --num-workers 0

# S2 多尺度
python run_arm.py --arm s2_ms_concat --num-workers 0
python run_arm.py --arm s2_ms_sum --num-workers 0

# S3 读出头
python run_arm.py --arm s3_mlp --num-workers 0
python run_arm.py --arm s3_stats --num-workers 0
python run_arm.py --arm s3_hier --num-workers 0
python run_arm.py --arm s3_three_only_tune --num-workers 0

# S4 训练目标
python run_arm.py --arm s4_focal --focal-gamma 1.0 --num-workers 0
python run_arm.py --arm s4_focal --focal-gamma 2.0 --num-workers 0
python run_arm.py --arm s4_class_weight --num-workers 0
python run_arm.py --arm s4_conf_agg --num-workers 0
python run_arm.py --arm s4_softvote_loss --num-workers 0

# S5 混合骨干
python run_arm.py --arm s5_res_pre --num-workers 0
python run_arm.py --arm s5_dual --num-workers 0
python run_arm.py --arm s5_se --num-workers 0

# 冒烟验证
python run_arm.py --arm s0 --max-folds 1 --max-epochs 2 --patience 2 --num-workers 0
python run_s1_grid.py --stage s1a --smoke --num-workers 0
```

## 阶段门控（必须按序）

```
S0 复现 → S1 超参网格 →（可选进 S2）→ S3 读出头 → S4 目标/聚合 → S5 轻量混合
```

- S1 无任何弱成功 → 仍开 S3；跳过 S2 直至 S3/S4 有信号
- S2-S4 连续两阶段阴性 → 停网络侧
- 单臂冒烟：`--max-folds 1`；正式结论必须 5 折齐

## 数据与协议

- 数据：`preprocess_lab/out/openbmi_2s_hop100/`（与正式 5060 共用）
- 冒烟数据：`preprocess_lab/out/openbmi_2s_fixed_cue2to4_noz/`
- 协议：`2s-hop100ms-balbatch-accpaper-openbmi`
- early_stop：Val Acc_paper
- 评测：Test Acc_paper + BalAcc_maj

## 输出

| 类型 | 路径 |
|------|------|
| 权重 | `train_lab/out/5060_shallow_net_enhance_three_accpaper/` |
| 五折记录 | `资料/模型训练/runs/5060_shallow_net_enhance_three/` |
| 结果登记表 | `资料/模型训练/09_旁路_shallow_网络结构增强_Three_openbmi_accpaper/总结/结果登记表.md` |

## Windows 注意

DataLoader 多 worker 在 Windows 上可能卡死，建议用 `--num-workers 0`。
