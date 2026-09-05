# 离线代码复现指南（code/ 目录）

> 覆盖 `preprocess_lab`（数据准备）、`train_lab`（离线训练与评测）、`adapt_engine`（算法本体）三块。
> 环境安装先看包根 `README_代码包说明.md` §3；数据下载与放置看包根 `01_数据集获取说明.md`。

## 1. 三块代码各管什么

| 目录 | 内容 | 与其他层的接口 |
|---|---|---|
| `preprocess_lab/` | 公开数据集统一切窗：滤波（8–30 Hz + 50 Hz 陷波）→ CAR → 按通道重采样 Z-score → 滑窗切样。每个数据集一个 `config/*.yaml` + `src/datasets/<名字>/batch*.py` | 产出到自身 `out/`，供 train_lab 读取 |
| `train_lab/` | 全部离线实验。**一实验一目录**（`src/step/<机器号>_<实验名>/`），每个目录自含入口脚本与工具副本，拷走即可单独复跑 | 产物（权重 `best_*.pt`、概率 dump、指标 CSV）落自身 `out/` |
| `adapt_engine/` | 融合 `e1f.py`、因果读出 `readout.py`、增量微调 `ft.py`——CausalFuse-8 / 8FT 的算法本体 | 被第 2、4 层共同 import，线上线下同一实现 |

## 2. 目录命名约定（读懂 train_lab 的钥匙）

- `src/step/5060_*`、`5070_*`、`5090_*`：前缀数字是执行机器的编号，不是实验序号；同名实验在多台机器上的目录内容一致，取任一即可。
- `_accpaper` 后缀：该实验按「论文口径」跑的正式版本（与消融对照/草稿版本区分）。
- 每个实验目录内的 `README.md`（如有）登记了该实验目的、入口脚本与参数。

## 3. 从零复现三步

### 第 1 步 · 放数据

按 `01_数据集获取说明.md` 下载并放入 `DATA/`。各批处理脚本支持 `--data-root` 覆盖，默认值即文档中的路径。

### 第 2 步 · 预处理切窗（按需选数据集）

```bash
# OpenBMI · 3s / hop100ms 口径（主结果口径）
python code/preprocess_lab/src/datasets/openbmi/batch_3s_hop100.py

# BCI IV 2a · 3s / hop100ms
python code/preprocess_lab/src/datasets/bci2a/batch_3s_hop100.py

# Stieger2021 · 3s / hop100ms
python code/preprocess_lab/src/datasets/stieger/batch_3s_hop100.py
```

每个脚本 `--help` 可查全部参数；切窗参数与 `config/*.yaml` 一一对应（窗口、通道、滤波、标签映射都在里面）。产物写入 `code/preprocess_lab/out/<协议名>/`。

### 第 3 步 · 训练与评测

实验入口都在各自目录内，典型结构是「训练脚本 → 产物落 out/ → 评测/复算脚本读 out/」。关键复算入口（对应《离线性能验证报告》sheet 08）：

```bash
# 四成员集成（CausalFuse-8）各聚合方式的混淆矩阵与每类指标（对账锚点：多数票 0.6125 / 因果平滑 0.6188）
python code/train_lab/src/step/5090_ens_recipe_3s_hop100_accpaper/cm_e1f_arms.py

# 集成聚合器对比重放（classic / 流式 / 因果 / 多数票）
python code/train_lab/src/step/5090_ens_recipe_3s_hop100_accpaper/replay_classic_vs_causal.py
```

以上脚本依赖成员概率 dump（`code/train_lab/out/...`，未随包）。两条路取得：
1. **重训**：按 `02_附录A_实验证据链索引.md` 找到各成员训练入口，跑完五折即生成 dump；
2. **直接索取权重与 dump 包**：见 §4。

## 4. 权重衔接在线系统（重要）

在线系统 `experiment_game/config/e1f_four_member.json` 按相对路径引用四个成员的 `best_task.pt / best_three.pt`（指向 `code/train_lab/out/...`）。这些权重文件**不在压缩包内**，取得方式：

- 途径 A：按 §3 第 3 步重训（需要 GPU 与公开数据集，全量约数天）；
- 途径 B：随提交邮件附**权重网盘链接**（复训产物按相同目录结构解压到 `code/train_lab/out/` 即可，无需改配置）。

## 5. 常见问题

- **找不到 DATA**：脚本从自身位置向上定位仓库根；解压后保持包内目录结构不变即可。
- **显卡不同能否复现**：指标以随机种子固定 + 验证集早停控制；不同 CUDA 版本可能在 ±0.5 pp 内浮动，属正常。
- **只想核对一个数字**：不必全量重训，直接跑 §3 的两个复算脚本（前提 dump 可得）。
