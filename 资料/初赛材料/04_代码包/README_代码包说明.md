# 源代码包说明（XH-202610 · 随作品提交）

> 本文档用于（a）随源代码压缩包提交给评审的《代码使用说明》，以及（b）打包操作清单。  
> **对外模型名**：离线交卷 **QuadFold-59** · 在线主线 **CausalFuse-8** / 个体适配 **CausalFuse-8FT**（工程内部仍用 S0 / E1f / F5 路径与 JSON id）。

## 1. 基本信息

| 项 | 内容 |
|---|---|
| 编程语言 | Python 3.11（≥3.10）；前端为 HTML/JavaScript（无需构建工具） |
| 主要依赖 | torch、braindecode≥0.8、numpy、scipy、pandas、websockets、scikit-learn、brainflow、pylsl、PyYAML（完整清单见 `requirements.txt`） |
| 硬件环境 | 训练/推理推荐 NVIDIA GPU（实测 RTX 5070）；仅运行在线系统可 CPU |
| 采集硬件（可选） | OpenBCI Cyton 8 通道；无设备时可用内置合成板完整演示软件链路 |
| 操作系统 | Windows 10/11（开发与实测环境）；Linux 理论可用，未实测 |

## 2. 目录结构

```
MI/
├── experiment_game/     在线系统主程序（会话运行时 / 操作台 / 游戏前端 / 在线推理 / 采后微调）
│   ├── config/          冻结配置：v3_session.yaml（范式时序）、ft_policy.json（微调策略）
│   ├── web/             操作台与游戏前端页面
│   ├── experiment/      会话、推理、计分、微调核心逻辑
│   ├── tools/           入口脚本（open_operator 等）
│   └── data/            会话数据与个体模型落盘（运行后生成）
├── code/
│   ├── train_lab/       离线训练与评测（QuadFold-59 / CausalFuse-8 底座预训练、五折、融合）
│   ├── adapt_engine/    CausalFuse-8 融合 / 读出 / 增量微调（e1f.py、readout.py、ft.py）
│   └── preprocess_lab/  公开数据集统一切窗预处理
├── collect_data/        采集软件（LSL 连接、滤波推流、控制面板）
├── requirements.txt     统一依赖清单
└── README.md            仓库说明
```

## 3. 环境安装

```bash
# 1) 建议使用 conda
conda create -n cyy python=3.11 -y
conda activate cyy

# 2) 先按本机 CUDA 版本安装 torch（勿直接 pip install torch）
#    https://pytorch.org/get-started/locally/
#    例（CUDA 12.x）：pip install torch --index-url https://download.pytorch.org/whl/cu121
#    例（仅 CPU）：  pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3) 安装其余依赖（在仓库根目录）
pip install -r requirements.txt

# 4) 自检
python -m experiment_game.tools.preflight
```

## 4. 快速开始

### 4.1 在线系统（操作台 + 游戏闭环）

```
双击 experiment_game/open_operator.bat
浏览器打开 http://127.0.0.1:8080/operator.html#setup
Setup 中选择数据来源（Cyton 串口 / synthetic 合成板）→ 开始实验
被试端浏览器自动打开诱导页，按页面提示完成 流程；操作台可暂停(P)/代确认(N)/标记无效(R)
```

- 无硬件时选择 **synthetic 合成板**，可完整演示采集→分类→反馈链路（数据为合成信号，仅用于流程验证）。
- 游戏模式：配置 `game_mode=v3_test`（20 试次连击计分），权重加载被试个人模型；无个人模型时使用通用底座。

### 4.2 离线验证复现（对应《离线性能验证报告》）

```bash
# E1f 各臂混淆矩阵与每类指标（自动对账锚点：W=0.6125 / C=0.6188）
python code/train_lab/src/step/5090_ens_recipe_3s_hop100_accpaper/cm_e1f_arms.py

# E1f 聚合器各臂 replay（classic / 流式 S / 因果 C / 多数票 W）
python code/train_lab/src/step/5090_ens_recipe_3s_hop100_accpaper/replay_classic_vs_causal.py
```

- 需要成员概率 dump 与公开数据集（OpenBMI / BCI IV 2a / Stieger2021）。数据集获取方式与预处理参数见《技术报告》§3.4 与《离线性能验证报告》sheet 07/08。
- 混淆矩阵工具入口即 `cm_e1f_arms.py`（本文件 §4.2 第一条命令），其输出与《离线性能验证报告》sheet 08「原始数据索引」中的混淆矩阵/每类指标一一对应，可逐项对账。
- BCI2a Leave-Next 仿真、Stieger 伪在线脚本的入口与参数登记在对应实验登记表（见报告 sheet 08 原始数据索引）。

## 5. 关键配置（冻结口径，与《技术报告》附录 A 一致）

| 配置文件 | 内容 |
|---|---|
| `experiment_game/config/v3_session.yaml` | 范式时序：Rest 4s → prep 2s → Cue 1s → MI 4s → ITI 3s；在线窗 3s/hop100ms |
| `experiment_game/config/ft_policy.json` | 微调策略：FT 范围 all4、replay 0.10、门控 FAIL 强制晋升 + 告警落盘 |
| `experiment_game/config/protocol.yaml` | EEG 断流看门狗：stall 2.0 s 告警 / abort 5.0 s 中止会话 |
| 通道序 | `Cz, C3, C4, CP3, FC4, FC3, CP4, CPz`（即模型输入通道轴顺序，禁止重排） |
| 统计口径 | 展示 heldout_acc=因果平滑；门控=raw；F5=试次级平滑+多数票（`docs/统计口径方案A_20260831.md`） |

## 6. 数据与模型落盘

```
experiment_game/data/subjects/{被试ID}/
├── sessions/<会话名>/   eeg.csv、events.jsonl、manifest.json、run_config.json 等
├── models/current/      当前个人模型（members/ 四成员权重 + e1f_overlay.json）
└── models/ft_runs/      历次微调记录（含门控与告警）
```

## 7. 打包清单（提交压缩包时）

**包含**：
- `code/`（train_lab、adapt_engine、preprocess_lab 源码）
- `collect_data/` 源码
- `experiment_game/` 源码与配置、`open_operator.bat`、`requirements.txt`（不含 `data/subjects/` 下真实数据，或仅保留 1 个脱敏示例会话）
- `README_代码包说明.md`（本文件）

**排除**：
- `DATA/`（公开数据集约数十 GB，评审可按说明获取）、`experiment_game/data/` 真实被试数据（隐私）
- `.venv/`、`__pycache__/`、`.git/`、`.idea/`、`.pytest_cache/`
- `sim_subjects/`、`_analysis/` 中间产物、`_build/` 材料生成脚本

**隐私合规**：提交前抽查所有目录，确保不含被试个人身份信息（姓名/联系方式）；被试一律以编号（如 syj0828）标识。

**随包附录**（与本 README 同级放置）：
- `附录A_旁路实验索引.md`：主证据链与阴性结案实验的结论/登记表索引，支撑《技术报告》§12 可复现性声明与《离线性能验证报告》sheet 08。
- `附录B_自采数据质控样例.md`：自采协议（OpenBMI-Align v1）参数表 + 单被试脱敏质控勾选样例 + 每人采集后质控清单模板。

## 8. 已知限制

- 训练底座所用的公开数据集体积较大，压缩包不含数据本体；获取与切窗方法已在文档中给全。
- `cm_e1f_arms.py` 等复算脚本依赖概率 dump 产物路径（`code/train_lab/out/...`），若未随包附带，可按登记表中的训练入口重新生成。
- 合成板数据仅用于流程演示，不构成分类性能证据。
