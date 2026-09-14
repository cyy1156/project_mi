# project_mi — 运动想象脑机接口（MI-BCI）研究项目

> 面向少样本个性化的智能运动想象脑机交互系统：从**自采实验 → 预处理 → 模型训练 → 离线验证**的完整工程链路。
> 仓库：[cyy1156/project_mi](https://github.com/cyy1156/project_mi)

---

## 这个仓库里有什么（以及没有什么）

| | 内容 |
|---|---|
| ✅ **有** | 全部源码、配置、预处理/训练流水线、实验诱导程序、设计文档与实验材料（`资料/`） |
| ❌ **没有** | 原始采集数据（约 324MB）与训练输出/权重/prob dump（约 490MB） |

后两类是**可复现的中间产物**，体量约占仓库 90%，故不入库以保持仓库轻量。它们仍完整保留在本地，获取/重建方式见下方「数据与产物」一节。

## 目录结构

| 路径 | 说明 |
|---|---|
| `code/preprocess_lab/` | 预处理流水线（数据集加载、切分、缓存） |
| `code/train_lab/` | 训练流水线（模型、五折实验、评测与汇总） |
| `code/adapt_engine/` | 少样本自适应引擎 |
| `experiment_game/` | 诱导实验网页 + 操作台采集系统（`data/` 为本地数据，不入库） |
| `collect_data/` | LSL / Cyton 采集与连接相关代码 |
| `find_best_trail/` | 寻优 / 对比实验脚本 |
| `self_learing/`、`self_model/`、`shallow/` | 学习与模型原型实验 |
| `资料/` | 技术报告、实验方案、结果与说明文档 |
| `DATA/` | 本地大数据集（不入库） |

## 数据与产物

| 内容 | 约体积 | 是否入库 | 说明 |
|---|---|---|---|
| `experiment_game/data/` | 332 MB | ❌ | 3 名被试正式采集的原始脑电与派生 epochs 训练样本（sub01/sub02/sub03，2026-07-23）；本地专属，整目录不入库 |
| `code/train_lab/out/**` | 490 MB | ❌ | 五折训练输出、`*.pt` 权重、prob dump CSV |
| `资料/` | 90 MB | ✅ | 技术报告、实验方案与结果文档 |

> 被移出仓库的文件**仍在本地原路径**。如需在其它机器复现，请重新采集（`experiment_game/`）或重新训练（`code/train_lab/`）。

## 复现与运行

```bash
# 1) 环境
pip install -r requirements.txt      # 或在 code/ 下按 pyproject.toml 安装

# 2) 预处理（生成 epochs 缓存）
python code/preprocess_lab/...       # 具体入口见该目录 README

# 3) 训练 / 评测
python code/train_lab/...            # 具体入口见该目录 README

# 4) 实验操作台（诱导实验 + 采集）
experiment_game\open_operator.bat
```

## 技术要点

- **少样本个性化**：以 OpenBMI、BCI Competition IV 2a 等公开数据集为基准，研究跨被试迁移与少样本自适应
- **完整链路**：自研诱导实验程序（网页 + 操作台）→ LSL 采集 → 预处理 → 训练 → 离线验证
- **可复现实验**：五折交叉验证 + 统一的 metrics 汇总

## 许可

MIT，见 [LICENSE](LICENSE)（仅覆盖本仓库代码与文档）。
