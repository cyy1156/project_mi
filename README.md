# project_mi

MI 项目（运动想象脑机接口）

仓库：[cyy1156/project_mi](https://github.com/cyy1156/project_mi)

## 目录结构

| 路径 | 说明 |
|------|------|
| `code/` | 预处理 / 训练流水线（`preprocess_lab`、`train_lab`） |
| `experiment_game/` | 诱导实验网页 + 操作台采集系统 |
| `collect_data/` | LSL / Cyton 采集与连接相关代码 |
| `资料/` | 文档与说明 |
| `DATA/` | 本地大数据集（**不入库**，见 `.gitignore`） |

## 自采数据

`experiment_game/data/` 默认忽略调试会话；仓库中保留 3 名被试正式采集：

- `sub01_ses01_20260723_154749`
- `sub02_ses01_20260723_180607`
- `sub03_ses01_20260723_185153`

（含对应 `sessions/` 与 `epochs/`）

## 快速入口

```text
# 操作台（需本机 Python / 依赖）
experiment_game\open_operator.bat
```
