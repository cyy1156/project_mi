# 在线系统运行指南（experiment_game/ 目录）

> 对应演示视频里的系统：操作台（本机浏览器）+ 游戏前端（被试端浏览器）+ 实时推理与采后微调。
> 环境安装见包根 `README_代码包说明.md` §3；权重获取见 `code/README_离线代码复现指南.md` §4。

## 1. 一分钟跑起来

```
1) 双击 experiment_game/open_operator.bat（即 python experiment_game/tools/open_operator.py）
2) 浏览器打开 http://127.0.0.1:8080/operator.html#setup
3) Setup 中选数据源：合成板 synthetic（无硬件）或 Cyton 串口（有设备）
4) 开始实验 → 被试端自动开诱导页 → 按提示完成流程
```

操作台快捷键：暂停 `P` / 代确认 `N` / 标记无效 `R`。

## 2. 目录导读

| 位置 | 内容 |
|---|---|
| `tools/` | 启动与诊断入口（`open_operator.py`、`preflight` 自检） |
| `experiment/` | 核心逻辑：会话状态机、在线推理、试次级读出（因果平滑多数票）、计分、采后微调 |
| `config/` | **冻结配置**：`v3_session.yaml`（范式时序）、`ft_policy.json`（微调策略）、`e1f_four_member.json`（四成员权重登记）、`protocol.yaml`（断流看门狗） |
| `web/` | 操作台与游戏前端（纯 HTML/JS） |
| `data/` | 运行时生成：`data/subjects/<被试ID>/`（会话原始数据、个人模型、微调记录）。**包内不含真实被试数据** |

## 3. 前置自检

```bash
python -m experiment_game.tools.preflight
```

会逐项核对依赖、端口、配置文件与权重路径是否可用。其中「权重路径」指向 `code/train_lab/out/...`（压缩包内不带，见上文链接），缺失时：按途径 A 重训取得，或按途径 B 从随邮件提供的权重包解压到对应位置。

## 4. 数据源说明

- **合成板（synthetic）**：内置信号发生器，可完整演示 采集 → 分类 → 反馈 → 计分 链路，不需要任何硬件。用于流程验证，其数据不构成分类性能证据。
- **OpenBCI Cyton**：8 导真实采集，需先运行 `collect_data/` 采集软件推 LSL 流（见 `collect_data/README_采集软件说明.md`），操作台数据源选 Cyton 串口。

## 5. 采后微调（Leave-Next）

每场会话结束按 `ft_policy.json` 自动执行：逐场次采后增量微调（四成员全量）→ 保留评测门控 PASS/FAIL → 通过则晋升个人模型，失败则保留上一版并落盘告警。个人模型与全部微调记录在 `data/subjects/<被试ID>/models/`，可追溯每一轮。
