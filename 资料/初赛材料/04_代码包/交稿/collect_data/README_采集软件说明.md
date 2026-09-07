# 采集软件说明（collect_data/ 目录）

> 第 1 层 · 数据采集：OpenBCI Cyton 8 导放大器的连接、滤波与 LSL 推流，供 `experiment_game/` 在线系统以 LSL 消费。
> 只做离线复现（不接硬件）的评审**无需运行本层**，可跳过。

## 1. 职责

- 连接 Cyton（串口）并校验 8 导信号质量；
- 带通滤波（1–50 Hz）与 50 Hz 陷波；
- 以 LSL 标准推流（流名与采样率与在线系统 `protocol.yaml` 约定一致）。

## 2. 运行

```bash
# 在包根目录、已装好依赖的前提下
python collect_data/LSL_connect_model/LSL_connect_model/main.py
```

面板中选择 Cyton 串口 → 连接 → 开始推流。之后在 `experiment_game` 操作台 Setup 中选 Cyton 数据源即可进入真实采集。

## 3. 自采会话遵循的范式

自采数据使用 OpenBMI-Align 范式（与公开集对齐的时序与通道序），质控勾选表与单被试样例见包根 `03_附录B_自采数据质控样例.md`。被试一律以化名编号标识。
