# ShallowFBCSP 滑窗 3s/步长100ms 模型接入方案

> 日期：2026-08-24 · 约束：只出方案与审查，不代改代码。  
> 关联：《操作台顺畅度审查与优化清单》F5/F6/F11、《v2操作台升级文档》G1–G13、官方 R4（运算延迟指标）。

## 0. 现状结论（先说清楚再谈"接入"）

**该模型已经是在线模型**：`config/v2_session.yaml` 的 `s3_task_ckpt`/`s3_three_ckpt` 指向 `run_20260822_094942` 的 `best_task.pt`/`best_three.pt`（已验证在盘），即 OpenBMI 上 3s/750pt、hop100ms 滑窗训练的 ShallowFBCSPNet 双头（task 2 类 / three 3 类）。加载链：`session_v2._try_imports` → `ModelRegistry` → `RingBuffer(LSL OpenBCI_EEG)` → `OnlinePreprocessor` → `InferenceService.judge`（D8 栅格逐档取 3s 窗）→ `serial_gating`。

**缺的不是模型，是五件事**：① 启动自检（ckpt 缺失现在静默降级演练，F11）；② 在线/离线一致性验收未例行化；③ 模型卡（哪个 ckpt、什么口径、指标多少）无登记；④ 推理延迟无测量（官方 R4 要报）；⑤ 换模型流程无文档。本方案补齐这五项。

## 1. 口径语义澄清（"步长100ms"指什么）

| 层 | 口径 | 说明 |
|----|------|------|
| 训练切窗 | 3s/750pt，**hop 100ms 滑切** | 决定训练样本分布（`openbmi_3s_hop100` 数据集） |
| 在线判定 | 3s 窗，结束于 D8 判定档（**0.6s 栅格**） | 模型输入与训练同构（3s 窗）；100ms 只是训练集采样密度，不要求在线每 100ms 判一次 |
| 预处理 | 在线/离线同函数（CAR→notch+8–30Hz→逐窗 z-score），差异仅在滤波上下文（离线整段 vs 在线尾 12s），由 `filter_consistency_report()` 量化 | inference_v2.py:188 |

**结论**：训练 hop100ms 与在线 0.6s 判定栅格不冲突，无需改。若真要在线 100ms 滑判（扩展 B，§5），改的是判定密度与 D8 权重重映射，不是模型。

## 2. 接入步骤（S1–S6）

### S1 · ckpt 验收（换任何模型都先过这关）
- blob 格式 `{"model": state_dict, "n_outputs": N}`；task 头 n_outputs=2、three 头=3；
- `load_head` 可加载、单窗 `(8,750)` forward 输出 logits 形状正确；
- 通道序断言 = `CHANNEL_ORDER`（Cz,C3,C4,CP3,FC4,FC3,CP4,CPz，inference_v2.py:22 冻结序）。

### S2 · 启动自检（堵 F11 静默降级）
- 会话启动时：合成一窗过双头，记录单次推理延迟；
- **真机非演练模式**下 ckpt 缺失/加载失败 → 报错中止，不再静默降级；演练模式才允许降级并挂红色横幅（审查 F11）。

### S3 · 在线/离线一致性验收（例行化）
- 现有 `filter_consistency_report()`（inference_v2.py:188）目前只用合成数据手跑；
- 改为：每次换 ckpt 后，用 ≥1 场**真实录制会话**的 continuous 档跑离线窗 vs 在线窗，判据：argmax 不一致率 <1%、max|Δz| 存档；
- 纳入冒烟清单（与审查清单 V 系列合并执行）。

### S4 · 模型卡登记
登记表（Excel/Markdown 均可）首行先记 S3：

| 字段 | S3 值 |
|------|-------|
| 名称 | S3 双头 ShallowFBCSPNet |
| 训练集 | OpenBMI（openbmi_3s_hop100） |
| 切窗 | 3s/750pt · hop100ms |
| 5-fold 指标 | Task 0.7410 / Three 0.5873 |
| ckpt 路径 | run_20260822_094942/{task,three}/fold0/best_*.pt |
| 文件哈希/日期 | 登记时补（sha256 + 2026-08-22） |
| 零样本参照 | Stieger Three 0.42 |

后续 base v2、集成模型按同一模板追加行。

### S5 · 冒烟会话（合成板）
v2 全流程走一遍，验收：judge 档出分、gate 盒更新、Score 随档跳动（依赖审查 F5/F6 监控桥接通）、引导/轮过渡有文字（F1/F2）。

### S6 · 延迟落盘（官方 R4）
- 每次 `judge` 计时（取窗+预处理+forward），会话结束写 meta：p50/p95 ms；
- 作为提交 Excel"运算延迟"栏的数据来源；启动自检的单窗延迟作首测值。

## 3. 换模型流程（base v2 / 新训练 run 复用 S1–S6）

1. 训练侧产出同格式 blob（train_lab step 克隆：改数据集=汇合自采 npy、初始化=S3 权重）；
2. yaml 两路径改指新 ckpt（或按会话传 `v2_config_path`，orchestrator:846 已支持）；
3. 重跑 S1–S6；模型卡追加行；
4. 集成（可选）：`ModelRegistry` 接受 ckpt 列表 → 概率平均（registry.py:92-112，已实现）。

换架构才需要 `build_fn` 接线（registry.py:45；`_try_imports` 目前未传）——本次不涉及。

## 4. 验收清单

| # | 标准 |
|---|------|
| V1 | 启动自检通过：双头 forward 形状正确、延迟有首测值；ckpt 缺失真机模式报错不静默 |
| V2 | 真实会话一致性：argmax 不一致率 <1%，报告存档 |
| V3 | 模型卡 S3 行完整（含哈希） |
| V4 | 合成板冒烟：judge/gate/Score 全活 |
| V5 | meta 含推理延迟 p50/p95 |
| V6 | yaml 换 base v2 路径后 S1–S6 复跑通过（演练用假路径验证报错路径） |

## 5. 可选扩展 B：在线 100ms 滑判（默认不做）

- 动机：更平滑的反馈/更早的早停；
- 代价：D8 计分权重按 0.6s 档定义，100ms 档需重映射（票权 0.5/1.0 的时点、Score≥5 阈值都要重标）；判定调用 10 次/秒（17k 参数，算力无压力，但延迟测量与落盘量 ×6）；
- 建议：评分栅格保持 0.6s；若需更细，仅对 `DRIFT_METRIC`/反馈平滑启用 100ms 旁路，不动 D8 主链。触发条件：演示视频评审反馈"状态切换响应"不够细时再上。
