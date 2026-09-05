# 附录B · 自采数据质控样例（XH-202610 源代码包随附）

> 用途：佐证自采数据（≥20 人目标）的采集规范性与质控流程。本样例为脱敏模板，
> 不含任何原始 EEG 数据与个人身份信息；被试一律以编号标识。

---

## 1. 自采协议（OpenBMI-Align v1，冻结）

| 项 | 设定 |
|---|---|
| 通道 | 8 通道：`Cz, C3, C4, CP3, FC4, FC3, CP4, CPz`（模型输入通道轴顺序，禁止重排） |
| 试次时序 | Rest 4s → prep 2s → Cue 1s → MI 4s → ITI 3s |
| 特征窗 | 3 s / 750 点，hop 100 ms，逐窗 z-score |
| 类别 | 左手 MI / 右手 MI / 空闲（Rest）三分类 |
| 每被试流程 | 校验帽检 → 多 session 采集（含 holdout session）→ 采后 Leave-Next 增量微调 → 门控判定 → 部署 current 模型 |

## 2. 每人采集后 10 分钟质控勾选（模板）

- [ ] 通道序与帽检记录符合现行配置
- [ ] 采样率 / 时序为 OpenBMI-Align（Rest4 → prep2 → Cue1 → MI4 → ITI3）
- [ ] `eeg.csv` / `events.jsonl` / `manifest.json` / run_config 齐全
- [ ] 对齐检查 PASS（事件标记与范式时序逐试次对账）
- [ ] 零样本与 Leave-Next 评测数字已登记
- [ ] 门控判定已落盘（PASS 晋升 / FAIL 强制晋升 + 告警）
- [ ] 被试进度总表已更新

## 3. 脱敏样例（单被试 · 化名编号 syj0828）

> 编号口径与《技术报告》《离线性能验证报告》一致；编号与真实身份的对应表不随作品提交。

| 项 | 记录 |
|---|---|
| 被试编号 | syj0828 |
| session 结构 | ws02–ws06 共 5 个 session，其中末位为 holdout |
| 数据落盘 | `experiment_game/data/subjects/S01/sessions/<会话名>/`（eeg.csv、events.jsonl、manifest.json） |
| 对齐检查 | PASS（逐试次事件标记与 v3_session.yaml 时序一致） |
| 零样本 MI | 41.7% |
| Leave-Next 末档 MI | 94.4%（逐轮 63.9% → 66.7% → 75.0% → 77.8% → 94.4%） |
| 门控 | 全部 PASS，current 模型已晋升 |
| 备注 | 强 MI 适配样例；弱 MI 被试同样走完整流程，按门控策略标记/强制晋升（见《技术报告》个性化适配章节） |

## 4. 隐私声明

- 压缩包内不含 `experiment_game/data/` 下的任何真实被试数据；
- 全部被试以化名编号标识，编号与真实身份的对应表不随作品提交；
- 自采已获得被试知情同意。

---

*源代码包随附 · 2026-09-05*
