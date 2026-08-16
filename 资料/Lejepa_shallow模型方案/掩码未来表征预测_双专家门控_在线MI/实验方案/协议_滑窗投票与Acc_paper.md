# 协议：滑窗投票 · Acc_paper 早停 · OpenBMI 数据与超参锚点

> 本文件把新方法实验与现有 **OpenBMI Acc_paper** 旁路口径对齐。  
> 权威来源：  
> - [`code/train_lab/src/step/5060_baselines_openbmi_2s_hop100_accpaper/shared_hparams.py`](../../../../code/train_lab/src/step/5060_baselines_openbmi_2s_hop100_accpaper/shared_hparams.py)  
> - [`code/train_lab/src/step/5090_baselines_openbmi_2s_hop100_accpaper/`](../../../../code/train_lab/src/step/5090_baselines_openbmi_2s_hop100_accpaper/)  
> - Acc_paper / 试次聚合逻辑同 `self_model/train_shallow_hop100_accpaper.py` 的 `aggregate_metrics`  
> - 历史 BCI2a 仅作跨库对照，**本臂主数据改为 OpenBMI**

**新方法实验必须遵守本协议**；模块开关仍见各方案文件，但**选模/早停/读数口径不得另起一套**。

---

## 1. 滑窗几何与数据（冻结）

| 项 | 取值 |
|----|------|
| 当前窗长 \(T_w\) | **2 s** → 500 点 @ 250 Hz |
| 步长 hop | **100 ms** → 25 点 |
| 通道 | **8**（OpenBMI 预处理通道序；与 `openbmi_2s_hop100` 一致） |
| **数据** | **OpenBMI**（`data_tag=openbmi_2s_hop100`） |
| 会话 | **sess01 + sess02** |
| subject_key | `openbmi:subjNN` |
| blocks | `EEG_MI_train`（与现 OpenBMI Acc_paper 臂一致） |
| 预处理产物（基线） | `code/preprocess_lab/out/openbmi_2s_hop100` → `(N,*,8,500)` |
| 新方法扩展 | 每窗再取 past 100 + future 400；锚点仍是当前 500、hop100 |
| 训练窗 | **仅** trial 内能取满 past+cur+future 的锚点（保证有真 future；禁止零填 future） |
| 评估 / Acc_paper | 可见 past+cur≥600 即可；可含无 future 的尾窗 |
| 冷启动 | 可见 past+cur &lt; 600 → 不预测；之后每个 hop 均够 |

切窗语义：

- 与现 OpenBMI `2s_hop100` 旁路一致（MI / Rest 定义跟预处理配置）  
- 每个窗带 `(subject, trial_id)`，**禁止**把同一 trial 的窗拆到 train 与 val/test  

**不再以 BCI2a `A0*T` 作为本臂主数据**（若做跨库迁移，另开方案，不混进主表）。

---

## 2. 滑窗投票 / Acc_paper（早停与主报）

### 2.1 窗级预测

```text
ŷ_win = argmax(p_final)   # 或 A0/A1 的单头 argmax(p)
```

训练 loss 仍在**窗级** CE（+ 方案内其他损失）；**选模不看窗级 BalAcc 夺冠**。

### 2.2 Acc_paper

对每个 `(subject, trial_id)`：

1. 收集该试次全部窗预测  
2. `rate = mean(ŷ_win == y_trial)`  
3. **rate > 0.5** → 试次计对；**恰 50% 计错**  
4. Acc_paper = 试次成功率  

### 2.3 附报：多数票（不夺冠）

众数 → BalAcc_maj 等，**只附报**。

### 2.4 早停与选模（冻结）

| 阶段 | 标准 |
|------|------|
| Val 早停 | **Val Acc_paper**；patience 见 §3 |
| Test 主报 | **Test Acc_paper** 五折 mean±std |
| **禁止** | 窗级 Val BalAcc 夺冠 |

读数口径字符串：

```text
Tw=2s hop=100ms openbmi_sess01+02
subject_key=openbmi:subjNN
early_stop=val_acc_paper
select=test_acc_paper
balbatch no_rap
patience=20
batch=256/512
```

新方法可追加：`mask_future_dual_expert_P1` 等。

---

## 3. 超参锚点（OpenBMI Acc_paper 臂 + 本臂 batch）

| 项 | 锚点值 | 说明 |
|----|--------|------|
| `data_tag` | **`openbmi_2s_hop100`** | |
| `n_folds` / `val_ratio` / `seed` | 5 / 0.2 / 42 | 被试独立五折 |
| `max_epochs` | 300 | |
| `patience` | **20** | 与 OpenBMI Acc_paper `shared_hparams` 一致（非 BCI2a 的 18） |
| `batch_train` / `batch_eval` | **256 / 512** | OOM：128/256 → 64/128（meta 注明） |
| `lr` / `weight_decay` / `drop_prob` | **1e-4 / 1e-4 / 0.5** | |
| `train_sampler` | **balanced_invfreq** | balbatch |
| `optimizer` | Adam 或 AdamW | 同轮对比内固定 |
| `early_stop` | **acc_paper** | |
| `n_times_expected` | A0=**500**；P*=**1000** | |
| Encoder 实现 | A0=**braindecode**；P0+=**`self_model/shallowfbcsp.py`** | 禁止混载权重 |
| SIGReg | arXiv:2511.08544 / `rbalestr-lab/lejepa`，`num_slices=1024` | 见定稿 §3.7 |
| `protocol` | `2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | |

训练流程：

1. **Task 与 Three 均必做**（两套头独立训）  
2. Train：窗 CE + balbatch  
3. Val：聚 **Acc_paper** 早停；多损失时**不要**用 `L_total` 早停  

---

## 4. 模型性能参考（OpenBMI 对照）

> 用于 A0 量级检查。主报必须是 **同一 OpenBMI + Acc_paper** 口径。

### 4.1 OpenBMI Shallow Acc_paper（主对照）

示例旧 run（量级参考，batch 多为 128/256）：

- `资料/模型训练/runs/5090_openbmi_accpaper/20260805_135815_shallow_openbmi_2s_hop100_balbatch_accpaper/`

**A0 主表数字**：必须按本协议 **batch=256/512** 重训 `shallow_openbmi_2s_hop100_balbatch_accpaper`（Task+Three），填入 A0 结果表。

### 4.2 BCI2a 历史（不进本臂主表）

| 来源 | 说明 |
|------|------|
| self_model BCI2a T Acc_paper | 旧主数据臂；本臂已切换 OpenBMI |
| trialmaj 复评 shallow Task 0.6463±0.0288 | **no_retrain**，不可夺冠混比 |

### 4.3 新方法同时报

| 指标 | 用途 |
|------|------|
| Test Acc_paper | **主报** |
| BalAcc_maj / 窗级 BalAcc | 附报 |
| vs A0 Δ | 方法增益 |

---

## 5. 缓存队列 / 冷启动与训练窗

| 场景 | 规则 |
|------|------|
| 在线 | FIFO；**仅一开始**可见 &lt; 600 → 不输出窗预测；之后每个触发点都够 |
| 训练滑窗 | 锚点=当前 **500**、hop **100 ms**；**只保留** past+cur+future 在 trial 内齐全的窗，保证训练样本都有真 future |
| 禁止 | 用零填 future 当 `X_full`；对无 future 窗只训 CE、mask `L_pred` |
| Acc_paper | 聚合实际产出预测的窗（可含无 future 尾窗）；冷启动未输出的不计；有效窗=0 的 trial 跳过并记日志 |

---

## 6. 检查清单

- [ ] 数据为 **OpenBMI** `openbmi_2s_hop100`（sess01+02），带 `trial_id`  
- [ ] 早停 = Val **Acc_paper**，patience=**20**  
- [ ] batch=**256/512**（OOM 降配写 meta）  
- [ ] **Task + Three 均跑**  
- [ ] Train balbatch；不按窗乱拆 trial  
- [ ] 主报 Test Acc_paper；附报 BalAcc_maj  
- [ ] 口径字符串 §2.4  
- [ ] A0：**braindecode** + `(B,8,500)` 按本协议重训  
- [ ] P0+：**自写** `self_model/shallowfbcsp.py` + `(B,8,1000)` + `forward_features`  
- [ ] 含 `L_pred`：§3.2.1 索引单测通过；full 路默认 **no_grad**  
- [ ] 滑窗：当前 500 + hop100；**训练仅 future 齐全窗**（有真 future）  
- [ ] 在线：仅冷启动 &lt;600 不预测  
- [ ] Acc_paper 可含无 future 尾窗（与训练窗集差异已知）  
- [ ] SIGReg：LeJEPA 官方，`num_slices=1024`（见定稿 §3.7）  
- [ ] **D=40 vs 40→128** 消融已跑并回填较好者 

---

## 7. 结果记录模板

| 项 | 值 |
|----|-----|
| data_tag / sess | `openbmi_2s_hop100` / sess01+02 |
| protocol / early_stop | / `acc_paper` |
| batch / patience | 256/512 / 20 |
| optimizer / lr / wd / drop | |
| Val Acc_paper mean±std | |
| Test Acc_paper mean±std | |
| Test BalAcc_maj mean±std | |
| vs A0 Δ | |
| run 路径 | |
