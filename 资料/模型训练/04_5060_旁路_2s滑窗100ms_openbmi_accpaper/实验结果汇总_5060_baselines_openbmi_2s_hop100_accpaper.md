# 实验结果汇总：OpenBMI · 2s/hop100 · Acc_paper（十一模型 · 正式）

> **正式口径**：RTX 5060 · Fast · 被试键 A · patience=20 · Task+Three · 仅 `EEG_MI_train`  
> 源数据：[`资料/实验结果/5060/openbmi滑窗_paper_acc/`](../../实验结果/5060/openbmi滑窗_paper_acc/)  
> 完整叙述报告：同目录 [`总结/OpenBMI 2s-hop100 运动想象模型对比实验完整报告.md`](../../实验结果/5060/openbmi滑窗_paper_acc/总结/OpenBMI%202s-hop100%20运动想象模型对比实验完整报告.md)  
> 权重清单：[`../5060_openbmi_accpaper_实验与权重清单.md`](../5060_openbmi_accpaper_实验与权重清单.md)  
> 补齐时间：2026-09-05（从已落盘五折 MD / 结果副本回填）

读数口径：`Tw=2s hop=100ms openbmi early_stop=val_acc_paper select=test_acc_paper balbatch`

---

## 1. Task（二分类 · 按 Test Acc_paper 降序）

| 排名 | 模型 | Val Acc_paper | Test Acc_paper | Test BalAcc_maj | 窗级 BalAcc |
|:----:|------|---------------|---------------|-----------------|-------------|
| 1 | **deep** | 0.6980±0.0316 | **0.7169±0.0335** | 0.6774±0.0214 | 0.6523±0.0200 |
| 2 | conformer | 0.7150±0.0268 | 0.7071±0.0355 | 0.6502±0.0207 | 0.6337±0.0173 |
| 3 | dgcnn_raw | 0.7103±0.0350 | 0.7059±0.0227 | 0.6543±0.0187 | 0.6375±0.0163 |
| 4 | shallow | 0.6827±0.0362 | 0.6941±0.0349 | 0.6762±0.0184 | 0.6510±0.0150 |
| 5 | eegtcnet | 0.7139±0.0350 | 0.6938±0.0237 | 0.5951±0.0521 | 0.5836±0.0455 |
| 6 | dbn_raw | 0.7118±0.0448 | 0.6919±0.0458 | 0.6395±0.0175 | 0.6275±0.0138 |
| 7 | eegnet | 0.6886±0.0165 | 0.6869±0.0334 | 0.6597±0.0275 | 0.6373±0.0214 |
| 8 | gcbnet_raw | 0.6865±0.0353 | 0.6722±0.0240 | 0.6447±0.0076 | 0.6262±0.0052 |
| 9 | dbn | 0.6297±0.0278 | 0.6210±0.0437 | 0.5437±0.0121 | 0.5411±0.0112 |
| 10 | gcbnet | 0.6157±0.0268 | 0.6169±0.0426 | 0.5624±0.0149 | 0.5539±0.0106 |
| 11 | dgcnn | 0.5885±0.0409 | 0.5943±0.0480 | 0.5659±0.0181 | 0.5588±0.0147 |

**Task 冠军：`deep`** = **0.7169±0.0335**

---

## 2. Three（三分类 · 按 Test Acc_paper 降序）

| 排名 | 模型 | Val Acc_paper | Test Acc_paper | Test BalAcc_maj | 窗级 BalAcc |
|:----:|------|---------------|---------------|-----------------|-------------|
| 1 | **shallow** | 0.5226±0.0316 | **0.5404±0.0256** | 0.5583±0.0256 | 0.5300±0.0221 |
| 2 | deep | 0.5212±0.0381 | 0.5400±0.0296 | 0.5574±0.0279 | 0.5313±0.0237 |
| 3 | conformer | 0.5159±0.0317 | 0.5375±0.0249 | 0.5533±0.0254 | 0.5283±0.0215 |
| 4 | eegnet | 0.5107±0.0358 | 0.5322±0.0291 | 0.5468±0.0290 | 0.5255±0.0218 |
| 5 | eegtcnet | 0.4944±0.0318 | 0.5103±0.0107 | 0.5215±0.0117 | 0.5055±0.0078 |
| 6 | dgcnn_raw | 0.4882±0.0364 | 0.4911±0.0306 | 0.5065±0.0314 | 0.4863±0.0259 |
| 7 | dbn_raw | 0.4875±0.0359 | 0.4883±0.0359 | 0.5040±0.0354 | 0.4867±0.0291 |
| 8 | gcbnet_raw | 0.4761±0.0352 | 0.4802±0.0229 | 0.4992±0.0216 | 0.4795±0.0190 |
| 9 | dgcnn | 0.3859±0.0203 | 0.3916±0.0152 | 0.4099±0.0151 | 0.3999±0.0114 |
| 10 | dbn | 0.3812±0.0117 | 0.3810±0.0154 | 0.3944±0.0180 | 0.3893±0.0151 |
| 11 | gcbnet | 0.3773±0.0131 | 0.3702±0.0159 | 0.3899±0.0147 | 0.3885±0.0137 |

**Three 冠军：`shallow`** = **0.5404±0.0256**

---

## 3. 同表速览（按 Three 降序）

| 排名 | 模型 | Task Acc_paper | Three Acc_paper |
|:----:|------|----------------|-----------------|
| 1 | shallow | 0.6941±0.0349 | 0.5404±0.0256 |
| 2 | deep | 0.7169±0.0335 | 0.5400±0.0296 |
| 3 | conformer | 0.7071±0.0355 | 0.5375±0.0249 |
| 4 | eegnet | 0.6869±0.0334 | 0.5322±0.0291 |
| 5 | eegtcnet | 0.6938±0.0237 | 0.5103±0.0107 |
| 6 | dgcnn_raw | 0.7059±0.0227 | 0.4911±0.0306 |
| 7 | dbn_raw | 0.6919±0.0458 | 0.4883±0.0359 |
| 8 | gcbnet_raw | 0.6722±0.0240 | 0.4802±0.0229 |
| 9 | dgcnn | 0.5943±0.0480 | 0.3916±0.0152 |
| 10 | dbn | 0.6210±0.0437 | 0.3810±0.0154 |
| 11 | gcbnet | 0.6169±0.0426 | 0.3702±0.0159 |

---

## 4. 选型备注（与报告一致）

1. Task 上 Deep 略高（约 +2.3 pp vs shallow），Three 上 shallow / deep 几乎并列（0.5404 vs 0.5400）。  
2. 后续主干取 **shallow**：领先幅度落在折间噪声内，且参数量、在线延迟、少样本微调更稳。  
3. bandpower 图模型（dgcnn/gcbnet/dbn）明显弱于 raw/CNN；`*_raw` 有 Encoder 提升但仍不及 shallow/deep 的 Three。  
4. Conformer / EEGNet 保留为后续集成异构成员。

---

## 5. 结果文件索引

| 模型 | 结果副本 MD |
|------|-------------|
| `conformer` | `资料/实验结果/5060/openbmi滑窗_paper_acc/conformer_openbmi_2s_hop100_balbatch_accpaper五折实验记录.md` |
| `dbn` | `资料/实验结果/5060/openbmi滑窗_paper_acc/dbn_openbmi_2s_hop100_balbatch_accpaper五折实验记录.md` |
| `dbn_raw` | `资料/实验结果/5060/openbmi滑窗_paper_acc/dbn_raw_openbmi_2s_hop100_balbatch_accpaper五折实验记录.md` |
| `deep` | `资料/实验结果/5060/openbmi滑窗_paper_acc/deep_openbmi_2s_hop100_balbatch_accpaper五折实验记录.md` |
| `dgcnn` | `资料/实验结果/5060/openbmi滑窗_paper_acc/dgcnn_openbmi_2s_hop100_balbatch_accpaper五折实验记录.md` |
| `dgcnn_raw` | `资料/实验结果/5060/openbmi滑窗_paper_acc/dgcnn_raw_openbmi_2s_hop100_balbatch_accpaper五折实验记录.md` |
| `eegnet` | `资料/实验结果/5060/openbmi滑窗_paper_acc/eegnet_openbmi_2s_hop100_balbatch_accpaper五折实验记录.md` |
| `eegtcnet` | `资料/实验结果/5060/openbmi滑窗_paper_acc/eegtcnet_openbmi_2s_hop100_balbatch_accpaper五折实验记录.md` |
| `gcbnet` | `资料/实验结果/5060/openbmi滑窗_paper_acc/gcbnet_openbmi_2s_hop100_balbatch_accpaper五折实验记录.md` |
| `gcbnet_raw` | `资料/实验结果/5060/openbmi滑窗_paper_acc/gcbnet_raw_openbmi_2s_hop100_balbatch_accpaper五折实验记录.md` |
| `shallow` | `资料/实验结果/5060/openbmi滑窗_paper_acc/shallow_openbmi_2s_hop100_balbatch_accpaper五折实验记录.md` |

