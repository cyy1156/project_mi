# 实验结果汇总：BCI2a T · 2s/hop100 · Acc_paper 重训（十一模型）

> **现行 03 结论**（Acc_paper 早停重训；**不是** trialmaj no_retrain 复评）  
> 历史复评另见：[`实验结果汇总_baselines_2s_hop100_trialmaj_bci2a.md`](./实验结果汇总_baselines_2s_hop100_trialmaj_bci2a.md)  
> 代码：`code/train_lab/src/step/baselines_2s_hop100_accpaper/`  
> 权重：`code/train_lab/out/baseline_2s_hop100_accpaper/`  
> 五折记录：`资料/模型训练/runs/20260804_*_2s_hop100_balbatch_accpaper/`  
> 补齐时间：2026-09-05（从已落盘五折 MD 回填）

读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch no_E retrain=true`

---

## 1. Task（按 Test Acc_paper 降序）

| 排名 | 模型 | Val Acc_paper | Test Acc_paper | Test BalAcc_maj | 窗级 BalAcc |
|:----:|------|---------------|---------------|-----------------|-------------|
| 1 | **shallow** | 0.7429±0.0341 | **0.6576±0.0455** | 0.6381±0.0469 | 0.6047±0.0341 |
| 2 | conformer | 0.6984±0.0493 | 0.6446±0.0531 | 0.6251±0.0399 | 0.6007±0.0268 |
| 3 | eegtcnet | 0.6930±0.0701 | 0.6087±0.1224 | 0.5993±0.0357 | 0.5791±0.0249 |
| 4 | dbn | 0.7093±0.0502 | 0.6036±0.1197 | 0.5164±0.0137 | 0.5185±0.0175 |
| 5 | eegnet | 0.6965±0.0475 | 0.5868±0.0794 | 0.6019±0.0608 | 0.5785±0.0381 |
| 6 | deep | 0.6455±0.1460 | 0.5799±0.1158 | 0.5705±0.0328 | 0.5555±0.0272 |
| 7 | dgcnn | 0.6812±0.0400 | 0.5542±0.1238 | 0.5132±0.0136 | 0.5267±0.0098 |
| 8 | gcbnet | 0.6845±0.0466 | 0.5424±0.1308 | 0.5272±0.0210 | 0.5286±0.0176 |
| 9 | dgcnn_raw | 0.5924±0.1116 | 0.4889±0.1360 | 0.5094±0.0437 | 0.5064±0.0338 |
| 10 | dbn_raw | 0.5217±0.1473 | 0.4025±0.1426 | 0.5050±0.0104 | 0.5092±0.0122 |
| 11 | gcbnet_raw | 0.5776±0.1272 | 0.3983±0.0809 | 0.5177±0.0335 | 0.5219±0.0304 |

**Task 冠军：`shallow`** = **0.6576±0.0455**

---

## 2. Three（按 Test Acc_paper 降序）

| 排名 | 模型 | Val Acc_paper | Test Acc_paper | Test BalAcc_maj | 窗级 BalAcc |
|:----:|------|---------------|---------------|-----------------|-------------|
| 1 | **conformer** | 0.5069±0.0417 | **0.4667±0.0328** | 0.4851±0.0290 | 0.4567±0.0273 |
| 2 | shallow | 0.4921±0.0390 | 0.4597±0.0480 | 0.4954±0.0448 | 0.4701±0.0393 |
| 3 | eegnet | 0.4630±0.0530 | 0.4201±0.0701 | 0.4521±0.0643 | 0.4421±0.0530 |
| 4 | deep | 0.4312±0.0485 | 0.3950±0.0635 | 0.4281±0.0580 | 0.4131±0.0345 |
| 5 | eegtcnet | 0.4272±0.0707 | 0.3803±0.0240 | 0.4057±0.0283 | 0.4010±0.0214 |
| 6 | dgcnn | 0.3565±0.0387 | 0.3616±0.0356 | 0.3731±0.0329 | 0.3674±0.0262 |
| 7 | gcbnet_raw | 0.4266±0.0615 | 0.3475±0.0219 | 0.3734±0.0248 | 0.3779±0.0195 |
| 8 | dgcnn_raw | 0.4301±0.0798 | 0.3413±0.0280 | 0.3651±0.0278 | 0.3702±0.0249 |
| 9 | dbn_raw | 0.3888±0.0623 | 0.3411±0.0183 | 0.3500±0.0117 | 0.3531±0.0126 |
| 10 | gcbnet | 0.3795±0.0610 | 0.3353±0.0215 | 0.3449±0.0252 | 0.3551±0.0262 |
| 11 | dbn | 0.4046±0.0619 | 0.3250±0.0135 | 0.3291±0.0091 | 0.3331±0.0093 |

**Three 冠军：`conformer`** = **0.4667±0.0328**

---

## 3. 同表速览（按 Three 降序）

| 排名 | 模型 | Task Acc_paper | Three Acc_paper |
|:----:|------|----------------|-----------------|
| 1 | conformer | 0.6446±0.0531 | 0.4667±0.0328 |
| 2 | shallow | 0.6576±0.0455 | 0.4597±0.0480 |
| 3 | eegnet | 0.5868±0.0794 | 0.4201±0.0701 |
| 4 | deep | 0.5799±0.1158 | 0.3950±0.0635 |
| 5 | eegtcnet | 0.6087±0.1224 | 0.3803±0.0240 |
| 6 | dgcnn | 0.5542±0.1238 | 0.3616±0.0356 |
| 7 | gcbnet_raw | 0.3983±0.0809 | 0.3475±0.0219 |
| 8 | dgcnn_raw | 0.4889±0.1360 | 0.3413±0.0280 |
| 9 | dbn_raw | 0.4025±0.1426 | 0.3411±0.0183 |
| 10 | gcbnet | 0.5424±0.1308 | 0.3353±0.0215 |
| 11 | dbn | 0.6036±0.1197 | 0.3250±0.0135 |

---

## 4. 正式 run 索引

| 模型 | run 目录 |
|------|----------|
| `eegnet` | `资料/模型训练/runs/20260804_160649_eegnet_2s_hop100_balbatch_accpaper/` |
| `shallow` | `资料/模型训练/runs/20260804_165940_shallow_2s_hop100_balbatch_accpaper/` |
| `deep` | `资料/模型训练/runs/20260804_171252_deep_2s_hop100_balbatch_accpaper/` |
| `eegtcnet` | `资料/模型训练/runs/20260804_190821_eegtcnet_2s_hop100_balbatch_accpaper/` |
| `conformer` | `资料/模型训练/runs/20260804_195303_conformer_2s_hop100_balbatch_accpaper/` |
| `dbn` | `资料/模型训练/runs/20260804_203236_dbn_2s_hop100_balbatch_accpaper/` |
| `gcbnet` | `资料/模型训练/runs/20260804_203813_gcbnet_2s_hop100_balbatch_accpaper/` |
| `dgcnn` | `资料/模型训练/runs/20260804_204801_dgcnn_2s_hop100_balbatch_accpaper/` |
| `dbn_raw` | `资料/模型训练/runs/20260804_205728_dbn_raw_2s_hop100_balbatch_accpaper/` |
| `gcbnet_raw` | `资料/模型训练/runs/20260804_211840_gcbnet_raw_2s_hop100_balbatch_accpaper/` |
| `dgcnn_raw` | `资料/模型训练/runs/20260804_214540_dgcnn_raw_2s_hop100_balbatch_accpaper/` |


---

## 5. 质量备注

- `dbn` (`20260804_203236_dbn_2s_hop100_balbatch_accpaper`): early_stop=`balanced_accuracy`（非 Acc_paper；表中仍收录，选模时需谨慎）
- `gcbnet` (`20260804_203813_gcbnet_2s_hop100_balbatch_accpaper`): early_stop=`balanced_accuracy`（非 Acc_paper；表中仍收录，选模时需谨慎）
- `dgcnn` (`20260804_204801_dgcnn_2s_hop100_balbatch_accpaper`): early_stop=`balanced_accuracy`（非 Acc_paper；表中仍收录，选模时需谨慎）
- `dbn_raw` (`20260804_205728_dbn_raw_2s_hop100_balbatch_accpaper`): early_stop=`balanced_accuracy`（非 Acc_paper；表中仍收录，选模时需谨慎）
- `gcbnet_raw` (`20260804_211840_gcbnet_raw_2s_hop100_balbatch_accpaper`): early_stop=`balanced_accuracy`（非 Acc_paper；表中仍收录，选模时需谨慎）
- `dgcnn_raw` (`20260804_214540_dgcnn_raw_2s_hop100_balbatch_accpaper`): early_stop=`balanced_accuracy`（非 Acc_paper；表中仍收录，选模时需谨慎）

> 若图模型行的 `early_stop` 不是 `acc_paper`，表示该次落盘可能混入了旧协议尾巴；**主选模仍以 Acc_paper 早停的 CNN/Transformer 行为准**，图模型数字仅作同目录补齐，不宜单独夺冠。

---

## 6. 与历史 trialmaj 复评的关系

| 表 | 含义 | 能否当现行结论 |
|----|------|----------------|
| 本表 | Acc_paper **重训** | **可以** |
| `…trialmaj…md` | 复用 01 窗级权重的 no_retrain 复评 | **不可以** |
