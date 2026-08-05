# 实验结果汇总：baselines_fixed_2s（BCI2a 固定窗 Cue+2~4s）

> 生成时间：2026-08-04
> 协议：`Tw=2s fixed cue+2_4s rest_cue_before_2s bci2a_only balacc+balbatch no_rap` · `n_times=500` @ 250 Hz
> 配方：普通 CE · Val **BalAcc** 早停 · train **batch balance** · **无 RAP**
> 数据：`bci2a_2s`（N=1719；Rest=556 / Task=1163；仅 BCI2a，不 merged）
> 流程：每个模型先 **Task**（静息/任务）再 **Three**（空闲/左/右；独立重训、不迁移权重）
> 训练包：`code/train_lab/src/step/baselines_fixed_2s/`

---

## 1. 读数口径与对照锚点

| 项 | 说明 |
|----|------|
| 主指标（Task） | 窗级=试次级 **Test Balanced Accuracy** |
| 主指标（Three） | **Test BalAcc**（= Recall-macro）与 **F1-macro** |
| 早停 | Val BalAcc；patience=18；max_epochs=300；lr=1e-4；batch_train=32 |
| 采样 | Task 二类 1:1；Three 三类 inverse-freq |
| Deep 结构 | **Deep4Net 默认**（非 compat；塌缩判定见 §6） |
| 历史固定 2s 锚点 | EEGNet balbatch+BalAcc ≈ **0.6395 ± 0.0775**（`20260802_151609`） |
| 旁路 2s/100ms | Task 冠军 shallow **0.6027**；Three 冠军 eegnet **0.4640** |

**正式读数 run**（均含 Task+Three；剔除中断的 `20260804_100416_eegnet`）：

| 模型 | run stamp | 权重目录 |
|------|-----------|----------|
| `eegnet` | `run_20260804_102108` | `D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\eegnet_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102108` |
| `shallow` | `run_20260804_102522` | `D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\shallow_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102522` |
| `deep` | `run_20260804_102628` | `D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\deep_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102628` |
| `eegtcnet` | `run_20260804_102741` | `D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\eegtcnet_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102741` |
| `conformer` | `run_20260804_102954` | `D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\conformer_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102954` |
| `dbn` | `run_20260804_103240` | `D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\dbn_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103240` |
| `gcbnet` | `run_20260804_103311` | `D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\gcbnet_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103311` |
| `dgcnn` | `run_20260804_103434` | `D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\dgcnn_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103434` |
| `dbn_raw` | `run_20260804_103525` | `D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\dbn_raw_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103525` |
| `gcbnet_raw` | `run_20260804_103643` | `D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\gcbnet_raw_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103643` |
| `dgcnn_raw` | `run_20260804_103808` | `D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\dgcnn_raw_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103808` |

---

## 2. Task 总表（静息 / 任务）

按 **Test BalAcc** 降序。

| 排名 | 模型 | Test BalAcc | Test Spec | Test Rec | Test F1 | Test Acc | Val BalAcc | vs hop100 Δ |
|:----:|------|-------------|-----------|----------|---------|----------|------------|-------------|
| 1 | **shallow** | 0.6539 ± 0.0788 | 0.6096 ± 0.1587 | 0.6982 ± 0.1087 | 0.7380 ± 0.0688 | 0.6694 ± 0.0733 | 0.7230 ± 0.0632 | +0.0512 |
| 2 | eegnet | 0.6395 ± 0.0775 | 0.6621 ± 0.1170 | 0.6170 ± 0.0995 | 0.6905 ± 0.0814 | 0.6312 ± 0.0775 | 0.7054 ± 0.0235 | +0.0475 |
| 3 | conformer | 0.6382 ± 0.0435 | 0.6001 ± 0.2400 | 0.6762 ± 0.2649 | 0.6867 ± 0.1953 | 0.6505 ± 0.1080 | 0.6633 ± 0.0448 | +0.0608 |
| 4 | deep | 0.5679 ± 0.0394 | 0.5949 ± 0.2736 | 0.5410 ± 0.2951 | 0.5666 ± 0.2433 | 0.5542 ± 0.1215 | 0.6507 ± 0.0680 | -0.0023 |
| 5 | dgcnn | 0.5553 ± 0.0122 | 0.5573 ± 0.3193 | 0.5533 ± 0.3143 | 0.5752 ± 0.1942 | 0.5508 ± 0.1118 | 0.5968 ± 0.0895 | +0.0191 |
| 6 | gcbnet | 0.5543 ± 0.0454 | 0.5115 ± 0.3217 | 0.5971 ± 0.3147 | 0.5906 ± 0.2515 | 0.5650 ± 0.1219 | 0.5923 ± 0.0949 | +0.0191 |
| 7 | dgcnn_raw | 0.5537 ± 0.0394 | 0.7989 ± 0.1788 | 0.3084 ± 0.2559 | 0.3806 ± 0.2495 | 0.4659 ± 0.1151 | 0.5604 ± 0.0431 | +0.0271 |
| 8 | dbn_raw | 0.5351 ± 0.0595 | 0.8332 ± 0.1572 | 0.2371 ± 0.2270 | 0.3037 ± 0.2582 | 0.4272 ± 0.1155 | 0.5899 ± 0.0320 | +0.0219 |
| 9 | eegtcnet | 0.5132 ± 0.0488 | 0.2773 ± 0.2892 | 0.7492 ± 0.3249 | 0.6658 ± 0.2394 | 0.5988 ± 0.1350 | 0.5621 ± 0.0651 | -0.0724 |
| 10 | dbn | 0.5116 ± 0.0196 | 0.7084 ± 0.3826 | 0.3148 ± 0.3716 | 0.3207 ± 0.3045 | 0.4384 ± 0.1312 | 0.6002 ± 0.0254 | +0.0024 |
| 11 | gcbnet_raw | 0.5096 ± 0.0304 | 0.8637 ± 0.1414 | 0.1556 ± 0.1489 | 0.2207 ± 0.1974 | 0.3817 ± 0.0659 | 0.5376 ± 0.0395 | -0.0175 |

**Task 冠军：`shallow`**，Test BalAcc = **0.6539 ± 0.0788**。

### 2.1 与旁路 / 历史锚点对照（Task BalAcc）

| 设定 | 代表模型 | Test BalAcc | 说明 |
|------|----------|-------------|------|
| **本方案固定窗** | **shallow**（冠军） | **0.6539 ± 0.0788** | 每试次 1 窗；Cue+2~4s |
| 本方案固定窗 | eegnet | 0.6395 ± 0.0775 | vs 历史锚点 0.6395 |
| 历史固定 2s | EEGNet | ≈0.640 ± 0.078 | 同协议量级锚点 |
| 旁路 2s/100ms | shallow | 0.6027 ± 0.0347 | 约 21 窗/试次；窗级更难 |

---

## 3. Three 总表（空闲 / 左 / 右）

按 **Test BalAcc** 降序。

| 排名 | 模型 | Test BalAcc | Test F1-macro | Test Acc | Test P-macro | Test R-macro | Rec idle/left/right（mean） | Val BalAcc | vs hop100 Δ |
|:----:|------|-------------|---------------|----------|--------------|--------------|-----------------------------|------------|-------------|
| 1 | **shallow** | 0.5349 ± 0.0855 | 0.5283 ± 0.0893 | 0.5352 ± 0.0854 | 0.5510 ± 0.0805 | 0.5349 ± 0.0855 | 0.5269 / 0.5360 / 0.5417 | 0.5503 ± 0.0503 | +0.0735 |
| 2 | conformer | 0.5297 ± 0.1004 | 0.5158 ± 0.1160 | 0.5326 ± 0.1013 | 0.5386 ± 0.0939 | 0.5297 ± 0.1004 | 0.3577 / 0.6408 / 0.5906 | 0.5429 ± 0.0435 | +0.0730 |
| 3 | eegnet | 0.4383 ± 0.0779 | 0.4134 ± 0.0869 | 0.4348 ± 0.0752 | 0.4481 ± 0.0751 | 0.4383 ± 0.0779 | 0.4937 / 0.5826 / 0.2388 | 0.5060 ± 0.0789 | -0.0257 |
| 4 | gcbnet | 0.3866 ± 0.0459 | 0.3265 ± 0.0970 | 0.3825 ± 0.0511 | 0.3989 ± 0.1002 | 0.3866 ± 0.0459 | 0.4060 / 0.2930 / 0.4609 | 0.4371 ± 0.0711 | +0.0261 |
| 5 | dgcnn | 0.3832 ± 0.0383 | 0.3137 ± 0.0949 | 0.3773 ± 0.0469 | 0.4023 ± 0.1533 | 0.3832 ± 0.0383 | 0.5416 / 0.2418 / 0.3663 | 0.4384 ± 0.0501 | +0.0224 |
| 6 | deep | 0.3779 ± 0.0231 | 0.3076 ± 0.0532 | 0.3733 ± 0.0245 | 0.3310 ± 0.0894 | 0.3779 ± 0.0231 | 0.4405 / 0.3786 / 0.3146 | 0.4481 ± 0.0474 | -0.0449 |
| 7 | dbn_raw | 0.3751 ± 0.0471 | 0.2798 ± 0.1081 | 0.3680 ± 0.0557 | 0.3037 ± 0.1416 | 0.3751 ± 0.0471 | 0.5899 / 0.3495 / 0.1860 | 0.4475 ± 0.0435 | +0.0198 |
| 8 | gcbnet_raw | 0.3693 ± 0.0293 | 0.2757 ± 0.0897 | 0.3618 ± 0.0378 | 0.3101 ± 0.1269 | 0.3693 ± 0.0293 | 0.5318 / 0.3880 / 0.1882 | 0.4153 ± 0.0393 | -0.0029 |
| 9 | dgcnn_raw | 0.3685 ± 0.0214 | 0.2866 ± 0.0754 | 0.3619 ± 0.0323 | 0.3145 ± 0.1242 | 0.3685 ± 0.0214 | 0.5804 / 0.2466 / 0.2785 | 0.3944 ± 0.0556 | -0.0024 |
| 10 | eegtcnet | 0.3481 ± 0.0172 | 0.2756 ± 0.0597 | 0.3463 ± 0.0198 | 0.3021 ± 0.0843 | 0.3481 ± 0.0172 | 0.3442 / 0.4589 / 0.2412 | 0.4157 ± 0.0459 | -0.0654 |
| 11 | dbn | 0.3357 ± 0.0048 | 0.1853 ± 0.0419 | 0.3308 ± 0.0142 | 0.1343 ± 0.0503 | 0.3357 ± 0.0048 | 0.4688 / 0.1385 / 0.4000 | 0.3530 ± 0.0243 | -0.0089 |

**Three 冠军：`shallow`**，Test BalAcc = **0.5349 ± 0.0855**，F1-macro = **0.5283 ± 0.0893**。

---

## 4. Task ↔ Three 同模型对照

| 模型 | Task BalAcc | Three BalAcc | Three F1m | Task−Three（BalAcc） | 组别 |
|------|-------------|--------------|-----------|----------------------|------|
| eegnet | 0.6395 | 0.4383 | 0.4134 | +0.2012 | 时域CNN |
| shallow | 0.6539 | 0.5349 | 0.5283 | +0.1191 | 时域CNN |
| deep | 0.5679 | 0.3779 | 0.3076 | +0.1900 | 时域CNN |
| eegtcnet | 0.5132 | 0.3481 | 0.2756 | +0.1651 | 时域CNN |
| conformer | 0.6382 | 0.5297 | 0.5158 | +0.1084 | 时域CNN |
| dbn | 0.5116 | 0.3357 | 0.1853 | +0.1759 | bandpower |
| gcbnet | 0.5543 | 0.3866 | 0.3265 | +0.1677 | bandpower |
| dgcnn | 0.5553 | 0.3832 | 0.3137 | +0.1721 | bandpower |
| dbn_raw | 0.5351 | 0.3751 | 0.2798 | +0.1600 | raw+图 |
| gcbnet_raw | 0.5096 | 0.3693 | 0.2757 | +0.1403 | raw+图 |
| dgcnn_raw | 0.5537 | 0.3685 | 0.2866 | +0.1851 | raw+图 |

---

## 5. 分组对比

| 组别 | 模型数 | Task BalAcc 均值 | Three BalAcc 均值 | Three F1m 均值 |
|------|--------|------------------|-------------------|----------------|
| 时域CNN | 5 | 0.6026 | 0.4458 | 0.4081 |
| bandpower | 3 | 0.5404 | 0.3685 | 0.2752 |
| raw+图 | 3 | 0.5328 | 0.3710 | 0.2807 |

---

## 6. Deep 是否塌缩

| fold | Test BalAcc | Spec | Rec | F1 |
|------|-------------|------|-----|-----|
| 0 | 0.6317 | 0.5891 | 0.6742 | 0.7200 |
| 1 | 0.5957 | 0.5891 | 0.6022 | 0.6694 |
| 2 | 0.5455 | 0.1406 | 0.9504 | 0.8019 |
| 3 | 0.5366 | 0.6555 | 0.4177 | 0.5279 |
| 4 | 0.5302 | 1.0000 | 0.0603 | 0.1138 |

> **判定：均值未塌缩**（Spec=0.5949 ± 0.2736，Rec=0.5410 ± 0.2951），**fold4 单折塌缩**（Spec=1.0 / Rec≈0.06）。主表可保留默认 Deep4；若要压方差可另开 compat 消融。

---

## 7. 五折明细（Test BalAcc）

### 7.1 Task

| 模型 | fold0 | fold1 | fold2 | fold3 | fold4 | mean±std |
|------|-------|-------|-------|-------|-------|----------|
| shallow | 0.6543 | 0.7353 | 0.6168 | 0.5266 | 0.7366 | 0.6539 ± 0.0788 |
| eegnet | 0.6399 | 0.7064 | 0.6069 | 0.5130 | 0.7315 | 0.6395 ± 0.0775 |
| conformer | 0.6236 | 0.5776 | 0.6280 | 0.6506 | 0.7111 | 0.6382 ± 0.0435 |
| deep | 0.6317 | 0.5957 | 0.5455 | 0.5366 | 0.5302 | 0.5679 ± 0.0394 |
| dgcnn | 0.5722 | 0.5605 | 0.5392 | 0.5435 | 0.5611 | 0.5553 ± 0.0122 |
| gcbnet | 0.4889 | 0.5601 | 0.5653 | 0.6271 | 0.5302 | 0.5543 ± 0.0454 |
| dgcnn_raw | 0.6223 | 0.5000 | 0.5473 | 0.5427 | 0.5559 | 0.5537 ± 0.0394 |
| dbn_raw | 0.6269 | 0.5031 | 0.4650 | 0.5807 | 0.5000 | 0.5351 ± 0.0595 |
| eegtcnet | 0.5000 | 0.4568 | 0.5864 | 0.4716 | 0.5514 | 0.5132 ± 0.0488 |
| dbn | 0.5000 | 0.5000 | 0.5032 | 0.5507 | 0.5043 | 0.5116 ± 0.0196 |
| gcbnet_raw | 0.5475 | 0.4980 | 0.4639 | 0.5389 | 0.5000 | 0.5096 ± 0.0304 |

### 7.2 Three

| 模型 | fold0 | fold1 | fold2 | fold3 | fold4 | mean±std |
|------|-------|-------|-------|-------|-------|----------|
| shallow | 0.5063 | 0.5943 | 0.4507 | 0.4526 | 0.6704 | 0.5349 ± 0.0855 |
| conformer | 0.5640 | 0.6413 | 0.4164 | 0.4054 | 0.6216 | 0.5297 ± 0.1004 |
| eegnet | 0.4560 | 0.4329 | 0.3902 | 0.3399 | 0.5726 | 0.4383 ± 0.0779 |
| gcbnet | 0.3750 | 0.4727 | 0.3607 | 0.3861 | 0.3386 | 0.3866 ± 0.0459 |
| dgcnn | 0.4320 | 0.3794 | 0.3509 | 0.4204 | 0.3333 | 0.3832 ± 0.0383 |
| deep | 0.3971 | 0.3605 | 0.3570 | 0.4136 | 0.3613 | 0.3779 ± 0.0231 |
| dbn_raw | 0.4521 | 0.3330 | 0.3500 | 0.4071 | 0.3333 | 0.3751 ± 0.0471 |
| gcbnet_raw | 0.3703 | 0.3526 | 0.3690 | 0.4214 | 0.3333 | 0.3693 ± 0.0293 |
| dgcnn_raw | 0.3679 | 0.3779 | 0.3994 | 0.3640 | 0.3333 | 0.3685 ± 0.0214 |
| eegtcnet | 0.3501 | 0.3255 | 0.3358 | 0.3531 | 0.3760 | 0.3481 ± 0.0172 |
| dbn | 0.3333 | 0.3333 | 0.3454 | 0.3333 | 0.3333 | 0.3357 ± 0.0048 |

---

## 8. 逐模型完整指标与记录链接

### `eegnet`（`run_20260804_102108`）

- 记录：[`runs/20260804_102108_eegnet_fixed2s_balbatch_balacc/eegnet_fixed2s_balbatch_balacc五折实验记录.md`](../runs/20260804_102108_eegnet_fixed2s_balbatch_balacc/eegnet_fixed2s_balbatch_balacc五折实验记录.md)
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\eegnet_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102108`
- **Task**：Test BalAcc `0.6395 ± 0.0775` · Spec `0.6621 ± 0.1170` · Rec `0.6170 ± 0.0995` · F1 `0.6905 ± 0.0814` · Acc `0.6312 ± 0.0775` · Val `0.7054 ± 0.0235`
- **Three**：Test BalAcc `0.4383 ± 0.0779` · F1m `0.4134 ± 0.0869` · Acc `0.4348 ± 0.0752` · P-macro `0.4481 ± 0.0751` · Rec idle/left/right `0.4937±0.1590` / `0.5826±0.1873` / `0.2388±0.1065` · Val `0.5060 ± 0.0789`

### `shallow`（`run_20260804_102522`）

- 记录：[`runs/20260804_102522_shallow_fixed2s_balbatch_balacc/shallow_fixed2s_balbatch_balacc五折实验记录.md`](../runs/20260804_102522_shallow_fixed2s_balbatch_balacc/shallow_fixed2s_balbatch_balacc五折实验记录.md)
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\shallow_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102522`
- **Task**：Test BalAcc `0.6539 ± 0.0788` · Spec `0.6096 ± 0.1587` · Rec `0.6982 ± 0.1087` · F1 `0.7380 ± 0.0688` · Acc `0.6694 ± 0.0733` · Val `0.7230 ± 0.0632`
- **Three**：Test BalAcc `0.5349 ± 0.0855` · F1m `0.5283 ± 0.0893` · Acc `0.5352 ± 0.0854` · P-macro `0.5510 ± 0.0805` · Rec idle/left/right `0.5269±0.1710` / `0.5360±0.1333` / `0.5417±0.1543` · Val `0.5503 ± 0.0503`

### `deep`（`run_20260804_102628`）

- 记录：[`runs/20260804_102628_deep_fixed2s_balbatch_balacc/deep_fixed2s_balbatch_balacc五折实验记录.md`](../runs/20260804_102628_deep_fixed2s_balbatch_balacc/deep_fixed2s_balbatch_balacc五折实验记录.md)
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\deep_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102628`
- **Task**：Test BalAcc `0.5679 ± 0.0394` · Spec `0.5949 ± 0.2736` · Rec `0.5410 ± 0.2951` · F1 `0.5666 ± 0.2433` · Acc `0.5542 ± 0.1215` · Val `0.6507 ± 0.0680`
- **Three**：Test BalAcc `0.3779 ± 0.0231` · F1m `0.3076 ± 0.0532` · Acc `0.3733 ± 0.0245` · P-macro `0.3310 ± 0.0894` · Rec idle/left/right `0.4405±0.3624` / `0.3786±0.2602` / `0.3146±0.2321` · Val `0.4481 ± 0.0474`

### `eegtcnet`（`run_20260804_102741`）

- 记录：[`runs/20260804_102741_eegtcnet_fixed2s_balbatch_balacc/eegtcnet_fixed2s_balbatch_balacc五折实验记录.md`](../runs/20260804_102741_eegtcnet_fixed2s_balbatch_balacc/eegtcnet_fixed2s_balbatch_balacc五折实验记录.md)
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\eegtcnet_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102741`
- **Task**：Test BalAcc `0.5132 ± 0.0488` · Spec `0.2773 ± 0.2892` · Rec `0.7492 ± 0.3249` · F1 `0.6658 ± 0.2394` · Acc `0.5988 ± 0.1350` · Val `0.5621 ± 0.0651`
- **Three**：Test BalAcc `0.3481 ± 0.0172` · F1m `0.2756 ± 0.0597` · Acc `0.3463 ± 0.0198` · P-macro `0.3021 ± 0.0843` · Rec idle/left/right `0.3442±0.3224` / `0.4589±0.3095` / `0.2412±0.2507` · Val `0.4157 ± 0.0459`

### `conformer`（`run_20260804_102954`）

- 记录：[`runs/20260804_102954_conformer_fixed2s_balbatch_balacc/conformer_fixed2s_balbatch_balacc五折实验记录.md`](../runs/20260804_102954_conformer_fixed2s_balbatch_balacc/conformer_fixed2s_balbatch_balacc五折实验记录.md)
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\conformer_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102954`
- **Task**：Test BalAcc `0.6382 ± 0.0435` · Spec `0.6001 ± 0.2400` · Rec `0.6762 ± 0.2649` · F1 `0.6867 ± 0.1953` · Acc `0.6505 ± 0.1080` · Val `0.6633 ± 0.0448`
- **Three**：Test BalAcc `0.5297 ± 0.1004` · F1m `0.5158 ± 0.1160` · Acc `0.5326 ± 0.1013` · P-macro `0.5386 ± 0.0939` · Rec idle/left/right `0.3577±0.1772` / `0.6408±0.0931` / `0.5906±0.1798` · Val `0.5429 ± 0.0435`

### `dbn`（`run_20260804_103240`）

- 记录：[`runs/20260804_103240_dbn_fixed2s_balbatch_balacc/dbn_fixed2s_balbatch_balacc五折实验记录.md`](../runs/20260804_103240_dbn_fixed2s_balbatch_balacc/dbn_fixed2s_balbatch_balacc五折实验记录.md)
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\dbn_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103240`
- **Task**：Test BalAcc `0.5116 ± 0.0196` · Spec `0.7084 ± 0.3826` · Rec `0.3148 ± 0.3716` · F1 `0.3207 ± 0.3045` · Acc `0.4384 ± 0.1312` · Val `0.6002 ± 0.0254`
- **Three**：Test BalAcc `0.3357 ± 0.0048` · F1m `0.1853 ± 0.0419` · Acc `0.3308 ± 0.0142` · P-macro `0.1343 ± 0.0503` · Rec idle/left/right `0.4688±0.4516` / `0.1385±0.2769` / `0.4000±0.4899` · Val `0.3530 ± 0.0243`

### `gcbnet`（`run_20260804_103311`）

- 记录：[`runs/20260804_103311_gcbnet_fixed2s_balbatch_balacc/gcbnet_fixed2s_balbatch_balacc五折实验记录.md`](../runs/20260804_103311_gcbnet_fixed2s_balbatch_balacc/gcbnet_fixed2s_balbatch_balacc五折实验记录.md)
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\gcbnet_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103311`
- **Task**：Test BalAcc `0.5543 ± 0.0454` · Spec `0.5115 ± 0.3217` · Rec `0.5971 ± 0.3147` · F1 `0.5906 ± 0.2515` · Acc `0.5650 ± 0.1219` · Val `0.5923 ± 0.0949`
- **Three**：Test BalAcc `0.3866 ± 0.0459` · F1m `0.3265 ± 0.0970` · Acc `0.3825 ± 0.0511` · P-macro `0.3989 ± 0.1002` · Rec idle/left/right `0.4060±0.3733` / `0.2930±0.1676` / `0.4609±0.2305` · Val `0.4371 ± 0.0711`

### `dgcnn`（`run_20260804_103434`）

- 记录：[`runs/20260804_103434_dgcnn_fixed2s_balbatch_balacc/dgcnn_fixed2s_balbatch_balacc五折实验记录.md`](../runs/20260804_103434_dgcnn_fixed2s_balbatch_balacc/dgcnn_fixed2s_balbatch_balacc五折实验记录.md)
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\dgcnn_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103434`
- **Task**：Test BalAcc `0.5553 ± 0.0122` · Spec `0.5573 ± 0.3193` · Rec `0.5533 ± 0.3143` · F1 `0.5752 ± 0.1942` · Acc `0.5508 ± 0.1118` · Val `0.5968 ± 0.0895`
- **Three**：Test BalAcc `0.3832 ± 0.0383` · F1m `0.3137 ± 0.0949` · Acc `0.3773 ± 0.0469` · P-macro `0.4023 ± 0.1533` · Rec idle/left/right `0.5416±0.3741` / `0.2418±0.1724` / `0.3663±0.2741` · Val `0.4384 ± 0.0501`

### `dbn_raw`（`run_20260804_103525`）

- 记录：[`runs/20260804_103525_dbn_raw_fixed2s_balbatch_balacc/dbn_raw_fixed2s_balbatch_balacc五折实验记录.md`](../runs/20260804_103525_dbn_raw_fixed2s_balbatch_balacc/dbn_raw_fixed2s_balbatch_balacc五折实验记录.md)
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\dbn_raw_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103525`
- **Task**：Test BalAcc `0.5351 ± 0.0595` · Spec `0.8332 ± 0.1572` · Rec `0.2371 ± 0.2270` · F1 `0.3037 ± 0.2582` · Acc `0.4272 ± 0.1155` · Val `0.5899 ± 0.0320`
- **Three**：Test BalAcc `0.3751 ± 0.0471` · F1m `0.2798 ± 0.1081` · Acc `0.3680 ± 0.0557` · P-macro `0.3037 ± 0.1416` · Rec idle/left/right `0.5899±0.3299` / `0.3495±0.3226` / `0.1860±0.3024` · Val `0.4475 ± 0.0435`

### `gcbnet_raw`（`run_20260804_103643`）

- 记录：[`runs/20260804_103643_gcbnet_raw_fixed2s_balbatch_balacc/gcbnet_raw_fixed2s_balbatch_balacc五折实验记录.md`](../runs/20260804_103643_gcbnet_raw_fixed2s_balbatch_balacc/gcbnet_raw_fixed2s_balbatch_balacc五折实验记录.md)
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\gcbnet_raw_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103643`
- **Task**：Test BalAcc `0.5096 ± 0.0304` · Spec `0.8637 ± 0.1414` · Rec `0.1556 ± 0.1489` · F1 `0.2207 ± 0.1974` · Acc `0.3817 ± 0.0659` · Val `0.5376 ± 0.0395`
- **Three**：Test BalAcc `0.3693 ± 0.0293` · F1m `0.2757 ± 0.0897` · Acc `0.3618 ± 0.0378` · P-macro `0.3101 ± 0.1269` · Rec idle/left/right `0.5318±0.3868` / `0.3880±0.3106` / `0.1882±0.2399` · Val `0.4153 ± 0.0393`

### `dgcnn_raw`（`run_20260804_103808`）

- 记录：[`runs/20260804_103808_dgcnn_raw_fixed2s_balbatch_balacc/dgcnn_raw_fixed2s_balbatch_balacc五折实验记录.md`](../runs/20260804_103808_dgcnn_raw_fixed2s_balbatch_balacc/dgcnn_raw_fixed2s_balbatch_balacc五折实验记录.md)
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\dgcnn_raw_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103808`
- **Task**：Test BalAcc `0.5537 ± 0.0394` · Spec `0.7989 ± 0.1788` · Rec `0.3084 ± 0.2559` · F1 `0.3806 ± 0.2495` · Acc `0.4659 ± 0.1151` · Val `0.5604 ± 0.0431`
- **Three**：Test BalAcc `0.3685 ± 0.0214` · F1m `0.2866 ± 0.0754` · Acc `0.3619 ± 0.0323` · P-macro `0.3145 ± 0.1242` · Rec idle/left/right `0.5804±0.3503` / `0.2466±0.2150` / `0.2785±0.2571` · Val `0.3944 ± 0.0556`

---

## 9. 简要结论

1. **Task 冠军 `shallow`**（0.6539）；**Three 冠军 `shallow`**（0.5349）。
2. 本方案 EEGNet Task BalAcc = **0.6395**，相对历史锚点 0.6395 的差为 **+0.0000**。
3. 相对旁路 hop100：Task 冠军 +0.0512（vs shallow 0.6027）；Three 冠军 +0.0709（vs eegnet 0.4640）。
4. 时域 CNN 整体高于 bandpower / raw+图；Three 全面低于 Task。
5. 中断 run `20260804_100416_eegnet` **不计入**正式表。

