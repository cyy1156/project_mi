# 被试独立五折实验记录（20260809_064233 / shallow_a3_lat_mu_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-09T06:42:33`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-mi-feat subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_a3_lat_mu` | ShallowFBCSPNet A3 laterality+Mu envelope (12ch)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_shallow_mi_feat_openbmi_accpaper\shallow_a3_lat_mu_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260809_064233`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-mi-feat subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---

### Three（resume 完成后补写）
- Val Acc_paper：`0.5221 ± 0.0333`
- Test Acc_paper：`0.5435 ± 0.0277`
- Test BalAcc_maj：`0.5593 ± 0.0255`
- Test 窗级 BalAcc（附报）：`0.5305 ± 0.0209`
- resume：`2026-08-09T10:22:30` · fold0/1 reeval · fold2–4 retrain · num_workers=0

### Three 分折明细

#### Fold 0

- stopped_epoch：`14` | best_epoch：`14`
- Val Acc_paper（早停）：`0.5304`
- Val BalAcc_maj（附报）：`0.5444`

**Test 试次级**
- Acc_paper：`0.5194`
- BalAcc_maj：`0.5391`
- F1-macro（众数）：`0.5390`
- Rec idle/left/right：`0.5382` / `0.5182` / `0.5609`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5158` | F1m：`0.5158`

#### Fold 1

- stopped_epoch：`11` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5344`
- Val BalAcc_maj（附报）：`0.5537`

**Test 试次级**
- Acc_paper：`0.5636`
- BalAcc_maj：`0.5764`
- F1-macro（众数）：`0.5766`
- Rec idle/left/right：`0.5900` / `0.5982` / `0.5409`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5468` | F1m：`0.5467`

#### Fold 2

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5004`
- Val BalAcc_maj（附报）：`0.5156`

**Test 试次级**
- Acc_paper：`0.5045`
- BalAcc_maj：`0.5245`
- F1-macro（众数）：`0.5192`
- Rec idle/left/right：`0.6673` / `0.5127` / `0.3936`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5000` | F1m：`0.4960`

#### Fold 3

- stopped_epoch：`69` | best_epoch：`49`
- Val Acc_paper（早停）：`0.5719`
- Val BalAcc_maj（附报）：`0.5856`

**Test 试次级**
- Acc_paper：`0.5794`
- BalAcc_maj：`0.5961`
- F1-macro（众数）：`0.5937`
- Rec idle/left/right：`0.4873` / `0.5927` / `0.7082`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5584` | F1m：`0.5558`

#### Fold 4

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.4733`
- Val BalAcc_maj（附报）：`0.4933`

**Test 试次级**
- Acc_paper：`0.5503`
- BalAcc_maj：`0.5607`
- F1-macro（众数）：`0.5595`
- Rec idle/left/right：`0.5270` / `0.6470` / `0.5080`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5316` | F1m：`0.5304`
