# 被试独立五折实验记录（20260813_234737 / l_l1c_ciacnet_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-13T23:47:37`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-ciacnet-L subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`l_l1c_ciacnet` | L1 五折 CIACNet · Acc_paper | CIACNet full (CV1+CV2+IAT+TC)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_ciacnet_mi_accpaper\l_l1c_ciacnet_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260813_234737`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-ciacnet-L subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7173 ± 0.0251`
- Test Acc_paper：`0.7217 ± 0.0310`
- Test BalAcc_maj：`0.6811 ± 0.0194`
- Test 窗级 BalAcc（附报）：`0.6594 ± 0.0169`

### Task 分折明细

#### Fold 0

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.7348`
- Val BalAcc_maj（附报）：`0.6803`

**Test 试次级**
- Acc_paper：`0.7245`
- BalAcc_maj：`0.6775`
- Acc_majority：`0.7245`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6577` | F1：`0.7779` | Acc：`0.7006`

#### Fold 1

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.7504`
- Val BalAcc_maj（附报）：`0.6875`

**Test 试次级**
- Acc_paper：`0.7382`
- BalAcc_maj：`0.6968`
- Acc_majority：`0.7382`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6736` | F1：`0.7838` | Acc：`0.7110`

#### Fold 2

- stopped_epoch：`46` | best_epoch：`26`
- Val Acc_paper（早停）：`0.6893`
- Val BalAcc_maj（附报）：`0.6678`

**Test 试次级**
- Acc_paper：`0.6682`
- BalAcc_maj：`0.6568`
- Acc_majority：`0.6682`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6347` | F1：`0.7189` | Acc：`0.6481`

#### Fold 3

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.7248`
- Val BalAcc_maj（附报）：`0.6875`

**Test 试次级**
- Acc_paper：`0.7618`
- BalAcc_maj：`0.7091`
- Acc_majority：`0.7618`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6820` | F1：`0.8037` | Acc：`0.7305`

#### Fold 4

- stopped_epoch：`71` | best_epoch：`51`
- Val Acc_paper（早停）：`0.6870`
- Val BalAcc_maj（附报）：`0.5997`

**Test 试次级**
- Acc_paper：`0.7157`
- BalAcc_maj：`0.6653`
- Acc_majority：`0.7157`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6493` | F1：`0.7793` | Acc：`0.6988`

### Three
- Val Acc_paper：`0.5261 ± 0.0285`
- Test Acc_paper：`0.5459 ± 0.0253`
- Test BalAcc_maj：`0.5621 ± 0.0243`
- Test 窗级 BalAcc（附报）：`0.5375 ± 0.0224`

### Three 分折明细

#### Fold 0

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.5341`
- Val BalAcc_maj（附报）：`0.5507`

**Test 试次级**
- Acc_paper：`0.5248`
- BalAcc_maj：`0.5458`
- F1-macro（众数）：`0.5460`
- Rec idle/left/right：`0.5373` / `0.5336` / `0.5664`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5215` | F1m：`0.5214`

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5307`
- Val BalAcc_maj（附报）：`0.5511`

**Test 试次级**
- Acc_paper：`0.5839`
- BalAcc_maj：`0.5991`
- F1-macro（众数）：`0.5998`
- Rec idle/left/right：`0.5145` / `0.6500` / `0.6327`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5676` | F1m：`0.5675`

#### Fold 2

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.5189`
- Val BalAcc_maj（附报）：`0.5322`

**Test 试次级**
- Acc_paper：`0.5121`
- BalAcc_maj：`0.5276`
- F1-macro（众数）：`0.5224`
- Rec idle/left/right：`0.5245` / `0.6673` / `0.3909`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5038` | F1m：`0.4988`

#### Fold 3

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5678`
- Val BalAcc_maj（附报）：`0.5822`

**Test 试次级**
- Acc_paper：`0.5579`
- BalAcc_maj：`0.5721`
- F1-macro（众数）：`0.5664`
- Rec idle/left/right：`0.4364` / `0.5127` / `0.7673`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5479` | F1m：`0.5426`

#### Fold 4

- stopped_epoch：`71` | best_epoch：`51`
- Val Acc_paper（早停）：`0.4793`
- Val BalAcc_maj（附报）：`0.4952`

**Test 试次级**
- Acc_paper：`0.5507`
- BalAcc_maj：`0.5660`
- F1-macro（众数）：`0.5643`
- Rec idle/left/right：`0.5210` / `0.6610` / `0.5160`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5469` | F1m：`0.5457`

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 20,
  "batch_train": 128,
  "batch_eval": 256,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi-ciacnet-L subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 0,
  "pin_memory": true,
  "persistent_workers": true,
  "prefetch_factor": 2,
  "non_blocking": true,
  "torch_num_threads": 6,
  "cudnn_benchmark": true,
  "deterministic": false,
  "use_amp": true
}
```

- 结束：`2026-08-14T04:46:53`
