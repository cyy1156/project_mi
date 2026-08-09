# 被试独立五折实验记录（20260806_160857 / dgcnn_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T16:08:57`
- device：`cuda`
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dgcnn` | DGCNN(k=2, layers=[128]) + 2s bandpower
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_accpaper\dgcnn_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_160857`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'use_amp': True, 'cudnn_benchmark': False, 'gpu_memory_fraction': 1}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.5888 ± 0.0405`
- Test Acc_paper：`0.5947 ± 0.0481`
- Test BalAcc_maj：`0.5655 ± 0.0171`
- Test 窗级 BalAcc（附报）：`0.5585 ± 0.0145`

### Task 分折明细

#### Fold 0

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.5233`
- Val BalAcc_maj（附报）：`0.5400`

**Test 试次级**
- Acc_paper：`0.6145`
- BalAcc_maj：`0.5611`
- Acc_majority：`0.6145`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5540` | F1：`0.7030` | Acc：`0.6038`

#### Fold 1

- stopped_epoch：`64` | best_epoch：`44`
- Val Acc_paper（早停）：`0.5856`
- Val BalAcc_maj（附报）：`0.5914`

**Test 试次级**
- Acc_paper：`0.6573`
- BalAcc_maj：`0.5993`
- Acc_majority：`0.6573`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5870` | F1：`0.7348` | Acc：`0.6404`

#### Fold 2

- stopped_epoch：`72` | best_epoch：`52`
- Val Acc_paper（早停）：`0.6100`
- Val BalAcc_maj（附报）：`0.5703`

**Test 试次级**
- Acc_paper：`0.5276`
- BalAcc_maj：`0.5541`
- Acc_majority：`0.5276`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5473` | F1：`0.5781` | Acc：`0.5269`

#### Fold 3

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.6467`
- Val BalAcc_maj（附报）：`0.5786`

**Test 试次级**
- Acc_paper：`0.6233`
- BalAcc_maj：`0.5582`
- Acc_majority：`0.6233`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5530` | F1：`0.7157` | Acc：`0.6125`

#### Fold 4

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5785`
- Val BalAcc_maj（附报）：`0.5600`

**Test 试次级**
- Acc_paper：`0.5507`
- BalAcc_maj：`0.5547`
- Acc_majority：`0.5507`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5511` | F1：`0.6153` | Acc：`0.5481`

### Three
- Val Acc_paper：`0.3846 ± 0.0190`
- Test Acc_paper：`0.3891 ± 0.0130`
- Test BalAcc_maj：`0.4087 ± 0.0143`
- Test 窗级 BalAcc（附报）：`0.3997 ± 0.0103`

### Three 分折明细

#### Fold 0

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.3648`
- Val BalAcc_maj（附报）：`0.3774`

**Test 试次级**
- Acc_paper：`0.3821`
- BalAcc_maj：`0.4018`
- F1-macro（众数）：`0.3960`
- Rec idle/left/right：`0.3082` / `0.5427` / `0.3545`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3953` | F1m：`0.3900`

#### Fold 1

- stopped_epoch：`64` | best_epoch：`44`
- Val Acc_paper（早停）：`0.4111`
- Val BalAcc_maj（附报）：`0.4311`

**Test 试次级**
- Acc_paper：`0.4042`
- BalAcc_maj：`0.4264`
- F1-macro（众数）：`0.4100`
- Rec idle/left/right：`0.3218` / `0.6836` / `0.2736`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4129` | F1m：`0.4001`

#### Fold 2

- stopped_epoch：`78` | best_epoch：`58`
- Val Acc_paper（早停）：`0.3756`
- Val BalAcc_maj（附报）：`0.3952`

**Test 试次级**
- Acc_paper：`0.3685`
- BalAcc_maj：`0.3855`
- F1-macro（众数）：`0.3730`
- Rec idle/left/right：`0.5582` / `0.3745` / `0.2236`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.3825` | F1m：`0.3736`

#### Fold 3

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.4033`
- Val BalAcc_maj（附报）：`0.4148`

**Test 试次级**
- Acc_paper：`0.3897`
- BalAcc_maj：`0.4100`
- F1-macro（众数）：`0.4037`
- Rec idle/left/right：`0.4345` / `0.5173` / `0.2782`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4031` | F1m：`0.3988`

#### Fold 4

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.3681`
- Val BalAcc_maj（附报）：`0.3878`

**Test 试次级**
- Acc_paper：`0.4010`
- BalAcc_maj：`0.4197`
- F1-macro（众数）：`0.4193`
- Rec idle/left/right：`0.4520` / `0.4410` / `0.3660`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4047` | F1m：`0.4042`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 2,
  "pin_memory": true,
  "persistent_workers": true,
  "non_blocking": true,
  "use_amp": true,
  "cudnn_benchmark": false,
  "gpu_memory_fraction": 1
}
```

- 结束：`2026-08-06T19:01:30`
