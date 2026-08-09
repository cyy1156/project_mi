# 被试独立五折实验记录（20260807_054719 / dgcnn_raw_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-07T05:47:19`
- device：`cuda`
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dgcnn_raw` | TemporalEncoder(D=64) + DGCNN(k=2)；2s/hop100 原始时域 (B,8,500)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_accpaper\dgcnn_raw_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260807_054719`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'use_amp': True, 'cudnn_benchmark': False, 'gpu_memory_fraction': 1}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7159 ± 0.0303`
- Test Acc_paper：`0.7064 ± 0.0193`
- Test BalAcc_maj：`0.6533 ± 0.0161`
- Test 窗级 BalAcc（附报）：`0.6378 ± 0.0138`

### Task 分折明细

#### Fold 0

- stopped_epoch：`130` | best_epoch：`110`
- Val Acc_paper（早停）：`0.7248`
- Val BalAcc_maj（附报）：`0.6742`

**Test 试次级**
- Acc_paper：`0.7079`
- BalAcc_maj：`0.6484`
- Acc_majority：`0.7079`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6259` | F1：`0.7715` | Acc：`0.6840`

#### Fold 1

- stopped_epoch：`57` | best_epoch：`37`
- Val Acc_paper（早停）：`0.7363`
- Val BalAcc_maj（附报）：`0.6675`

**Test 试次级**
- Acc_paper：`0.7209`
- BalAcc_maj：`0.6818`
- Acc_majority：`0.7209`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6631` | F1：`0.7712` | Acc：`0.6972`

#### Fold 2

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.6952`
- Val BalAcc_maj（附报）：`0.6781`

**Test 试次级**
- Acc_paper：`0.6848`
- BalAcc_maj：`0.6505`
- Acc_majority：`0.6848`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6272` | F1：`0.7388` | Acc：`0.6591`

#### Fold 3

- stopped_epoch：`55` | best_epoch：`35`
- Val Acc_paper（早停）：`0.7541`
- Val BalAcc_maj（附报）：`0.6964`

**Test 试次级**
- Acc_paper：`0.7333`
- BalAcc_maj：`0.6539`
- Acc_majority：`0.7333`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6419` | F1：`0.8031` | Acc：`0.7167`

#### Fold 4

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.6689`
- Val BalAcc_maj（附报）：`0.5969`

**Test 试次级**
- Acc_paper：`0.6850`
- BalAcc_maj：`0.6320`
- Acc_majority：`0.6850`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6309` | F1：`0.7630` | Acc：`0.6790`

### Three
- Val Acc_paper：`0.4908 ± 0.0360`
- Test Acc_paper：`0.4906 ± 0.0243`
- Test BalAcc_maj：`0.5078 ± 0.0234`
- Test 窗级 BalAcc（附报）：`0.4875 ± 0.0212`

### Three 分折明细

#### Fold 0

- stopped_epoch：`66` | best_epoch：`46`
- Val Acc_paper（早停）：`0.4978`
- Val BalAcc_maj（附报）：`0.5193`

**Test 试次级**
- Acc_paper：`0.4924`
- BalAcc_maj：`0.5100`
- F1-macro（众数）：`0.5065`
- Rec idle/left/right：`0.4273` / `0.6509` / `0.4518`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4853` | F1m：`0.4821`

#### Fold 1

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4915`
- Val BalAcc_maj（附报）：`0.5100`

**Test 试次级**
- Acc_paper：`0.5082`
- BalAcc_maj：`0.5252`
- F1-macro（众数）：`0.5206`
- Rec idle/left/right：`0.6409` / `0.5309` / `0.4036`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5069` | F1m：`0.5033`

#### Fold 2

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.4759`
- Val BalAcc_maj（附报）：`0.4922`

**Test 试次级**
- Acc_paper：`0.4448`
- BalAcc_maj：`0.4627`
- F1-macro（众数）：`0.4595`
- Rec idle/left/right：`0.5445` / `0.4855` / `0.3582`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4489` | F1m：`0.4467`

#### Fold 3

- stopped_epoch：`59` | best_epoch：`39`
- Val Acc_paper（早停）：`0.5500`
- Val BalAcc_maj（附报）：`0.5607`

**Test 试次级**
- Acc_paper：`0.4933`
- BalAcc_maj：`0.5139`
- F1-macro（众数）：`0.5150`
- Rec idle/left/right：`0.4627` / `0.5336` / `0.5455`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4891` | F1m：`0.4897`

#### Fold 4

- stopped_epoch：`57` | best_epoch：`37`
- Val Acc_paper（早停）：`0.4389`
- Val BalAcc_maj（附报）：`0.4544`

**Test 试次级**
- Acc_paper：`0.5140`
- BalAcc_maj：`0.5270`
- F1-macro（众数）：`0.5270`
- Rec idle/left/right：`0.5070` / `0.5120` / `0.5620`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5071` | F1m：`0.5072`

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

- 结束：`2026-08-07T10:23:47`
