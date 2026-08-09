# 被试独立五折实验记录（20260806_031450 / dgcnn_raw_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T03:14:50`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dgcnn_raw` | TemporalEncoder(D=64) + DGCNN(k=2)；OpenBMI 原始时域 (B,8,500)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_accpaper\dgcnn_raw_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_031450`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7103 ± 0.0350`
- Test Acc_paper：`0.7059 ± 0.0227`
- Test BalAcc_maj：`0.6543 ± 0.0187`
- Test 窗级 BalAcc（附报）：`0.6375 ± 0.0163`

### Task 分折明细

#### Fold 0

- stopped_epoch：`138` | best_epoch：`118`
- Val Acc_paper（早停）：`0.7259`
- Val BalAcc_maj（附报）：`0.6689`

**Test 试次级**
- Acc_paper：`0.7021`
- BalAcc_maj：`0.6282`
- Acc_majority：`0.7021`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6137` | F1：`0.7768` | Acc：`0.6840`

#### Fold 1

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.7330`
- Val BalAcc_maj（附报）：`0.6644`

**Test 试次级**
- Acc_paper：`0.7152`
- BalAcc_maj：`0.6695`
- Acc_majority：`0.7152`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6541` | F1：`0.7732` | Acc：`0.6956`

#### Fold 2

- stopped_epoch：`55` | best_epoch：`35`
- Val Acc_paper（早停）：`0.6715`
- Val BalAcc_maj（附报）：`0.6569`

**Test 试次级**
- Acc_paper：`0.6703`
- BalAcc_maj：`0.6464`
- Acc_majority：`0.6703`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6244` | F1：`0.7225` | Acc：`0.6464`

#### Fold 3

- stopped_epoch：`111` | best_epoch：`91`
- Val Acc_paper（早停）：`0.7544`
- Val BalAcc_maj（附报）：`0.7064`

**Test 试次级**
- Acc_paper：`0.7403`
- BalAcc_maj：`0.6809`
- Acc_majority：`0.7403`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6548` | F1：`0.7918` | Acc：`0.7112`

#### Fold 4

- stopped_epoch：`86` | best_epoch：`66`
- Val Acc_paper（早停）：`0.6667`
- Val BalAcc_maj（附报）：`0.5881`

**Test 试次级**
- Acc_paper：`0.7017`
- BalAcc_maj：`0.6467`
- Acc_majority：`0.7017`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6403` | F1：`0.7741` | Acc：`0.6913`

### Three
- Val Acc_paper：`0.4882 ± 0.0364`
- Test Acc_paper：`0.4911 ± 0.0306`
- Test BalAcc_maj：`0.5065 ± 0.0314`
- Test 窗级 BalAcc（附报）：`0.4863 ± 0.0259`

### Three 分折明细

#### Fold 0

- stopped_epoch：`54` | best_epoch：`34`
- Val Acc_paper（早停）：`0.4904`
- Val BalAcc_maj（附报）：`0.5041`

**Test 试次级**
- Acc_paper：`0.4888`
- BalAcc_maj：`0.5018`
- F1-macro（众数）：`0.4965`
- Rec idle/left/right：`0.3864` / `0.6664` / `0.4527`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4845` | F1m：`0.4798`

#### Fold 1

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.4937`
- Val BalAcc_maj（附报）：`0.5093`

**Test 试次级**
- Acc_paper：`0.5330`
- BalAcc_maj：`0.5503`
- F1-macro（众数）：`0.5510`
- Rec idle/left/right：`0.5555` / `0.5655` / `0.5300`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5239` | F1m：`0.5244`

#### Fold 2

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.4733`
- Val BalAcc_maj（附报）：`0.4885`

**Test 试次级**
- Acc_paper：`0.4382`
- BalAcc_maj：`0.4527`
- F1-macro（众数）：`0.4498`
- Rec idle/left/right：`0.5382` / `0.4627` / `0.3573`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4429` | F1m：`0.4409`

#### Fold 3

- stopped_epoch：`92` | best_epoch：`72`
- Val Acc_paper（早停）：`0.5481`
- Val BalAcc_maj（附报）：`0.5626`

**Test 试次级**
- Acc_paper：`0.5006`
- BalAcc_maj：`0.5158`
- F1-macro（众数）：`0.5167`
- Rec idle/left/right：`0.4582` / `0.5155` / `0.5736`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4925` | F1m：`0.4930`

#### Fold 4

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.4356`
- Val BalAcc_maj（附报）：`0.4548`

**Test 试次级**
- Acc_paper：`0.4947`
- BalAcc_maj：`0.5117`
- F1-macro（众数）：`0.5102`
- Rec idle/left/right：`0.5640` / `0.4360` / `0.5350`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4879` | F1m：`0.4867`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 2,
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

- 结束：`2026-08-06T07:11:36`
