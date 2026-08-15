# 被试独立五折实验记录（20260814_162115 / shallow_a2_se_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-14T16:21:15`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-se-eca subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_a2_se` | A2 SE→Shallow · 五折 · Acc_paper | SE r=2
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_shallow_se_eca_accpaper\shallow_a2_se_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260814_162115`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-se-eca subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 0, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6840 ± 0.0321`
- Test Acc_paper：`0.6899 ± 0.0316`
- Test BalAcc_maj：`0.6695 ± 0.0177`
- Test 窗级 BalAcc（附报）：`0.6446 ± 0.0142`

### Task 分折明细

#### Fold 0

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.6752`
- Val BalAcc_maj（附报）：`0.6600`

**Test 试次级**
- Acc_paper：`0.6894`
- BalAcc_maj：`0.6773`
- Acc_majority：`0.6894`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6513` | F1：`0.7284` | Acc：`0.6613`

#### Fold 1

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.7256`
- Val BalAcc_maj（附报）：`0.6725`

**Test 试次级**
- Acc_paper：`0.7082`
- BalAcc_maj：`0.6909`
- Acc_majority：`0.7082`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6576` | F1：`0.7445` | Acc：`0.6751`

#### Fold 2

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.6548`
- Val BalAcc_maj（附报）：`0.6536`

**Test 试次级**
- Acc_paper：`0.6561`
- BalAcc_maj：`0.6511`
- Acc_majority：`0.6561`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6301` | F1：`0.7037` | Acc：`0.6361`

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.7174`
- Val BalAcc_maj（附报）：`0.6867`

**Test 试次级**
- Acc_paper：`0.7391`
- BalAcc_maj：`0.6823`
- Acc_majority：`0.7391`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6589` | F1：`0.7912` | Acc：`0.7120`

#### Fold 4

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6470`
- Val BalAcc_maj（附报）：`0.6097`

**Test 试次级**
- Acc_paper：`0.6570`
- BalAcc_maj：`0.6460`
- Acc_majority：`0.6570`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6251` | F1：`0.7058` | Acc：`0.6354`

### Three
- Val Acc_paper：`0.5211 ± 0.0306`
- Test Acc_paper：`0.5407 ± 0.0270`
- Test BalAcc_maj：`0.5581 ± 0.0259`
- Test 窗级 BalAcc（附报）：`0.5326 ± 0.0230`

### Three 分折明细

#### Fold 0

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.5296`
- Val BalAcc_maj（附报）：`0.5470`

**Test 试次级**
- Acc_paper：`0.5194`
- BalAcc_maj：`0.5403`
- F1-macro（众数）：`0.5396`
- Rec idle/left/right：`0.5964` / `0.5136` / `0.5109`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5171` | F1m：`0.5167`

#### Fold 1

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.5322`
- Val BalAcc_maj（附报）：`0.5504`

**Test 试次级**
- Acc_paper：`0.5676`
- BalAcc_maj：`0.5806`
- F1-macro（众数）：`0.5808`
- Rec idle/left/right：`0.5945` / `0.6073` / `0.5400`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5539` | F1m：`0.5540`

#### Fold 2

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5093`
- Val BalAcc_maj（附报）：`0.5256`

**Test 试次级**
- Acc_paper：`0.5021`
- BalAcc_maj：`0.5206`
- F1-macro（众数）：`0.5158`
- Rec idle/left/right：`0.6373` / `0.5436` / `0.3809`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4992` | F1m：`0.4948`

#### Fold 3

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.5637`
- Val BalAcc_maj（附报）：`0.5767`

**Test 试次级**
- Acc_paper：`0.5721`
- BalAcc_maj：`0.5915`
- F1-macro（众数）：`0.5900`
- Rec idle/left/right：`0.5273` / `0.5464` / `0.7009`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5616` | F1m：`0.5602`

#### Fold 4

- stopped_epoch：`56` | best_epoch：`36`
- Val Acc_paper（早停）：`0.4707`
- Val BalAcc_maj（附报）：`0.4911`

**Test 试次级**
- Acc_paper：`0.5423`
- BalAcc_maj：`0.5573`
- F1-macro（众数）：`0.5565`
- Rec idle/left/right：`0.5550` / `0.6180` / `0.4990`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5310` | F1m：`0.5305`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi-shallow-se-eca subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
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

- 结束：`2026-08-14T20:46:23`
