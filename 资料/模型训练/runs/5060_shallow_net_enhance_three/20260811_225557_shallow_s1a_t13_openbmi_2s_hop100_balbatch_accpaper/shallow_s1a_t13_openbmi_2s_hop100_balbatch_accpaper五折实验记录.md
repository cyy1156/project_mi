# 被试独立五折实验记录（20260811_225557 / shallow_s1a_t13_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-11T22:55:57`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi-shallow-net-enhance-three subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`shallow_s1a_t13` | ShallowFBCSPNet S1a_t13 | tlen=13 nF=40/40 pool_stride=15
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_shallow_net_enhance_three_accpaper\shallow_s1a_t13_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260811_225557`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi-shallow-net-enhance-three subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6828 ± 0.0344`
- Test Acc_paper：`0.6983 ± 0.0296`
- Test BalAcc_maj：`0.6735 ± 0.0200`
- Test 窗级 BalAcc（附报）：`0.6472 ± 0.0161`

### Task 分折明细

#### Fold 0

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.6878`
- Val BalAcc_maj（附报）：`0.6628`

**Test 试次级**
- Acc_paper：`0.6830`
- BalAcc_maj：`0.6575`
- Acc_majority：`0.6830`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6344` | F1：`0.7311` | Acc：`0.6565`

#### Fold 1

- stopped_epoch：`71` | best_epoch：`51`
- Val Acc_paper（早停）：`0.7352`
- Val BalAcc_maj（附报）：`0.6739`

**Test 试次级**
- Acc_paper：`0.7391`
- BalAcc_maj：`0.7050`
- Acc_majority：`0.7391`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6713` | F1：`0.7730` | Acc：`0.7016`

#### Fold 2

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.6441`
- Val BalAcc_maj（附报）：`0.6317`

**Test 试次级**
- Acc_paper：`0.6724`
- BalAcc_maj：`0.6641`
- Acc_majority：`0.6724`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6368` | F1：`0.7124` | Acc：`0.6446`

#### Fold 3

- stopped_epoch：`66` | best_epoch：`46`
- Val Acc_paper（早停）：`0.7007`
- Val BalAcc_maj（附报）：`0.6775`

**Test 试次级**
- Acc_paper：`0.7285`
- BalAcc_maj：`0.6882`
- Acc_majority：`0.7285`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6618` | F1：`0.7739` | Acc：`0.6988`

#### Fold 4

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.6463`
- Val BalAcc_maj（附报）：`0.6122`

**Test 试次级**
- Acc_paper：`0.6683`
- BalAcc_maj：`0.6525`
- Acc_majority：`0.6683`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6318` | F1：`0.7193` | Acc：`0.6473`

### Three
- Val Acc_paper：`0.5170 ± 0.0282`
- Test Acc_paper：`0.5295 ± 0.0289`
- Test BalAcc_maj：`0.5451 ± 0.0303`
- Test 窗级 BalAcc（附报）：`0.5236 ± 0.0223`

### Three 分折明细

#### Fold 0

- stopped_epoch：`57` | best_epoch：`37`
- Val Acc_paper（早停）：`0.5281`
- Val BalAcc_maj（附报）：`0.5422`

**Test 试次级**
- Acc_paper：`0.5079`
- BalAcc_maj：`0.5267`
- F1-macro（众数）：`0.5265`
- Rec idle/left/right：`0.5618` / `0.5027` / `0.5155`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5112` | F1m：`0.5109`

#### Fold 1

- stopped_epoch：`57` | best_epoch：`37`
- Val Acc_paper（早停）：`0.5374`
- Val BalAcc_maj（附报）：`0.5519`

**Test 试次级**
- Acc_paper：`0.5485`
- BalAcc_maj：`0.5642`
- F1-macro（众数）：`0.5643`
- Rec idle/left/right：`0.5118` / `0.6373` / `0.5436`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5361` | F1m：`0.5355`

#### Fold 2

- stopped_epoch：`64` | best_epoch：`44`
- Val Acc_paper（早停）：`0.4978`
- Val BalAcc_maj（附报）：`0.5074`

**Test 试次级**
- Acc_paper：`0.4958`
- BalAcc_maj：`0.5067`
- F1-macro（众数）：`0.4999`
- Rec idle/left/right：`0.6009` / `0.5764` / `0.3427`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4926` | F1m：`0.4869`

#### Fold 3

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.5496`
- Val BalAcc_maj（附报）：`0.5626`

**Test 试次级**
- Acc_paper：`0.5755`
- BalAcc_maj：`0.5930`
- F1-macro（众数）：`0.5909`
- Rec idle/left/right：`0.6100` / `0.4918` / `0.6773`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5584` | F1m：`0.5566`

#### Fold 4

- stopped_epoch：`56` | best_epoch：`36`
- Val Acc_paper（早停）：`0.4722`
- Val BalAcc_maj（附报）：`0.4904`

**Test 试次级**
- Acc_paper：`0.5200`
- BalAcc_maj：`0.5347`
- F1-macro（众数）：`0.5335`
- Rec idle/left/right：`0.5330` / `0.6020` / `0.4690`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5196` | F1m：`0.5191`

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
  "protocol": "2s-hop100ms-balbatch-accpaper-openbmi-shallow-net-enhance-three subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
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

- 结束：`2026-08-12T01:36:20`
