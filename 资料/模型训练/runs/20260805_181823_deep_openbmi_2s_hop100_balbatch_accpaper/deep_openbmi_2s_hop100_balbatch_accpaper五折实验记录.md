# 被试独立五折实验记录（20260805_181823 / deep_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-05T18:18:23`
- device：`cuda`
- 训练设备：**NVIDIA RTX 5090**（32GB · sm_120 · conda `cyy` · PyTorch 2.11+cu128）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`deep` | Deep4Net-compat（pool=1/1，对齐1s自动缩核；非满血stride3）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_baseline_openbmi_2s_hop100_accpaper\deep_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260805_181823`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'use_amp': True, 'cudnn_benchmark': False, 'gpu_memory_fraction': 1}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7028 ± 0.0238`
- Test Acc_paper：`0.7218 ± 0.0326`
- Test BalAcc_maj：`0.6811 ± 0.0233`
- Test 窗级 BalAcc（附报）：`0.6549 ± 0.0192`

### Task 分折明细

#### Fold 0

- stopped_epoch：`111` | best_epoch：`91`
- Val Acc_paper（早停）：`0.7159`
- Val BalAcc_maj（附报）：`0.6742`

**Test 试次级**
- Acc_paper：`0.7285`
- BalAcc_maj：`0.6834`
- Acc_majority：`0.7285`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6562` | F1：`0.7756` | Acc：`0.6982`

#### Fold 1

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.7341`
- Val BalAcc_maj（附报）：`0.6811`

**Test 试次级**
- Acc_paper：`0.7415`
- BalAcc_maj：`0.7018`
- Acc_majority：`0.7415`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6686` | F1：`0.7789` | Acc：`0.7053`

#### Fold 2

- stopped_epoch：`85` | best_epoch：`65`
- Val Acc_paper（早停）：`0.6763`
- Val BalAcc_maj（附报）：`0.6606`

**Test 试次级**
- Acc_paper：`0.6794`
- BalAcc_maj：`0.6468`
- Acc_majority：`0.6794`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6273` | F1：`0.7383` | Acc：`0.6589`

#### Fold 3

- stopped_epoch：`82` | best_epoch：`62`
- Val Acc_paper（早停）：`0.7141`
- Val BalAcc_maj（附报）：`0.6772`

**Test 试次级**
- Acc_paper：`0.7682`
- BalAcc_maj：`0.7095`
- Acc_majority：`0.7682`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6814` | F1：`0.8103` | Acc：`0.7361`

#### Fold 4

- stopped_epoch：`59` | best_epoch：`39`
- Val Acc_paper（早停）：`0.6737`
- Val BalAcc_maj（附报）：`0.6078`

**Test 试次级**
- Acc_paper：`0.6913`
- BalAcc_maj：`0.6638`
- Acc_majority：`0.6913`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6409` | F1：`0.7412` | Acc：`0.6663`

### Three
- Val Acc_paper：`0.5203 ± 0.0355`
- Test Acc_paper：`0.5431 ± 0.0304`
- Test BalAcc_maj：`0.5632 ± 0.0282`
- Test 窗级 BalAcc（附报）：`0.5342 ± 0.0227`

### Three 分折明细

#### Fold 0

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.5256`
- Val BalAcc_maj（附报）：`0.5422`

**Test 试次级**
- Acc_paper：`0.5221`
- BalAcc_maj：`0.5464`
- F1-macro（众数）：`0.5459`
- Rec idle/left/right：`0.5900` / `0.5409` / `0.5082`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5183` | F1m：`0.5180`

#### Fold 1

- stopped_epoch：`71` | best_epoch：`51`
- Val Acc_paper（早停）：`0.5496`
- Val BalAcc_maj（附报）：`0.5667`

**Test 试次级**
- Acc_paper：`0.5776`
- BalAcc_maj：`0.5970`
- F1-macro（众数）：`0.5977`
- Rec idle/left/right：`0.5782` / `0.6555` / `0.5573`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5612` | F1m：`0.5612`

#### Fold 2

- stopped_epoch：`86` | best_epoch：`66`
- Val Acc_paper（早停）：`0.5130`
- Val BalAcc_maj（附报）：`0.5252`

**Test 试次级**
- Acc_paper：`0.5106`
- BalAcc_maj：`0.5312`
- F1-macro（众数）：`0.5270`
- Rec idle/left/right：`0.4336` / `0.6982` / `0.4618`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5052` | F1m：`0.5007`

#### Fold 3

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.5567`
- Val BalAcc_maj（附报）：`0.5707`

**Test 试次级**
- Acc_paper：`0.5824`
- BalAcc_maj：`0.5973`
- F1-macro（众数）：`0.5938`
- Rec idle/left/right：`0.5000` / `0.5227` / `0.7691`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5603` | F1m：`0.5565`

#### Fold 4

- stopped_epoch：`59` | best_epoch：`39`
- Val Acc_paper（早停）：`0.4567`
- Val BalAcc_maj（附报）：`0.4778`

**Test 试次级**
- Acc_paper：`0.5230`
- BalAcc_maj：`0.5440`
- F1-macro（众数）：`0.5438`
- Rec idle/left/right：`0.5680` / `0.5230` / `0.5410`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5259` | F1m：`0.5258`

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

- 结束：`2026-08-06T00:35:13`
