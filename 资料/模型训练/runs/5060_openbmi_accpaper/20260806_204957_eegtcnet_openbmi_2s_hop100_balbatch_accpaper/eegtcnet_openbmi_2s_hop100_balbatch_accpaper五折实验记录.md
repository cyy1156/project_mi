# 被试独立五折实验记录（20260806_204957 / eegtcnet_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T20:49:57`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`eegtcnet` | EEGTCNet（braindecode 默认）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_accpaper\eegtcnet_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_204957`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7139 ± 0.0350`
- Test Acc_paper：`0.6938 ± 0.0237`
- Test BalAcc_maj：`0.5951 ± 0.0521`
- Test 窗级 BalAcc（附报）：`0.5836 ± 0.0455`

### Task 分折明细

#### Fold 0

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.6815`
- Val BalAcc_maj（附报）：`0.5925`

**Test 试次级**
- Acc_paper：`0.6685`
- BalAcc_maj：`0.6048`
- Acc_majority：`0.6685`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5912` | F1：`0.7445` | Acc：`0.6495`

#### Fold 1

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.7604`
- Val BalAcc_maj（附报）：`0.6769`

**Test 试次级**
- Acc_paper：`0.7042`
- BalAcc_maj：`0.6520`
- Acc_majority：`0.7042`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6382` | F1：`0.7679` | Acc：`0.6856`

#### Fold 2

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.7215`
- Val BalAcc_maj（附报）：`0.6081`

**Test 试次级**
- Acc_paper：`0.7000`
- BalAcc_maj：`0.5900`
- Acc_majority：`0.7000`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5808` | F1：`0.7836` | Acc：`0.6784`

#### Fold 3

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.7393`
- Val BalAcc_maj（附报）：`0.6500`

**Test 试次级**
- Acc_paper：`0.7297`
- BalAcc_maj：`0.6289`
- Acc_majority：`0.7297`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6066` | F1：`0.7936` | Acc：`0.6963`

#### Fold 4

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6667`
- Val BalAcc_maj（附报）：`0.5000`

**Test 试次级**
- Acc_paper：`0.6667`
- BalAcc_maj：`0.5000`
- Acc_majority：`0.6667`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5013` | F1：`0.7997` | Acc：`0.6668`

### Three
- Val Acc_paper：`0.4944 ± 0.0318`
- Test Acc_paper：`0.5103 ± 0.0107`
- Test BalAcc_maj：`0.5215 ± 0.0117`
- Test 窗级 BalAcc（附报）：`0.5055 ± 0.0078`

### Three 分折明细

#### Fold 0

- stopped_epoch：`114` | best_epoch：`94`
- Val Acc_paper（早停）：`0.5022`
- Val BalAcc_maj（附报）：`0.5148`

**Test 试次级**
- Acc_paper：`0.5112`
- BalAcc_maj：`0.5212`
- F1-macro（众数）：`0.5160`
- Rec idle/left/right：`0.3891` / `0.5091` / `0.6655`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5032` | F1m：`0.4983`

#### Fold 1

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5107`
- Val BalAcc_maj（附报）：`0.5248`

**Test 试次级**
- Acc_paper：`0.5000`
- BalAcc_maj：`0.5088`
- F1-macro（众数）：`0.4874`
- Rec idle/left/right：`0.2518` / `0.7582` / `0.5164`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4995` | F1m：`0.4801`

#### Fold 2

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.5004`
- Val BalAcc_maj（附报）：`0.5133`

**Test 试次级**
- Acc_paper：`0.5000`
- BalAcc_maj：`0.5142`
- F1-macro（众数）：`0.5107`
- Rec idle/left/right：`0.4427` / `0.6400` / `0.4600`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4977` | F1m：`0.4950`

#### Fold 3

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.5252`
- Val BalAcc_maj（附报）：`0.5381`

**Test 试次级**
- Acc_paper：`0.5294`
- BalAcc_maj：`0.5430`
- F1-macro（众数）：`0.5289`
- Rec idle/left/right：`0.3718` / `0.4355` / `0.8218`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5197` | F1m：`0.5067`

#### Fold 4

- stopped_epoch：`98` | best_epoch：`78`
- Val Acc_paper（早停）：`0.4333`
- Val BalAcc_maj（附报）：`0.4433`

**Test 试次级**
- Acc_paper：`0.5107`
- BalAcc_maj：`0.5203`
- F1-macro（众数）：`0.5126`
- Rec idle/left/right：`0.4470` / `0.3980` / `0.7160`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5074` | F1m：`0.5008`

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

- 结束：`2026-08-07T00:25:42`
