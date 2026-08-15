# 被试独立五折实验记录（20260810_105457 / conformer_openbmi_2s_hop100_noz_balbatch_accpaper）

- 开始：`2026-08-10T10:54:57`
- device：`cuda` | **train_mode=`fast`**（正式出数）
- 训练设备：**NVIDIA RTX 5060 Laptop**（正式结果以本机 Fast 为准）
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100_noz`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`conformer` | EEGConformer num_layers=2, num_heads=10, att_drop=0.5
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_noz_accpaper\conformer_openbmi_2s_hop100_noz_balbatch_accpaper\openbmi_2s_hop100_noz\run_20260810_105457`
- shared hp：`{'data_tag': 'openbmi_2s_hop100_noz', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 4, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6927 ± 0.0357`
- Test Acc_paper：`0.7068 ± 0.0228`
- Test BalAcc_maj：`0.6616 ± 0.0276`
- Test 窗级 BalAcc（附报）：`0.6444 ± 0.0245`

### Task 分折明细

#### Fold 0

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.7337`
- Val BalAcc_maj（附报）：`0.6544`

**Test 试次级**
- Acc_paper：`0.6882`
- BalAcc_maj：`0.6302`
- Acc_majority：`0.6882`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6242` | F1：`0.7648` | Acc：`0.6780`

#### Fold 1

- stopped_epoch：`58` | best_epoch：`38`
- Val Acc_paper（早停）：`0.7222`
- Val BalAcc_maj（附报）：`0.6467`

**Test 试次级**
- Acc_paper：`0.7491`
- BalAcc_maj：`0.6927`
- Acc_majority：`0.7491`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6699` | F1：`0.7994` | Acc：`0.7228`

#### Fold 2

- stopped_epoch：`61` | best_epoch：`41`
- Val Acc_paper（早停）：`0.6426`
- Val BalAcc_maj（附报）：`0.6139`

**Test 试次级**
- Acc_paper：`0.6897`
- BalAcc_maj：`0.6268`
- Acc_majority：`0.6897`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6070` | F1：`0.7601` | Acc：`0.6678`

#### Fold 3

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.7059`
- Val BalAcc_maj（附报）：`0.6753`

**Test 试次级**
- Acc_paper：`0.6948`
- BalAcc_maj：`0.6748`
- Acc_majority：`0.6948`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6568` | F1：`0.7455` | Acc：`0.6755`

#### Fold 4

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.6593`
- Val BalAcc_maj（附报）：`0.6142`

**Test 试次级**
- Acc_paper：`0.7123`
- BalAcc_maj：`0.6835`
- Acc_majority：`0.7123`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6643` | F1：`0.7657` | Acc：`0.6934`

### Three
- Val Acc_paper：`0.5044 ± 0.0401`
- Test Acc_paper：`0.5315 ± 0.0267`
- Test BalAcc_maj：`0.5437 ± 0.0288`
- Test 窗级 BalAcc（附报）：`0.5241 ± 0.0243`

### Three 分折明细

#### Fold 0

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5233`
- Val BalAcc_maj（附报）：`0.5330`

**Test 试次级**
- Acc_paper：`0.4909`
- BalAcc_maj：`0.4988`
- F1-macro（众数）：`0.4964`
- Rec idle/left/right：`0.4236` / `0.4700` / `0.6027`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4885` | F1m：`0.4864`

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5289`
- Val BalAcc_maj（附报）：`0.5411`

**Test 试次级**
- Acc_paper：`0.5579`
- BalAcc_maj：`0.5715`
- F1-macro（众数）：`0.5713`
- Rec idle/left/right：`0.5355` / `0.6527` / `0.5264`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5476` | F1m：`0.5471`

#### Fold 2

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.4811`
- Val BalAcc_maj（附报）：`0.4900`

**Test 试次级**
- Acc_paper：`0.5115`
- BalAcc_maj：`0.5233`
- F1-macro（众数）：`0.5167`
- Rec idle/left/right：`0.4673` / `0.7091` / `0.3936`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5026` | F1m：`0.4961`

#### Fold 3

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5507`
- Val BalAcc_maj（附报）：`0.5633`

**Test 试次级**
- Acc_paper：`0.5594`
- BalAcc_maj：`0.5736`
- F1-macro（众数）：`0.5727`
- Rec idle/left/right：`0.6082` / `0.5091` / `0.6036`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5484` | F1m：`0.5476`

#### Fold 4

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.4381`
- Val BalAcc_maj（附报）：`0.4526`

**Test 试次级**
- Acc_paper：`0.5380`
- BalAcc_maj：`0.5513`
- F1-macro（众数）：`0.5508`
- Rec idle/left/right：`0.4940` / `0.5790` / `0.5810`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5337` | F1m：`0.5327`

### 共用超参
```json
{
  "data_tag": "openbmi_2s_hop100_noz",
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
  "protocol": "2s-hop100ms-nozscore-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "openbmi_only": true,
  "num_workers": 4,
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

- 结束：`2026-08-10T15:39:01`
