# 被试独立五折实验记录（20260806_045743 / eegnet_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T04:57:43`
- device：`cuda`
- 训练设备：**NVIDIA RTX 5090**（32GB · sm_120 · conda `cyy` · PyTorch 2.11+cu128）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`eegnet` | EEGNet F1=8, D=2, F2=16（默认池化）
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_baseline_openbmi_2s_hop100_accpaper\eegnet_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_045743`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'use_amp': True, 'cudnn_benchmark': False, 'gpu_memory_fraction': 1}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6510 ± 0.0393`
- Test Acc_paper：`0.6433 ± 0.0479`
- Test BalAcc_maj：`0.6352 ± 0.0406`
- Test 窗级 BalAcc（附报）：`0.6149 ± 0.0315`

### Task 分折明细

#### Fold 0

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6041`
- Val BalAcc_maj（附报）：`0.6067`

**Test 试次级**
- Acc_paper：`0.6058`
- BalAcc_maj：`0.6032`
- Acc_majority：`0.6058`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5854` | F1：`0.6628` | Acc：`0.5912`

#### Fold 1

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.7096`
- Val BalAcc_maj（附报）：`0.6861`

**Test 试次级**
- Acc_paper：`0.6742`
- BalAcc_maj：`0.6650`
- Acc_majority：`0.6742`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6432` | F1：`0.7197` | Acc：`0.6522`

#### Fold 2

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.6630`
- Val BalAcc_maj（附报）：`0.6431`

**Test 试次级**
- Acc_paper：`0.6342`
- BalAcc_maj：`0.6275`
- Acc_majority：`0.6342`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6052` | F1：`0.6851` | Acc：`0.6136`

#### Fold 3

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.6681`
- Val BalAcc_maj（附报）：`0.6669`

**Test 试次级**
- Acc_paper：`0.7179`
- BalAcc_maj：`0.6959`
- Acc_majority：`0.7179`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6600` | F1：`0.7479` | Acc：`0.6784`

#### Fold 4

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6104`
- Val BalAcc_maj（附报）：`0.5819`

**Test 试次级**
- Acc_paper：`0.5843`
- BalAcc_maj：`0.5842`
- Acc_majority：`0.5843`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5809` | F1：`0.6450` | Acc：`0.5786`

### Three
- Val Acc_paper：`0.5120 ± 0.0410`
- Test Acc_paper：`0.5307 ± 0.0232`
- Test BalAcc_maj：`0.5451 ± 0.0245`
- Test 窗级 BalAcc（附报）：`0.5222 ± 0.0206`

### Three 分折明细

#### Fold 0

- stopped_epoch：`66` | best_epoch：`46`
- Val Acc_paper（早停）：`0.5126`
- Val BalAcc_maj（附报）：`0.5233`

**Test 试次级**
- Acc_paper：`0.5097`
- BalAcc_maj：`0.5233`
- F1-macro（众数）：`0.5221`
- Rec idle/left/right：`0.5664` / `0.5518` / `0.4518`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5040` | F1m：`0.5033`

#### Fold 1

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.5381`
- Val BalAcc_maj（附报）：`0.5530`

**Test 试次级**
- Acc_paper：`0.5570`
- BalAcc_maj：`0.5700`
- F1-macro（众数）：`0.5701`
- Rec idle/left/right：`0.5336` / `0.6118` / `0.5645`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5416` | F1m：`0.5412`

#### Fold 2

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.4989`
- Val BalAcc_maj（附报）：`0.5100`

**Test 试次级**
- Acc_paper：`0.5021`
- BalAcc_maj：`0.5158`
- F1-macro（众数）：`0.5117`
- Rec idle/left/right：`0.5591` / `0.6009` / `0.3873`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4971` | F1m：`0.4941`

#### Fold 3

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5663`
- Val BalAcc_maj（附报）：`0.5826`

**Test 试次级**
- Acc_paper：`0.5576`
- BalAcc_maj：`0.5770`
- F1-macro（众数）：`0.5752`
- Rec idle/left/right：`0.4855` / `0.5573` / `0.6882`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5500` | F1m：`0.5478`

#### Fold 4

- stopped_epoch：`71` | best_epoch：`51`
- Val Acc_paper（早停）：`0.4441`
- Val BalAcc_maj（附报）：`0.4615`

**Test 试次级**
- Acc_paper：`0.5270`
- BalAcc_maj：`0.5397`
- F1-macro（众数）：`0.5391`
- Rec idle/left/right：`0.5950` / `0.5000` / `0.5240`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5183` | F1m：`0.5178`

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

- 结束：`2026-08-06T07:22:34`
