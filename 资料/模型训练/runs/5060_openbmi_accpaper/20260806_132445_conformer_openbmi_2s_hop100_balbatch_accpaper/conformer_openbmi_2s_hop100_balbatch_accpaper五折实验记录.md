# 被试独立五折实验记录（20260806_132445 / conformer_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T13:24:45`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 OpenBMI / hop100**；blocks=EEG_MI_train）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`conformer` | EEGConformer num_layers=2, num_heads=10, att_drop=0.5
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_accpaper\conformer_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_132445`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02 blocks=EEG_MI_train', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'non_blocking': True, 'torch_num_threads': 6, 'cudnn_benchmark': True, 'deterministic': False, 'use_amp': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7150 ± 0.0268`
- Test Acc_paper：`0.7071 ± 0.0355`
- Test BalAcc_maj：`0.6502 ± 0.0207`
- Test 窗级 BalAcc（附报）：`0.6337 ± 0.0173`

### Task 分折明细

#### Fold 0

- stopped_epoch：`66` | best_epoch：`46`
- Val Acc_paper（早停）：`0.7344`
- Val BalAcc_maj（附报）：`0.6719`

**Test 试次级**
- Acc_paper：`0.6994`
- BalAcc_maj：`0.6489`
- Acc_majority：`0.6994`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6262` | F1：`0.7586` | Acc：`0.6738`

#### Fold 1

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.7315`
- Val BalAcc_maj（附报）：`0.6431`

**Test 试次级**
- Acc_paper：`0.7585`
- BalAcc_maj：`0.6870`
- Acc_majority：`0.7585`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6664` | F1：`0.8093` | Acc：`0.7304`

#### Fold 2

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.6696`
- Val BalAcc_maj（附报）：`0.6325`

**Test 试次级**
- Acc_paper：`0.6524`
- BalAcc_maj：`0.6305`
- Acc_majority：`0.6524`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6188` | F1：`0.7176` | Acc：`0.6408`

#### Fold 3

- stopped_epoch：`44` | best_epoch：`24`
- Val Acc_paper（早停）：`0.7400`
- Val BalAcc_maj（附报）：`0.6714`

**Test 试次级**
- Acc_paper：`0.7294`
- BalAcc_maj：`0.6541`
- Acc_majority：`0.7294`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6355` | F1：`0.7918` | Acc：`0.7046`

#### Fold 4

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.6993`
- Val BalAcc_maj（附报）：`0.6111`

**Test 试次级**
- Acc_paper：`0.6960`
- BalAcc_maj：`0.6305`
- Acc_majority：`0.6960`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6216` | F1：`0.7692` | Acc：`0.6806`

### Three
- Val Acc_paper：`0.5159 ± 0.0317`
- Test Acc_paper：`0.5375 ± 0.0249`
- Test BalAcc_maj：`0.5533 ± 0.0254`
- Test 窗级 BalAcc（附报）：`0.5283 ± 0.0215`

### Three 分折明细

#### Fold 0

- stopped_epoch：`34` | best_epoch：`14`
- Val Acc_paper（早停）：`0.5152`
- Val BalAcc_maj（附报）：`0.5293`

**Test 试次级**
- Acc_paper：`0.5115`
- BalAcc_maj：`0.5255`
- F1-macro（众数）：`0.5212`
- Rec idle/left/right：`0.3982` / `0.6491` / `0.5291`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5073` | F1m：`0.5034`

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5430`
- Val BalAcc_maj（附报）：`0.5570`

**Test 试次级**
- Acc_paper：`0.5724`
- BalAcc_maj：`0.5848`
- F1-macro（众数）：`0.5831`
- Rec idle/left/right：`0.5073` / `0.7282` / `0.5191`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5533` | F1m：`0.5508`

#### Fold 2

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5037`
- Val BalAcc_maj（附报）：`0.5181`

**Test 试次级**
- Acc_paper：`0.5085`
- BalAcc_maj：`0.5236`
- F1-macro（众数）：`0.5226`
- Rec idle/left/right：`0.4609` / `0.5864` / `0.5236`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4998` | F1m：`0.4983`

#### Fold 3

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5537`
- Val BalAcc_maj（附报）：`0.5659`

**Test 试次级**
- Acc_paper：`0.5564`
- BalAcc_maj：`0.5770`
- F1-macro（众数）：`0.5707`
- Rec idle/left/right：`0.4709` / `0.4845` / `0.7755`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5487` | F1m：`0.5425`

#### Fold 4

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.4637`
- Val BalAcc_maj（附报）：`0.4796`

**Test 试次级**
- Acc_paper：`0.5387`
- BalAcc_maj：`0.5557`
- F1-macro（众数）：`0.5546`
- Rec idle/left/right：`0.4860` / `0.5780` / `0.6030`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5322` | F1m：`0.5310`

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

- 结束：`2026-08-06T17:22:13`
