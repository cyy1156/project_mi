# 被试独立五折实验记录（20260806_204501 / dbn_raw_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-06T20:45:01`
- device：`cuda`
- 训练设备：**NVIDIA RTX 5090**（32GB · sm_120 · conda `cyy` · PyTorch 2.11+cu128）
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`dbn_raw` | TemporalEncoder(D=64) + DBN；2s/hop100 原始时域 (B,8,500)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5090_baseline_openbmi_2s_hop100_accpaper\dbn_raw_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260806_204501`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'use_amp': True, 'cudnn_benchmark': False, 'gpu_memory_fraction': 1}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.7086 ± 0.0434`
- Test Acc_paper：`0.6940 ± 0.0450`
- Test BalAcc_maj：`0.6338 ± 0.0147`
- Test 窗级 BalAcc（附报）：`0.6246 ± 0.0130`

### Task 分折明细

#### Fold 0

- stopped_epoch：`87` | best_epoch：`67`
- Val Acc_paper（早停）：`0.7289`
- Val BalAcc_maj（附报）：`0.6664`

**Test 试次级**
- Acc_paper：`0.7127`
- BalAcc_maj：`0.6311`
- Acc_majority：`0.7127`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6194` | F1：`0.7876` | Acc：`0.6954`

#### Fold 1

- stopped_epoch：`34` | best_epoch：`14`
- Val Acc_paper（早停）：`0.7330`
- Val BalAcc_maj（附报）：`0.6250`

**Test 试次级**
- Acc_paper：`0.7509`
- BalAcc_maj：`0.6600`
- Acc_majority：`0.7509`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6490` | F1：`0.8189` | Acc：`0.7337`

#### Fold 2

- stopped_epoch：`101` | best_epoch：`81`
- Val Acc_paper（早停）：`0.6881`
- Val BalAcc_maj（附报）：`0.6608`

**Test 试次级**
- Acc_paper：`0.6433`
- BalAcc_maj：`0.6284`
- Acc_majority：`0.6433`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6179` | F1：`0.7065` | Acc：`0.6328`

#### Fold 3

- stopped_epoch：`69` | best_epoch：`49`
- Val Acc_paper（早停）：`0.7585`
- Val BalAcc_maj（附报）：`0.6689`

**Test 试次级**
- Acc_paper：`0.7242`
- BalAcc_maj：`0.6148`
- Acc_majority：`0.7242`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6111` | F1：`0.8122` | Acc：`0.7153`

#### Fold 4

- stopped_epoch：`63` | best_epoch：`43`
- Val Acc_paper（早停）：`0.6344`
- Val BalAcc_maj（附报）：`0.6019`

**Test 试次级**
- Acc_paper：`0.6387`
- BalAcc_maj：`0.6347`
- Acc_majority：`0.6387`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6254` | F1：`0.6986` | Acc：`0.6308`

### Three
- Val Acc_paper：`0.4885 ± 0.0364`
- Test Acc_paper：`0.4885 ± 0.0262`
- Test BalAcc_maj：`0.5034 ± 0.0284`
- Test 窗级 BalAcc（附报）：`0.4872 ± 0.0231`

### Three 分折明细

#### Fold 0

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.4904`
- Val BalAcc_maj（附报）：`0.5041`

**Test 试次级**
- Acc_paper：`0.4812`
- BalAcc_maj：`0.4942`
- F1-macro（众数）：`0.4848`
- Rec idle/left/right：`0.3309` / `0.6882` / `0.4636`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4792` | F1m：`0.4710`

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.4900`
- Val BalAcc_maj（附报）：`0.5063`

**Test 试次级**
- Acc_paper：`0.5248`
- BalAcc_maj：`0.5448`
- F1-macro（众数）：`0.5453`
- Rec idle/left/right：`0.4882` / `0.6145` / `0.5318`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5183` | F1m：`0.5181`

#### Fold 2

- stopped_epoch：`34` | best_epoch：`14`
- Val Acc_paper（早停）：`0.4752`
- Val BalAcc_maj（附报）：`0.4870`

**Test 试次级**
- Acc_paper：`0.4485`
- BalAcc_maj：`0.4609`
- F1-macro（众数）：`0.4461`
- Rec idle/left/right：`0.5527` / `0.5873` / `0.2427`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4517` | F1m：`0.4404`

#### Fold 3

- stopped_epoch：`64` | best_epoch：`44`
- Val Acc_paper（早停）：`0.5500`
- Val BalAcc_maj（附报）：`0.5600`

**Test 试次级**
- Acc_paper：`0.4800`
- BalAcc_maj：`0.4948`
- F1-macro（众数）：`0.4956`
- Rec idle/left/right：`0.4436` / `0.5055` / `0.5355`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4812` | F1m：`0.4819`

#### Fold 4

- stopped_epoch：`58` | best_epoch：`38`
- Val Acc_paper（早停）：`0.4370`
- Val BalAcc_maj（附报）：`0.4522`

**Test 试次级**
- Acc_paper：`0.5080`
- BalAcc_maj：`0.5220`
- F1-macro（众数）：`0.5216`
- Rec idle/left/right：`0.5220` / `0.4740` / `0.5700`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5057` | F1m：`0.5054`

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

- 结束：`2026-08-07T01:18:24`
