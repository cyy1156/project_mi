# 被试独立五折实验记录（20260807_011827 / gcbnet_raw_openbmi_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-07T01:18:27`
- device：`cuda`
- data：`F:\Cyy\MI\code\preprocess_lab\out\openbmi_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience=20`
- model：`gcbnet_raw` | TemporalEncoder(D=64) + GCBNet(k=2)；2s/hop100 原始时域 (B,8,500)
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`F:\Cyy\MI\code\train_lab\out\5060_baseline_openbmi_2s_hop100_accpaper\gcbnet_raw_openbmi_2s_hop100_balbatch_accpaper\openbmi_2s_hop100\run_20260807_011827`
- shared hp：`{'data_tag': 'openbmi_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 20, 'batch_train': 128, 'batch_eval': 256, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-openbmi subject_key=openbmi:subjNN sess01+02', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'openbmi_only': True, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'use_amp': True, 'cudnn_benchmark': False, 'gpu_memory_fraction': 1}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6874 ± 0.0256`
- Test Acc_paper：`0.6741 ± 0.0181`
- Test BalAcc_maj：`0.6423 ± 0.0118`
- Test 窗级 BalAcc（附报）：`0.6228 ± 0.0072`

### Task 分折明细

#### Fold 0

- stopped_epoch：`61` | best_epoch：`41`
- Val Acc_paper（早停）：`0.6963`
- Val BalAcc_maj（附报）：`0.6611`

**Test 试次级**
- Acc_paper：`0.6809`
- BalAcc_maj：`0.6393`
- Acc_majority：`0.6809`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6218` | F1：`0.7409` | Acc：`0.6585`

#### Fold 1

- stopped_epoch：`80` | best_epoch：`60`
- Val Acc_paper（早停）：`0.6856`
- Val BalAcc_maj（附报）：`0.6581`

**Test 试次级**
- Acc_paper：`0.6418`
- BalAcc_maj：`0.6450`
- Acc_majority：`0.6418`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6251` | F1：`0.6828` | Acc：`0.6208`

#### Fold 2

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.6941`
- Val BalAcc_maj（附报）：`0.6567`

**Test 试次级**
- Acc_paper：`0.6797`
- BalAcc_maj：`0.6316`
- Acc_majority：`0.6797`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6175` | F1：`0.7463` | Acc：`0.6610`

#### Fold 3

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.7196`
- Val BalAcc_maj（附报）：`0.6906`

**Test 试次级**
- Acc_paper：`0.6967`
- BalAcc_maj：`0.6636`
- Acc_majority：`0.6967`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6351` | F1：`0.7461` | Acc：`0.6676`

#### Fold 4

- stopped_epoch：`100` | best_epoch：`80`
- Val Acc_paper（早停）：`0.6415`
- Val BalAcc_maj（附报）：`0.5808`

**Test 试次级**
- Acc_paper：`0.6713`
- BalAcc_maj：`0.6318`
- Acc_majority：`0.6713`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.6144` | F1：`0.7346` | Acc：`0.6510`

### Three
- Val Acc_paper：`0.4819 ± 0.0351`
- Test Acc_paper：`0.4764 ± 0.0147`
- Test BalAcc_maj：`0.4951 ± 0.0129`
- Test 窗级 BalAcc（附报）：`0.4755 ± 0.0134`

### Three 分折明细

#### Fold 0

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.4819`
- Val BalAcc_maj（附报）：`0.5004`

**Test 试次级**
- Acc_paper：`0.4691`
- BalAcc_maj：`0.4930`
- F1-macro（众数）：`0.4927`
- Rec idle/left/right：`0.5055` / `0.5282` / `0.4455`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4671` | F1m：`0.4670`

#### Fold 1

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.4856`
- Val BalAcc_maj（附报）：`0.5037`

**Test 试次级**
- Acc_paper：`0.4939`
- BalAcc_maj：`0.5094`
- F1-macro（众数）：`0.5033`
- Rec idle/left/right：`0.6691` / `0.4609` / `0.3982`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4923` | F1m：`0.4868`

#### Fold 2

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.4730`
- Val BalAcc_maj（附报）：`0.4893`

**Test 试次级**
- Acc_paper：`0.4518`
- BalAcc_maj：`0.4715`
- F1-macro（众数）：`0.4723`
- Rec idle/left/right：`0.4655` / `0.4755` / `0.4736`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4550` | F1m：`0.4556`

#### Fold 3

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.5396`
- Val BalAcc_maj（附报）：`0.5559`

**Test 试次级**
- Acc_paper：`0.4818`
- BalAcc_maj：`0.5021`
- F1-macro（众数）：`0.5021`
- Rec idle/left/right：`0.4609` / `0.4755` / `0.5700`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.4764` | F1m：`0.4760`

#### Fold 4

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.4296`
- Val BalAcc_maj（附报）：`0.4459`

**Test 试次级**
- Acc_paper：`0.4853`
- BalAcc_maj：`0.4993`
- F1-macro（众数）：`0.4995`
- Rec idle/left/right：`0.4860` / `0.5200` / `0.4920`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.4865` | F1m：`0.4865`

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

- 结束：`2026-08-07T05:47:15`
