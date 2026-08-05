# 被试独立五折实验记录（20260804_195303 / conformer_2s_hop100_balbatch_accpaper）

- 开始：`2026-08-04T19:53:03`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（**仅 BCI2a T / hop100**；不用 E）
- protocol：`2s-hop100ms-balbatch-accpaper-T-only` | early_stop=**Acc_paper** | **balbatch** | no_rap
- 读数口径：`Tw=2s hop=100ms bci2a_T_only early_stop=val_acc_paper select=test_acc_paper balbatch`
- model：`conformer` | EEGConformer num_layers=2, num_heads=10, att_drop=0.5
- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100_accpaper\conformer_2s_hop100_balbatch_accpaper\bci2a_2s_hop100\run_20260804_195303`
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-balbatch-accpaper-T-only', 'early_stop': 'acc_paper', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True, 'no_balbatch': False, 'bci2a_T_only': True}`

---
## 最终结论（主报 Acc_paper）

### Task
- Val Acc_paper：`0.6984 ± 0.0493`
- Test Acc_paper：`0.6446 ± 0.0531`
- Test BalAcc_maj：`0.6251 ± 0.0399`
- Test 窗级 BalAcc（附报）：`0.6007 ± 0.0268`

### Task 分折明细

#### Fold 0

- stopped_epoch：`20` | best_epoch：`2`
- Val Acc_paper（早停）：`0.7939`
- Val BalAcc_maj（附报）：`0.7457`

**Test 试次级**
- Acc_paper：`0.5758`
- BalAcc_maj：`0.5712`
- Acc_majority：`0.5758`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.5606` | F1：`0.6268` | Acc：`0.5560`

#### Fold 1

- stopped_epoch：`31` | best_epoch：`13`
- Val Acc_paper（早停）：`0.6684`
- Val BalAcc_maj（附报）：`0.6564`

**Test 试次级**
- Acc_paper：`0.6231`
- BalAcc_maj：`0.6264`
- Acc_majority：`0.6231`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.6072` | F1：`0.6726` | Acc：`0.6032`

#### Fold 2

- stopped_epoch：`31` | best_epoch：`13`
- Val Acc_paper（早停）：`0.6947`
- Val BalAcc_maj（附报）：`0.5678`

**Test 试次级**
- Acc_paper：`0.6590`
- BalAcc_maj：`0.6083`
- Acc_majority：`0.6590`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.5866` | F1：`0.7248` | Acc：`0.6321`

#### Fold 3

- stopped_epoch：`19` | best_epoch：`1`
- Val Acc_paper（早停）：`0.6579`
- Val BalAcc_maj（附报）：`0.5061`

**Test 试次级**
- Acc_paper：`0.7364`
- BalAcc_maj：`0.6253`
- Acc_majority：`0.7364`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.6073` | F1：`0.7965` | Acc：`0.6999`

#### Fold 4

- stopped_epoch：`24` | best_epoch：`6`
- Val Acc_paper（早停）：`0.6769`
- Val BalAcc_maj（附报）：`0.5897`

**Test 试次级**
- Acc_paper：`0.6287`
- BalAcc_maj：`0.6943`
- Acc_majority：`0.6287`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.6418` | F1：`0.6296` | Acc：`0.5860`

### Three
- Val Acc_paper：`0.5069 ± 0.0417`
- Test Acc_paper：`0.4667 ± 0.0328`
- Test BalAcc_maj：`0.4851 ± 0.0290`
- Test 窗级 BalAcc（附报）：`0.4567 ± 0.0273`

### Three 分折明细

#### Fold 0

- stopped_epoch：`33` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5818`
- Val BalAcc_maj（附报）：`0.5987`

**Test 试次级**
- Acc_paper：`0.4470`
- BalAcc_maj：`0.4825`
- F1-macro（众数）：`0.4786`
- Rec idle/left/right：`0.3953` / `0.4198` / `0.6324`
- n_trials：`396`

**Test 窗级（附报）**
- BalAcc：`0.4539` | F1m：`0.4515`

#### Fold 1

- stopped_epoch：`31` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5211`
- Val BalAcc_maj（附报）：`0.5495`

**Test 试次级**
- Acc_paper：`0.4799`
- BalAcc_maj：`0.4891`
- F1-macro（众数）：`0.4851`
- Rec idle/left/right：`0.6124` / `0.4593` / `0.3955`
- n_trials：`398`

**Test 窗级（附报）**
- BalAcc：`0.4849` | F1m：`0.4813`

#### Fold 2

- stopped_epoch：`30` | best_epoch：`12`
- Val Acc_paper（早停）：`0.4684`
- Val BalAcc_maj（附报）：`0.4942`

**Test 试次级**
- Acc_paper：`0.4615`
- BalAcc_maj：`0.4732`
- F1-macro（众数）：`0.4527`
- Rec idle/left/right：`0.5000` / `0.6846` / `0.2348`
- n_trials：`390`

**Test 窗级（附报）**
- BalAcc：`0.4276` | F1m：`0.4099`

#### Fold 3

- stopped_epoch：`52` | best_epoch：`34`
- Val Acc_paper（早停）：`0.4737`
- Val BalAcc_maj（附报）：`0.5072`

**Test 试次级**
- Acc_paper：`0.4239`
- BalAcc_maj：`0.4457`
- F1-macro（众数）：`0.4263`
- Rec idle/left/right：`0.3950` / `0.2358` / `0.7063`
- n_trials：`368`

**Test 窗级（附报）**
- BalAcc：`0.4263` | F1m：`0.4111`

#### Fold 4

- stopped_epoch：`49` | best_epoch：`31`
- Val Acc_paper（早停）：`0.4897`
- Val BalAcc_maj（附报）：`0.5241`

**Test 试次级**
- Acc_paper：`0.5210`
- BalAcc_maj：`0.5349`
- F1-macro（众数）：`0.5093`
- Rec idle/left/right：`0.8039` / `0.2453` / `0.5556`
- n_trials：`167`

**Test 窗级（附报）**
- BalAcc：`0.4908` | F1m：`0.4670`

### 共用超参
```json
{
  "data_tag": "bci2a_2s_hop100",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 18,
  "batch_train": 32,
  "batch_eval": 64,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "protocol": "2s-hop100ms-balbatch-accpaper-T-only",
  "early_stop": "acc_paper",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true,
  "no_balbatch": false,
  "bci2a_T_only": true
}
```

- 结束：`2026-08-04T20:32:16`
