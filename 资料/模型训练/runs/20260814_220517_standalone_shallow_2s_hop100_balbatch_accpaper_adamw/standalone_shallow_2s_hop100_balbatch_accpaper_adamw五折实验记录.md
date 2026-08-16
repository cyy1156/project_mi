# 5-fold experiment record (20260814_220517 / standalone_shallow_2s_hop100_balbatch_accpaper_adamw)

- start: `2026-08-14T22:05:17`
- device: `cpu`
- data: `D:\360MoveData\Users\ckgxnn\Desktop\MI\code\preprocess_lab\out\bci2a_2s_hop100` (BCI2a T / hop100 only)
- protocol: `2s-hop100ms-balbatch-accpaper-T-only` | early_stop=**Acc_paper** | **balbatch**
- optimizer: `adamw` | lr=`0.0001` | weight_decay=`0.0001`
- model: `standalone ShallowFBCSPNet` | no braindecode
- Train: window CE + batch balance; Val/Test: trial Acc_paper
- pipeline: Task(2-class) then Three(3-class)
- weights: `D:\360MoveData\Users\ckgxnn\Desktop\MI\self_model\out\optimizer_compare_full\adamw`
- shared hp: `{'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'n_times_expected': 500, 'n_chans': 8, 'protocol': '2s-hop100ms-balbatch-accpaper-T-only', 'optimizer': 'adamw'}`

---

### Task
`0.5614` / `0.6981`

### Task fold details

#### Fold 0

- stopped_epoch: `21` | best_epoch: `3`
- Val Acc_paper: `0.8182`
- Val BalAcc_maj: `0.7842`

**Test trial**
- Acc_paper: `0.6010`
- BalAcc_maj: `0.5859`
- Acc_majority: `0.6010`
- Sens/Spec/F1: `0.6292` / `0.5426` / `0.6802`
- CM: `[[70, 59], [99, 168]]`
- n_trials: `396`

**Test window**
- BalAcc: `0.5660` | F1: `0.6468` | Sens: `0.5780` | Spec: `0.5540`
- CM: `[[1458, 1174], [2366, 3241]]`

#### Fold 1

- stopped_epoch: `33` | best_epoch: `15`
- Val Acc_paper: `0.7000`
- Val BalAcc_maj: `0.6235`

**Test trial**
- Acc_paper: `0.6206`
- BalAcc_maj: `0.6165`
- Acc_majority: `0.6206`
- Sens/Spec/F1: `0.6283` / `0.6047` / `0.6912`
- CM: `[[78, 51], [100, 169]]`
- n_trials: `398`

**Test window**
- BalAcc: `0.6022` | F1: `0.6880` | Sens: `0.6268` | Spec: `0.5777`
- CM: `[[1510, 1104], [2108, 3541]]`

#### Fold 2

- stopped_epoch: `25` | best_epoch: `7`
- Val Acc_paper: `0.7000`
- Val BalAcc_maj: `0.6019`

**Test trial**
- Acc_paper: `0.7026`
- BalAcc_maj: `0.6548`
- Acc_majority: `0.7026`
- Sens/Spec/F1: `0.7939` / `0.5156` / `0.7820`
- CM: `[[66, 62], [54, 208]]`
- n_trials: `390`

**Test window**
- BalAcc: `0.6200` | F1: `0.7460` | Sens: `0.7336` | Spec: `0.5064`
- CM: `[[1315, 1282], [1466, 4036]]`

#### Fold 3

- stopped_epoch: `31` | best_epoch: `13`
- Val Acc_paper: `0.6895`
- Val BalAcc_maj: `0.6374`

**Test trial**
- Acc_paper: `0.6630`
- BalAcc_maj: `0.5821`
- Acc_majority: `0.6630`
- Sens/Spec/F1: `0.8112` / `0.3529` / `0.7652`
- CM: `[[42, 77], [47, 202]]`
- n_trials: `368`

**Test window**
- BalAcc: `0.5634` | F1: `0.7269` | Sens: `0.7292` | Spec: `0.3975`
- CM: `[[956, 1449], [1416, 3813]]`

#### Fold 4

- stopped_epoch: `29` | best_epoch: `11`
- Val Acc_paper: `0.7513`
- Val BalAcc_maj: `0.6770`

**Test trial**
- Acc_paper: `0.6647`
- BalAcc_maj: `0.7147`
- Acc_majority: `0.6647`
- Sens/Spec/F1: `0.5862` / `0.8431` / `0.7083`
- CM: `[[43, 8], [48, 68]]`
- n_trials: `167`

**Test window**
- BalAcc: `0.6703` | F1: `0.6828` | Sens: `0.5690` | Spec: `0.7716`
- CM: `[[804, 238], [1050, 1386]]`

### Three
`0.7349`

### Three fold details

#### Fold 0

- stopped_epoch: `24` | best_epoch: `6`
- Val Acc_paper: `0.5636`
- Val BalAcc_maj: `0.6133`

**Test trial**
- Acc_paper: `0.4444`
- BalAcc_maj: `0.4776`
- F1-macro: `0.4733`
- Rec idle/left/right: `0.4186` / `0.3893` / `0.6250`
- Spec(macro): `0.7393`
- CM: `[[54, 29, 46], [37, 51, 43], [29, 22, 85]]`
- n_trials: `396`

**Test window**
- BalAcc: `0.4558` | F1m: `0.4536` | Spec: `0.7283`
- CM: `[[1148, 536, 948], [878, 1039, 834], [680, 595, 1581]]`

#### Fold 1

- stopped_epoch: `26` | best_epoch: `8`
- Val Acc_paper: `0.4895`
- Val BalAcc_maj: `0.5100`

**Test trial**
- Acc_paper: `0.4774`
- BalAcc_maj: `0.4920`
- F1-macro: `0.4895`
- Rec idle/left/right: `0.6434` / `0.4370` / `0.3955`
- Spec(macro): `0.7465`
- CM: `[[83, 24, 22], [61, 59, 15], [65, 16, 53]]`
- n_trials: `398`

**Test window**
- BalAcc: `0.4838` | F1m: `0.4787` | Spec: `0.7423`
- CM: `[[1653, 470, 491], [1313, 1127, 395], [1169, 459, 1186]]`

#### Fold 2

- stopped_epoch: `37` | best_epoch: `19`
- Val Acc_paper: `0.4684`
- Val BalAcc_maj: `0.4801`

**Test trial**
- Acc_paper: `0.4231`
- BalAcc_maj: `0.4466`
- F1-macro: `0.4205`
- Rec idle/left/right: `0.3203` / `0.7846` / `0.2348`
- Spec(macro): `0.7230`
- CM: `[[41, 77, 10], [8, 102, 20], [19, 82, 31]]`
- n_trials: `390`

**Test window**
- BalAcc: `0.4189` | F1m: `0.4019` | Spec: `0.7092`
- CM: `[[778, 1454, 365], [359, 1871, 500], [452, 1567, 753]]`

#### Fold 3

- stopped_epoch: `34` | best_epoch: `16`
- Val Acc_paper: `0.5211`
- Val BalAcc_maj: `0.5210`

**Test trial**
- Acc_paper: `0.4620`
- BalAcc_maj: `0.4665`
- F1-macro: `0.4481`
- Rec idle/left/right: `0.3529` / `0.2846` / `0.7619`
- Spec(macro): `0.7335`
- CM: `[[42, 30, 47], [19, 35, 69], [14, 16, 96]]`
- n_trials: `368`

**Test window**
- BalAcc: `0.4466` | F1m: `0.4341` | Spec: `0.7235`
- CM: `[[889, 548, 968], [572, 752, 1259], [433, 416, 1797]]`

#### Fold 4

- stopped_epoch: `23` | best_epoch: `5`
- Val Acc_paper: `0.4538`
- Val BalAcc_maj: `0.4834`

**Test trial**
- Acc_paper: `0.5509`
- BalAcc_maj: `0.5814`
- F1-macro: `0.5674`
- Rec idle/left/right: `0.8431` / `0.3774` / `0.5238`
- Spec(macro): `0.7936`
- CM: `[[43, 7, 1], [29, 20, 4], [23, 7, 33]]`
- n_trials: `167`

**Test window**
- BalAcc: `0.5379` | F1m: `0.5252` | Spec: `0.7714`
- CM: `[[786, 194, 62], [566, 423, 124], [499, 190, 634]]`

```json
{
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
  "n_times_expected": 500,
  "n_chans": 8,
  "protocol": "2s-hop100ms-balbatch-accpaper-T-only",
  "optimizer": "adamw"
}
```

