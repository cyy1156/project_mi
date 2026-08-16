# 5-fold experiment record (20260814_174520 / standalone_shallow_2s_hop100_balbatch_accpaper_adam)

- start: `2026-08-14T17:45:20`
- device: `cpu`
- data: `D:\360MoveData\Users\ckgxnn\Desktop\MI\code\preprocess_lab\out\bci2a_2s_hop100` (BCI2a T / hop100 only)
- protocol: `2s-hop100ms-balbatch-accpaper-T-only` | early_stop=**Acc_paper** | **balbatch**
- optimizer: `adam` | lr=`0.0001` | weight_decay=`0.0001`
- model: `standalone ShallowFBCSPNet` | no braindecode
- Train: window CE + batch balance; Val/Test: trial Acc_paper
- pipeline: Task(2-class) then Three(3-class) (skip-three)
- weights: `D:\360MoveData\Users\ckgxnn\Desktop\MI\self_model\out\optimizer_compare_smoke\adam`
- shared hp: `{'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 2, 'patience': 2, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'n_times_expected': 500, 'n_chans': 8, 'protocol': '2s-hop100ms-balbatch-accpaper-T-only', 'optimizer': 'adam'}`

---

### Task
`0.5076` / `0.6379`

### Task fold details

#### Fold 0

- stopped_epoch: `2` | best_epoch: `1`
- Val Acc_paper: `0.7879`
- Val BalAcc_maj: `0.7413`

**Test trial**
- Acc_paper: `0.5960`
- BalAcc_maj: `0.5742`
- Acc_majority: `0.5960`
- Sens/Spec/F1: `0.6367` / `0.5116` / `0.6800`
- CM: `[[66, 63], [97, 170]]`
- n_trials: `396`

**Test window**
- BalAcc: `0.5421` | F1: `0.6379` | Sens: `0.5766` | Spec: `0.5076`
- CM: `[[1336, 1296], [2374, 3233]]`

### Three
- (skipped this run)

```json
{
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 2,
  "patience": 2,
  "batch_train": 32,
  "batch_eval": 64,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "n_times_expected": 500,
  "n_chans": 8,
  "protocol": "2s-hop100ms-balbatch-accpaper-T-only",
  "optimizer": "adam"
}
```

