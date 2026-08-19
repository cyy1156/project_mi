# U12 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260819_191327_U12`
- Test Acc_paper: `0.5650 ± 0.0222`
- Test BalAcc_maj: `0.5704 ± 0.0227`
- Test win F1: `0.5587 ± 0.0207`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.5715`
- Val BalAcc_maj（附报）：`0.5778`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5467`
- BalAcc_maj：`0.5527`
- Acc_majority：`0.5527`
- F1-macro（众数）：`0.5529`
- Recall-macro：`0.5527`
- Recall idle/left/right：`0.5627` / `0.5736` / `0.5218`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5451` | F1m：`0.5453` | Acc：`0.5451`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    619    268    213
  true1    179    631    290
  true2    244    282    574
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10292   4702   3706
  true1   3370  10474   4856
  true2   4021   4866   9813
```

#### Fold 1

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5833`
- Val BalAcc_maj（附报）：`0.5919`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5776`
- BalAcc_maj：`0.5818`
- Acc_majority：`0.5818`
- F1-macro（众数）：`0.5800`
- Recall-macro：`0.5818`
- Recall idle/left/right：`0.4536` / `0.6955` / `0.5964`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5707` | F1m：`0.5686` | Acc：`0.5707`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    499    376    225
  true1     74    765    261
  true2     98    346    656
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8321   6266   4113
  true1   1471  12680   4549
  true2   1815   5872  11013
```

#### Fold 2

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.5411`
- Val BalAcc_maj（附报）：`0.5481`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5318`
- BalAcc_maj：`0.5358`
- Acc_majority：`0.5358`
- F1-macro（众数）：`0.5346`
- Recall-macro：`0.5358`
- Recall idle/left/right：`0.5518` / `0.5936` / `0.4618`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5276` | F1m：`0.5266` | Acc：`0.5276`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    607    277    216
  true1    258    653    189
  true2    224    368    508
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10137   4748   3815
  true1   4553  10895   3252
  true2   3839   6295   8566
```

#### Fold 3

- stopped_epoch：`40` | best_epoch：`20`
- Val Acc_paper（早停）：`0.5948`
- Val BalAcc_maj（附报）：`0.6015`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5921`
- BalAcc_maj：`0.5979`
- Acc_majority：`0.5979`
- F1-macro（众数）：`0.5962`
- Recall-macro：`0.5979`
- Recall idle/left/right：`0.5027` / `0.6027` / `0.6882`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5877` | F1m：`0.5864` | Acc：`0.5877`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    553    297    250
  true1    152    663    285
  true2    114    229    757
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9332   5088   4280
  true1   2626  11190   4884
  true2   2130   4123  12447
```

#### Fold 4

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5107`
- Val BalAcc_maj（附报）：`0.5174`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5770`
- BalAcc_maj：`0.5840`
- Acc_majority：`0.5840`
- F1-macro（众数）：`0.5826`
- Recall-macro：`0.5840`
- Recall idle/left/right：`0.6720` / `0.5770` / `0.5030`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5680` | F1m：`0.5664` | Acc：`0.5680`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    672    174    154
  true1    281    577    142
  true2    289    208    503
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11111   3124   2765
  true1   4797   9517   2686
  true2   4986   3676   8338
```
