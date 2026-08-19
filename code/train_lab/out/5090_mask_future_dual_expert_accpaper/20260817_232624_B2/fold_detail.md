# B2 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260817_232624_B2`
- Test Acc_paper: `0.5640 ± 0.0227`
- Test BalAcc_maj: `0.5696 ± 0.0217`
- Test win F1: `0.5571 ± 0.0200`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`46` | best_epoch：`26`
- Val Acc_paper（早停）：`0.5737`
- Val BalAcc_maj（附报）：`0.5781`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5455`
- BalAcc_maj：`0.5542`
- Acc_majority：`0.5542`
- F1-macro（众数）：`0.5543`
- Recall-macro：`0.5542`
- Recall idle/left/right：`0.5727` / `0.5118` / `0.5782`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5445` | F1m：`0.5442` | Acc：`0.5445`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    630    218    252
  true1    191    563    346
  true2    246    218    636
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10504   3902   4294
  true1   3476   9236   5988
  true2   4090   3805  10805
```

#### Fold 1

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.5848`
- Val BalAcc_maj（附报）：`0.5930`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5882`
- BalAcc_maj：`0.5918`
- Acc_majority：`0.5918`
- F1-macro（众数）：`0.5907`
- Recall-macro：`0.5918`
- Recall idle/left/right：`0.4918` / `0.7209` / `0.5627`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5799` | F1m：`0.5789` | Acc：`0.5799`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    541    347    212
  true1     69    793    238
  true2    118    363    619
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9048   6013   3639
  true1   1486  13079   4135
  true2   2087   6207  10406
```

#### Fold 2

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.5400`
- Val BalAcc_maj（附报）：`0.5467`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5294`
- BalAcc_maj：`0.5348`
- Acc_majority：`0.5348`
- F1-macro（众数）：`0.5345`
- Recall-macro：`0.5348`
- Recall idle/left/right：`0.5418` / `0.5836` / `0.4791`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5260` | F1m：`0.5256` | Acc：`0.5260`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    596    276    228
  true1    278    642    180
  true2    190    383    527
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9887   4891   3922
  true1   4770  10794   3136
  true2   3432   6438   8830
```

#### Fold 3

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.6096`
- Val BalAcc_maj（附报）：`0.6144`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5827`
- BalAcc_maj：`0.5879`
- Acc_majority：`0.5879`
- F1-macro（众数）：`0.5841`
- Recall-macro：`0.5879`
- Recall idle/left/right：`0.4764` / `0.5618` / `0.7255`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5789` | F1m：`0.5758` | Acc：`0.5789`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    524    277    299
  true1    158    618    324
  true2    109    193    798
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8994   4707   4999
  true1   2783  10246   5671
  true2   1944   3518  13238
```

#### Fold 4

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.5204`
- Val BalAcc_maj（附报）：`0.5285`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5740`
- BalAcc_maj：`0.5790`
- Acc_majority：`0.5790`
- F1-macro（众数）：`0.5769`
- Recall-macro：`0.5790`
- Recall idle/left/right：`0.6880` / `0.5560` / `0.4930`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5629` | F1m：`0.5610` | Acc：`0.5629`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    688    159    153
  true1    290    556    154
  true2    309    198    493
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11296   2942   2762
  true1   4995   9195   2810
  true2   5279   3502   8219
```
