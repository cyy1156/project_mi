# A2_pt · scheme21

Test Acc_paper: `0.5701 ± 0.0252`
Test BalAcc_maj: `0.5753 ± 0.0251`
Test win F1: `0.5671 ± 0.0238`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.5759`
- Val BalAcc_maj（附报）：`0.5841`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5436`
- BalAcc_maj：`0.5473`
- Acc_majority：`0.5473`
- F1-macro（众数）：`0.5474`
- Recall-macro：`0.5473`
- Recall idle/left/right：`0.5509` / `0.5782` / `0.5127`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5435` | F1m：`0.5437` | Acc：`0.5435`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    606    290    204
  true1    186    636    278
  true2    230    306    564
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10259   4888   3553
  true1   3251  10664   4785
  true2   3949   5183   9568
```

#### Fold 1

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5722`
- Val BalAcc_maj（附报）：`0.5796`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5982`
- BalAcc_maj：`0.6030`
- Acc_majority：`0.6030`
- F1-macro（众数）：`0.6044`
- Recall-macro：`0.6030`
- Recall idle/left/right：`0.5518` / `0.6700` / `0.5873`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5926` | F1m：`0.5937` | Acc：`0.5926`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    607    291    202
  true1     86    737    277
  true2    129    325    646
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10143   4847   3710
  true1   1703  12308   4689
  true2   2253   5654  10793
```

#### Fold 2

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5400`
- Val BalAcc_maj（附报）：`0.5474`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5364`
- BalAcc_maj：`0.5433`
- Acc_majority：`0.5433`
- F1-macro（众数）：`0.5413`
- Recall-macro：`0.5433`
- Recall idle/left/right：`0.6091` / `0.5764` / `0.4445`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5362` | F1m：`0.5343` | Acc：`0.5362`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    670    247    183
  true1    320    634    146
  true2    265    346    489
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11182   4386   3132
  true1   5450  10648   2602
  true2   4493   5958   8249
```

#### Fold 3

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.6041`
- Val BalAcc_maj（附报）：`0.6085`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5903`
- BalAcc_maj：`0.5958`
- Acc_majority：`0.5958`
- F1-macro（众数）：`0.5946`
- Recall-macro：`0.5958`
- Recall idle/left/right：`0.4827` / `0.6464` / `0.6582`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5883` | F1m：`0.5873` | Acc：`0.5883`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    531    347    222
  true1    125    711    264
  true2     94    282    724
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9058   5666   3976
  true1   2252  11908   4540
  true2   1726   4938  12036
```

#### Fold 4

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.5130`
- Val BalAcc_maj（附报）：`0.5163`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5820`
- BalAcc_maj：`0.5873`
- Acc_majority：`0.5873`
- F1-macro（众数）：`0.5865`
- Recall-macro：`0.5873`
- Recall idle/left/right：`0.6460` / `0.5850` / `0.5310`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5770` | F1m：`0.5762` | Acc：`0.5770`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    646    182    172
  true1    235    585    180
  true2    245    224    531
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10721   3149   3130
  true1   4056   9803   3141
  true2   4245   3851   8904
```
