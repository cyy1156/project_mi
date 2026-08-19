# B7 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_073937_B7`
- Test Acc_paper: `0.5635 ± 0.0153`
- Test BalAcc_maj: `0.5698 ± 0.0153`
- Test win F1: `0.5593 ± 0.0137`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.5737`
- Val BalAcc_maj（附报）：`0.5785`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5427`
- BalAcc_maj：`0.5494`
- Acc_majority：`0.5494`
- F1-macro（众数）：`0.5494`
- Recall-macro：`0.5494`
- Recall idle/left/right：`0.5818` / `0.5191` / `0.5473`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5429` | F1m：`0.5429` | Acc：`0.5429`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    640    239    221
  true1    205    571    324
  true2    243    255    602
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10687   4283   3730
  true1   3620   9592   5488
  true2   4147   4376  10177
```

#### Fold 1

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.5793`
- Val BalAcc_maj（附报）：`0.5859`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5809`
- BalAcc_maj：`0.5873`
- Acc_majority：`0.5873`
- F1-macro（众数）：`0.5853`
- Recall-macro：`0.5873`
- Recall idle/left/right：`0.4873` / `0.7618` / `0.5127`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5816` | F1m：`0.5791` | Acc：`0.5816`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    536    391    173
  true1     66    838    196
  true2    100    436    564
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8997   6576   3127
  true1   1304  14070   3326
  true2   1869   7270   9561
```

#### Fold 2

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.5400`
- Val BalAcc_maj（附报）：`0.5456`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5479`
- BalAcc_maj：`0.5536`
- Acc_majority：`0.5536`
- F1-macro（众数）：`0.5534`
- Recall-macro：`0.5536`
- Recall idle/left/right：`0.5464` / `0.5936` / `0.5209`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5448` | F1m：`0.5446` | Acc：`0.5448`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    601    240    259
  true1    248    653    199
  true2    192    335    573
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10028   4268   4404
  true1   4294  10850   3556
  true2   3392   5625   9683
```

#### Fold 3

- stopped_epoch：`78` | best_epoch：`58`
- Val Acc_paper（早停）：`0.6085`
- Val BalAcc_maj（附报）：`0.6159`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5712`
- BalAcc_maj：`0.5770`
- Acc_majority：`0.5770`
- F1-macro（众数）：`0.5748`
- Recall-macro：`0.5770`
- Recall idle/left/right：`0.5527` / `0.4964` / `0.6818`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5678` | F1m：`0.5661` | Acc：`0.5678`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    608    226    266
  true1    219    546    335
  true2    158    192    750
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10160   4022   4518
  true1   3682   9282   5736
  true2   2767   3519  12414
```

#### Fold 4

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.5107`
- Val BalAcc_maj（附报）：`0.5152`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5750`
- BalAcc_maj：`0.5817`
- Acc_majority：`0.5817`
- F1-macro（众数）：`0.5804`
- Recall-macro：`0.5817`
- Recall idle/left/right：`0.6750` / `0.5250` / `0.5450`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5649` | F1m：`0.5635` | Acc：`0.5649`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    675    155    170
  true1    289    525    186
  true2    285    170    545
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11170   2677   3153
  true1   4893   8704   3403
  true2   4968   3094   8938
```
