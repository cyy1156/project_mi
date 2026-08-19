# B8 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_093827_B8`
- Test Acc_paper: `0.5681 ± 0.0234`
- Test BalAcc_maj: `0.5746 ± 0.0233`
- Test win F1: `0.5627 ± 0.0190`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.5744`
- Val BalAcc_maj（附报）：`0.5785`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5445`
- BalAcc_maj：`0.5494`
- Acc_majority：`0.5494`
- F1-macro（众数）：`0.5487`
- Recall-macro：`0.5494`
- Recall idle/left/right：`0.5645` / `0.4773` / `0.6064`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5472` | F1m：`0.5462` | Acc：`0.5472`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    621    212    267
  true1    195    525    380
  true2    226    207    667
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10515   3668   4517
  true1   3375   8749   6576
  true2   3749   3518  11433
```

#### Fold 1

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.5874`
- Val BalAcc_maj（附报）：`0.5937`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5791`
- BalAcc_maj：`0.5839`
- Acc_majority：`0.5839`
- F1-macro（众数）：`0.5805`
- Recall-macro：`0.5839`
- Recall idle/left/right：`0.4945` / `0.7682` / `0.4891`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5750` | F1m：`0.5717` | Acc：`0.5750`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    544    378    178
  true1     75    845    180
  true2    124    438    538
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9130   6325   3245
  true1   1490  14035   3175
  true2   2168   7441   9091
```

#### Fold 2

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.5352`
- Val BalAcc_maj（附报）：`0.5400`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5361`
- BalAcc_maj：`0.5445`
- Acc_majority：`0.5445`
- F1-macro（众数）：`0.5418`
- Recall-macro：`0.5445`
- Recall idle/left/right：`0.5100` / `0.6682` / `0.4555`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5382` | F1m：`0.5357` | Acc：`0.5382`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    561    328    211
  true1    226    735    139
  true2    186    413    501
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9558   5419   3723
  true1   4098  12187   2415
  true2   3284   6970   8446
```

#### Fold 3

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.6026`
- Val BalAcc_maj（附报）：`0.6096`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5952`
- BalAcc_maj：`0.6021`
- Acc_majority：`0.6021`
- F1-macro（众数）：`0.5995`
- Recall-macro：`0.6021`
- Recall idle/left/right：`0.4836` / `0.5955` / `0.7273`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5903` | F1m：`0.5878` | Acc：`0.5903`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    532    289    279
  true1    107    655    338
  true2     93    207    800
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8939   4863   4898
  true1   2096  10901   5703
  true2   1636   3787  13277
```

#### Fold 4

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.5115`
- Val BalAcc_maj（附报）：`0.5170`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5857`
- BalAcc_maj：`0.5930`
- Acc_majority：`0.5930`
- F1-macro（众数）：`0.5885`
- Recall-macro：`0.5930`
- Recall idle/left/right：`0.7540` / `0.5490` / `0.4760`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5762` | F1m：`0.5719` | Acc：`0.5762`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    754    118    128
  true1    302    549    149
  true2    340    184    476
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  12426   2161   2413
  true1   5227   9018   2755
  true2   5811   3245   7944
```
