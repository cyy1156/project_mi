# T1_128 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260821_152002_T1_128`
- Test Acc_paper: `0.5708 ± 0.0205`
- Test BalAcc_maj: `0.5766 ± 0.0205`
- Test win F1: `0.5634 ± 0.0184`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5830`
- Val BalAcc_maj（附报）：`0.5874`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5503`
- BalAcc_maj：`0.5545`
- Acc_majority：`0.5545`
- F1-macro（众数）：`0.5540`
- Recall-macro：`0.5545`
- Recall idle/left/right：`0.5436` / `0.6236` / `0.4964`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5463` | F1m：`0.5459` | Acc：`0.5463`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    598    328    174
  true1    181    686    233
  true2    223    331    546
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9951   5645   3104
  true1   3160  11430   4110
  true2   3747   5688   9265
```

#### Fold 1

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5856`
- Val BalAcc_maj（附报）：`0.5922`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5836`
- BalAcc_maj：`0.5894`
- Acc_majority：`0.5894`
- F1-macro（众数）：`0.5886`
- Recall-macro：`0.5894`
- Recall idle/left/right：`0.4891` / `0.7136` / `0.5655`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5786` | F1m：`0.5773` | Acc：`0.5786`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    538    348    214
  true1     73    785    242
  true2    109    369    622
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8995   5907   3798
  true1   1475  13121   4104
  true2   2075   6283  10342
```

#### Fold 2

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.5474`
- Val BalAcc_maj（附报）：`0.5526`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5418`
- BalAcc_maj：`0.5488`
- Acc_majority：`0.5488`
- F1-macro（众数）：`0.5461`
- Recall-macro：`0.5488`
- Recall idle/left/right：`0.4818` / `0.6900` / `0.4745`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5390` | F1m：`0.5365` | Acc：`0.5390`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    530    361    209
  true1    190    759    151
  true2    144    434    522
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8809   6203   3688
  true1   3431  12597   2672
  true2   2553   7313   8834
```

#### Fold 3

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.5974`
- Val BalAcc_maj（附报）：`0.6022`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5891`
- BalAcc_maj：`0.5955`
- Acc_majority：`0.5955`
- F1-macro（众数）：`0.5899`
- Recall-macro：`0.5955`
- Recall idle/left/right：`0.4436` / `0.5673` / `0.7755`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5838` | F1m：`0.5789` | Acc：`0.5838`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    488    274    338
  true1     90    624    386
  true2     67    180    853
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8352   4619   5729
  true1   1795  10290   6615
  true2   1338   3254  14108
```

#### Fold 4

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5193`
- Val BalAcc_maj（附报）：`0.5259`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5893`
- BalAcc_maj：`0.5947`
- Acc_majority：`0.5947`
- F1-macro（众数）：`0.5937`
- Recall-macro：`0.5947`
- Recall idle/left/right：`0.6510` / `0.6030` / `0.5300`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5795` | F1m：`0.5786` | Acc：`0.5795`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    651    198    151
  true1    212    603    185
  true2    220    250    530
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10772   3431   2797
  true1   3802   9919   3279
  true2   3960   4176   8864
```
