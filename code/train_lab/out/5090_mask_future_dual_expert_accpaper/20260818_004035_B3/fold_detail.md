# B3 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_004035_B3`
- Test Acc_paper: `0.5691 ± 0.0247`
- Test BalAcc_maj: `0.5742 ± 0.0252`
- Test win F1: `0.5646 ± 0.0231`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`56` | best_epoch：`36`
- Val Acc_paper（早停）：`0.5759`
- Val BalAcc_maj（附报）：`0.5804`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5412`
- BalAcc_maj：`0.5448`
- Acc_majority：`0.5448`
- F1-macro（众数）：`0.5449`
- Recall-macro：`0.5448`
- Recall idle/left/right：`0.5555` / `0.5318` / `0.5473`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5375` | F1m：`0.5376` | Acc：`0.5375`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    611    269    220
  true1    205    585    310
  true2    258    240    602
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10177   4720   3803
  true1   3728   9721   5251
  true2   4220   4223  10257
```

#### Fold 1

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5774`
- Val BalAcc_maj（附报）：`0.5859`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5906`
- BalAcc_maj：`0.5958`
- Acc_majority：`0.5958`
- F1-macro（众数）：`0.5960`
- Recall-macro：`0.5958`
- Recall idle/left/right：`0.5518` / `0.6800` / `0.5555`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5856` | F1m：`0.5856` | Acc：`0.5856`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    607    299    194
  true1    105    748    247
  true2    154    335    611
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10029   5184   3487
  true1   2034  12510   4156
  true2   2686   5699  10315
```

#### Fold 2

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.5370`
- Val BalAcc_maj（附报）：`0.5444`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5373`
- BalAcc_maj：`0.5421`
- Acc_majority：`0.5421`
- F1-macro（众数）：`0.5419`
- Recall-macro：`0.5421`
- Recall idle/left/right：`0.5100` / `0.5682` / `0.5482`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5374` | F1m：`0.5373` | Acc：`0.5374`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    561    252    287
  true1    263    625    212
  true2    184    313    603
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9592   4291   4817
  true1   4447  10576   3677
  true2   3207   5512   9981
```

#### Fold 3

- stopped_epoch：`65` | best_epoch：`45`
- Val Acc_paper（早停）：`0.6067`
- Val BalAcc_maj（附报）：`0.6115`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5936`
- BalAcc_maj：`0.5982`
- Acc_majority：`0.5982`
- F1-macro（众数）：`0.5981`
- Recall-macro：`0.5982`
- Recall idle/left/right：`0.5382` / `0.6109` / `0.6455`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5913` | F1m：`0.5913` | Acc：`0.5913`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    592    314    194
  true1    160    672    268
  true2    147    243    710
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9979   5098   3623
  true1   2737  11299   4664
  true2   2471   4336  11893
```

#### Fold 4

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5074`
- Val BalAcc_maj（附报）：`0.5130`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5827`
- BalAcc_maj：`0.5900`
- Acc_majority：`0.5900`
- F1-macro（众数）：`0.5868`
- Recall-macro：`0.5900`
- Recall idle/left/right：`0.7110` / `0.5720` / `0.4870`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5737` | F1m：`0.5713` | Acc：`0.5737`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    711    156    133
  true1    262    572    166
  true2    300    213    487
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11536   2903   2561
  true1   4545   9505   2950
  true2   5146   3635   8219
```
