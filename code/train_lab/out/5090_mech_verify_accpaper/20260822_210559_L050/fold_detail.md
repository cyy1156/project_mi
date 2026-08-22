### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`56` | best_epoch：`36`
- Val Acc_paper（早停）：`0.5711`
- Val BalAcc_maj（附报）：`0.5752`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5461`
- BalAcc_maj：`0.5524`
- Acc_majority：`0.5524`
- F1-macro（众数）：`0.5523`
- Recall-macro：`0.5524`
- Recall idle/left/right：`0.5727` / `0.5109` / `0.5736`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5466` | F1m：`0.5465` | Acc：`0.5466`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    630    232    238
  true1    210    562    328
  true2    233    236    631
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10638   4132   3930
  true1   3715   9585   5400
  true2   4191   4067  10442
```

#### Fold 1

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.5752`
- Val BalAcc_maj（附报）：`0.5800`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5942`
- BalAcc_maj：`0.6003`
- Acc_majority：`0.6003`
- F1-macro（众数）：`0.6005`
- Recall-macro：`0.6003`
- Recall idle/left/right：`0.5164` / `0.6827` / `0.6018`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5891` | F1m：`0.5891` | Acc：`0.5891`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    568    285    247
  true1     77    751    272
  true2    123    315    662
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9536   4859   4305
  true1   1624  12459   4617
  true2   2182   5466  11052
```

#### Fold 2

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5441`
- Val BalAcc_maj（附报）：`0.5519`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5527`
- BalAcc_maj：`0.5591`
- Acc_majority：`0.5591`
- F1-macro（众数）：`0.5587`
- Recall-macro：`0.5591`
- Recall idle/left/right：`0.5364` / `0.6118` / `0.5291`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5506` | F1m：`0.5503` | Acc：`0.5506`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    590    244    266
  true1    230    673    197
  true2    177    341    582
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10015   4200   4485
  true1   4007  11147   3546
  true2   3234   5741   9725
```

#### Fold 3

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.5970`
- Val BalAcc_maj（附报）：`0.6022`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5945`
- BalAcc_maj：`0.6000`
- Acc_majority：`0.6000`
- F1-macro（众数）：`0.5976`
- Recall-macro：`0.6000`
- Recall idle/left/right：`0.4855` / `0.5991` / `0.7155`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5906` | F1m：`0.5885` | Acc：`0.5906`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    534    300    266
  true1    118    659    323
  true2     92    221    787
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9038   5071   4591
  true1   2154  11009   5537
  true2   1672   3944  13084
```

#### Fold 4

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5163`
- Val BalAcc_maj（附报）：`0.5226`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5850`
- BalAcc_maj：`0.5927`
- Acc_majority：`0.5927`
- F1-macro（众数）：`0.5909`
- Recall-macro：`0.5927`
- Recall idle/left/right：`0.6880` / `0.5740` / `0.5160`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5773` | F1m：`0.5756` | Acc：`0.5773`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    688    151    161
  true1    268    574    158
  true2    275    209    516
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11424   2618   2958
  true1   4651   9438   2911
  true2   4819   3600   8581
```
