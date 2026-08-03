# 阈值扫描（20260801_181523）

- run_dir：`D:\cyy\MI\code\train_lab\out\baseline\shallow_wce2_balacc\merged_2s\run_20260801_153617`
- data：`merged_2s`
- 选 τ 规则：val 上优先 Spec≥0.4 且 Rec≥0.75，取 BalAcc 最大

## 五折汇总（Test）

- **τ=0.5（训练默认）**：Spec `0.3603±0.1337` | Rec `0.7699±0.1093` | BalAcc `0.5651±0.0390`
- **val 选定 τ 后**：τ `0.47±0.05` | Spec `0.2953±0.1361` | Rec `0.8164±0.1228` | BalAcc `0.5559±0.0311`
- val 存在双过关 τ 的折数：`0/5`
- test 曲线上存在双过关 τ 的折数：`1/5`

## 各折

### Fold 0
- val 选 τ=`0.40` (val Spec=0.3689 Rec=0.8333 BalAcc=0.6011)
- test@选中τ：Spec=`0.2867` Rec=`0.8255` BalAcc=`0.5561` F1=`0.7808`
- test@0.5：Spec=`0.5191` Rec=`0.5994` BalAcc=`0.5593`
- val 双过关τ：`False` | test 曲线双过关τ：`False`

### Fold 1
- val 选 τ=`0.55` (val Spec=0.2972 Rec=0.8161 BalAcc=0.5567)
- test@选中τ：Spec=`0.5357` Rec=`0.5772` BalAcc=`0.5564` F1=`0.6561`
- test@0.5：Spec=`0.4520` Rec=`0.6820` BalAcc=`0.5670`
- val 双过关τ：`False` | test 曲线双过关τ：`False`

### Fold 2
- val 选 τ=`0.45` (val Spec=0.3602 Rec=0.8004 BalAcc=0.5803)
- test@选中τ：Spec=`0.3177` Rec=`0.9045` BalAcc=`0.6111` F1=`0.8467`
- test@0.5：Spec=`0.4289` Rec=`0.8448` BalAcc=`0.6369`
- val 双过关τ：`False` | test 曲线双过关τ：`True`

### Fold 3
- val 选 τ=`0.45` (val Spec=0.3730 Rec=0.7737 BalAcc=0.5734)
- test@选中τ：Spec=`0.1366` Rec=`0.8979` BalAcc=`0.5172` F1=`0.7868`
- test@0.5：Spec=`0.2016` Rec=`0.8460` BalAcc=`0.5238`
- val 双过关τ：`False` | test 曲线双过关τ：`False`

### Fold 4
- val 选 τ=`0.50` (val Spec=0.3898 Rec=0.7982 BalAcc=0.5940)
- test@选中τ：Spec=`0.1997` Rec=`0.8772` BalAcc=`0.5384` F1=`0.7947`
- test@0.5：Spec=`0.1997` Rec=`0.8772` BalAcc=`0.5384`
- val 双过关τ：`False` | test 曲线双过关τ：`False`

