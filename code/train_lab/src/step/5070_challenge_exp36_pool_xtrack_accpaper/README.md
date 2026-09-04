# 实验 36 · 代码入口

> 方案：`资料/模型训练/36_旁路_官方主交卷_扩池与跨轨融合_accpaper/方案.md`  
> out：`code/train_lab/out/5070_challenge_exp36_pool_xtrack_accpaper/`

## Day0（零训练）

```powershell
conda activate cyy
cd D:\MI\code\train_lab\src\step\5070_challenge_exp36_pool_xtrack_accpaper
python run_exp36.py --stage day0 --update-registry
```

臂：**S0**（复现）· **M7**（A59×B8-ft，w_B8≤0.4）· **MC0**（nested 边际校正）· **M7c**（可选）。

## Day2（C1 · 45ch）

```powershell
python train_c1.py --stage all --run-tag day2_YYYYMMDD
python replay_day2.py
```

数据：`challenge_mi_3s_45ch`（59→45 切片）· `openbmi_3s_fixed_45ch`（固定 3s）。  
臂：C1 单模 · M7_AC · M7_ABC。判定同 Day0。

判定：Val≥0.568 且 Wilcoxon p&lt;0.05 vs S0。
