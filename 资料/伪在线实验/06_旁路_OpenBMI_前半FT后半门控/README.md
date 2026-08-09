# 06 · OpenBMI 前半 FT · 后半 + 在线门控（旁路）

在 [05](../05_旁路_OpenBMI_前半微调后半评/) 冻结 FT 权重上，对后半试次做 H0–H3 在线门控评测（协议同 03/04）。  
状态：**已结案**（见 [`总结/结果登记表.md`](总结/结果登记表.md)）。

> 注意：本臂是**伪在线 06**，不是 [`模型训练/06_旁路_可教试次…`](../../模型训练/06_旁路_可教试次_子集评估_微调_openbmi_accpaper/)。

| 文件 | 说明 |
|------|------|
| [方案.md](方案.md) | 冻结范围 |
| [总结/结果登记表.md](总结/结果登记表.md) | 主报 |

```bash
PY=D:/cyy/MI/.venv/Scripts/python.exe
cd code/train_lab/src/step/game_pseudo_online_hop100
$PY eval_ft_gated.py --gates H0,H1,H2,H3
```
