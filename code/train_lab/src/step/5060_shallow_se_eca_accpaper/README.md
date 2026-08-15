# 5060_shallow_se_eca_accpaper · 方案 14

旁路：Shallow 输入端前置 **SE** / **ECA**，协议同正式 Acc_paper；不改正式表。

## 入口

```bat
cd code\train_lab\src\step\5060_shallow_se_eca_accpaper

python run_arm.py --arm A0 --max-epochs 2 --patience 2
python run_arm.py --arm B0 --max-epochs 2 --patience 2
python run_arm.py --arm A1
python run_arm.py --arm B1
python run_arm.py --arm A2
python run_arm.py --arm B2
python run_arm.py --arm S0
```

| 臂 | 含义 |
|----|------|
| S0 | 原版 Shallow（五折） |
| A0 / B0 | SE / ECA 冒烟（fold0；建议 `--max-epochs 2`） |
| A1 / B1 | SE / ECA · fold0 |
| A2 / B2 | SE / ECA · 五折 |

## 路径

- out：`code/train_lab/out/5060_shallow_se_eca_accpaper/`
- 方案：`资料/模型训练/14_旁路_shallow_前置SE_ECA_openbmi_accpaper/`
