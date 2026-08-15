# 5060 · Shallow 网络结构增强 · Three（方案 09 旁路）

> 方案：`资料/模型训练/09_旁路_shallow_网络结构增强_Three_openbmi_accpaper/方案.md`  
> **非正式表**；不覆盖 `run_20260807_135828`。

## 跑法

```powershell
cd D:\cyy\MI\code\train_lab\src\step\5060_shallow_net_enhance_three_accpaper
$env:PYTHONUNBUFFERED = "1"

# S0 复现锚点（五折）
python -W ignore run_arm.py --arm S0

# S1a 核长
python -W ignore run_arm.py --arm S1a_t13
python -W ignore run_arm.py --arm S1a_t50

# 冒烟
python -W ignore run_arm.py --arm S0 --max-folds 1 --max-epochs 40 --patience 10
```

Out：`code/train_lab/out/5060_shallow_net_enhance_three_accpaper/`  
Runs MD：`资料/模型训练/runs/5060_shallow_net_enhance_three/`
