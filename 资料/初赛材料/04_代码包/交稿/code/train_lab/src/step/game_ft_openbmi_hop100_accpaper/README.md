# game_ft_openbmi_hop100_accpaper

旁路 **05**：OpenBMI Acc_paper shallow init → 通道重排 → 前半全模型微调 → 后半伪在线评。

- 方案：`资料/伪在线实验/05_旁路_OpenBMI_前半微调后半评/`
- 对照：`game_ft_hop100_accpaper`（02 · BCI2a init）
- 通道：复用 `game_pseudo_online_hop100/channel_remap.py`

```bash
PY=D:/cyy/MI/.venv/Scripts/python.exe
$PY build_splits.py
$PY baseline_shallow.py
```
