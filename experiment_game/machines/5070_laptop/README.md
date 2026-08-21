# 本机专用 · 5070 Laptop（hostname: `cyy`）

> **机位标记**：RTX **5070** Laptop · 仓库根 **`D:\MI`** · Python：**conda `cyy`**  
> **禁止**：为迁就本机去改 `experiment_game/` 根目录给 **5060** 用的 bat（它们写死仓库根 `.venv` / `D:\cyy\MI`）。

## 首次 / 每次检查依赖

```powershell
conda activate cyy
cd D:\MI
python experiment_game\machines\5070_laptop\check_deps.py
```

或双击 / 运行：`check_deps.bat`

## 启动（请用本目录 bat）

| 用途 | 本机入口 |
|------|----------|
| 诱导页 | [`打开诱导页.bat`](./打开诱导页.bat) 或 [`open_induction.bat`](./open_induction.bat) |
| 操作台 | [`open_operator.bat`](./open_operator.bat) |

等价命令：

```powershell
conda activate cyy
cd D:\MI
python -m experiment_game.tools.open_induction
python -m experiment_game.tools.open_operator
```

## 与 5060 的差异

| 项 | 5060（共享根 bat） | 本机 5070 |
|----|-------------------|-----------|
| 仓库 | 常 `D:\cyy\MI` | `D:\MI` |
| Python | `%REPO%\.venv\Scripts\python.exe` | conda **`cyy`** |
| 入口 | `experiment_game\*.bat` | **本文件夹** `machines\5070_laptop\*.bat` |
