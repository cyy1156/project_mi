# 机位模板

复制本目录为 `machines/<新机名>/`，然后：

1. 改 `machine.json`：`machine_id` / `hostname` / `gpu` / `repo_root` / `conda_env`
2. 如需其它 conda 名：在 bat 前 `set MACHINE_CONDA_ENV=你的环境`
3. 跑 `check_deps.bat`（调用共享 `python -m experiment_game.tools.preflight`）
4. 冒烟通过后在 [`../VERIFIED.md`](../VERIFIED.md) 登记一行

**禁止**在 `_resolve_python.bat` 里写死 `C:\Users\<别人>\...`。
