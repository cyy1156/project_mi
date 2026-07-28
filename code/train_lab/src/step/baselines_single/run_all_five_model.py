import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent

DATA="merged_2s"
def main():
   for name in ("eegnet", "shallow", "deep", "eegtcnet", "conformer"):
        print(f"===== {name} =====", flush=True)
        r=subprocess.run([sys.executable, str(DIR / f"baseline_{name}.py"),"--data",DATA],cwd=DIR )
        if r.returncode != 0:
            raise SystemExit(f"{name}  failed:{r.returncode}")

   print("all done")

if __name__ == "__main__":
    main()