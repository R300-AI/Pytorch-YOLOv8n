import shutil
from pathlib import Path

def Make_Directory(dir):
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir(parents=True, exist_ok=True)
    return dir