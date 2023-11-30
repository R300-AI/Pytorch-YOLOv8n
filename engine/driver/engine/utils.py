from pathlib import Path
import shutil

def make_dirs(dir):
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir(parents=True, exist_ok=True)
    return dir