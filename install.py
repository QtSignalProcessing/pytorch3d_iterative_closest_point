import os
from pathlib import Path
from sys import platform
import shutil

def delete_lib(pattern):
    pyd = [x for x in Path('.').glob('*.' + pattern)]
    pyd = pyd[0] if len(pyd) == 1 else pyd
    if isinstance(pyd, Path) and pyd.exists():
        pyd.unlink()


egg_f = [x for x in Path('.').glob('*.egg-info')]
egg_f = egg_f[0] if len(egg_f) > 0 else egg_f
if isinstance(egg_f, Path) and egg_f.exists():
    shutil.rmtree(egg_f)
if platform == 'win32':
    delete_lib('pyd')
elif platform == 'linux':
    delete_lib('so')

os.system('python ./knn/setup.py develop')