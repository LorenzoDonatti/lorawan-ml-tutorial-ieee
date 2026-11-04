"""Run the teaching notebooks with this interpreter, without inherited venv paths."""
from pathlib import Path
import os
import subprocess
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    # IPython otherwise adds VIRTUAL_ENV/site-packages even with an explicit Python.
    for key in ['VIRTUAL_ENV', 'PYTHONPATH']:
        environment.pop(key, None)
    environment.setdefault('OPENBLAS_NUM_THREADS', '2')
    environment.setdefault('OMP_NUM_THREADS', '2')
    raise SystemExit(subprocess.call(
        [sys.executable, str(root/'scripts/execute_notebooks.py'), *sys.argv[1:]],
        cwd=root, env=environment))
