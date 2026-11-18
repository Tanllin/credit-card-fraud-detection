"""Run the project from a terminal with the same Python used for installation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    'eda': '01_EDA_Preprocessing.ipynb',
    'models': '02_Model_Selection.ipynb',
    'imbalance': '03_Class_Imbalance.ipynb',
    'thresholds': 'scripts/analyze_xgboost_thresholds.py',
    'evaluate': '04_Final_Test_Evaluation.ipynb',
}


def run_stage(name):
    os.chdir(ROOT)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    (ROOT / 'Data' / 'processed').mkdir(parents=True, exist_ok=True)
    path = ROOT / STAGES[name]
    if path.suffix == '.py':
        subprocess.run([sys.executable, str(path)], check=True)
        return
    namespace = {'__name__': '__main__'}
    for number, cell in enumerate(json.loads(path.read_text())['cells'], start=1):
        if cell['cell_type'] == 'code':
            print(f'{path.name}, cell {number}', flush=True)
            source = ''.join(cell['source'])
            exec(compile(source, f'{path.name} [cell {number}]', 'exec'), namespace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='all', choices=['all', *STAGES])
    stage = parser.parse_args().stage
    if stage == 'all':
        for name in STAGES:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), name], check=True)
    else:
        run_stage(stage)
