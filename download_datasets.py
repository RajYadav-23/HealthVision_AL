"""
Run this script once to download both datasets using Kaggle CLI.

Requirements:
  1. pip install kaggle
  2. Place kaggle.json in C:\\Users\\<YourName>\\.kaggle\\kaggle.json
     Get it from: https://www.kaggle.com/settings → API → Create New Token
"""

import os
import subprocess
import sys


def run(cmd: str) -> None:
    print(f'\n>> {cmd}')
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f'ERROR: command failed with code {result.returncode}')
        sys.exit(1)


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    # ── 1. Install kaggle if missing ───────────────────────────
    try:
        import kaggle
    except ImportError:
        print('Installing kaggle...')
        run(f'{sys.executable} -m pip install kaggle')

    # ── 2. ANN Dataset – Pima Indians Diabetes ─────────────────
    ann_dir = os.path.join(base, 'datasets')
    os.makedirs(ann_dir, exist_ok=True)

    diabetes_csv = os.path.join(ann_dir, 'diabetes.csv')
    if os.path.exists(diabetes_csv):
        print(f'\n[SKIP] diabetes.csv already exists at {diabetes_csv}')
    else:
        print('\n[DOWNLOADING] Pima Indians Diabetes Dataset...')
        run(f'kaggle datasets download -d uciml/pima-indians-diabetes-database '
            f'-p "{ann_dir}" --unzip')
        if os.path.exists(diabetes_csv):
            print(f'[OK] diabetes.csv saved to {diabetes_csv}')
        else:
            print('[WARN] diabetes.csv not found after download. Check the datasets folder.')

    # ── 3. CNN Dataset – Brain Tumor MRI ───────────────────────
    cnn_dir = os.path.join(base, 'datasets', 'brain_tumor')
    os.makedirs(cnn_dir, exist_ok=True)

    # Check if already downloaded (Training folder exists)
    train_dir = os.path.join(cnn_dir, 'Training')
    if os.path.exists(train_dir):
        print(f'\n[SKIP] Brain Tumor dataset already exists at {cnn_dir}')
    else:
        print('\n[DOWNLOADING] Brain Tumor MRI Dataset (~150MB)...')
        run(f'kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset '
            f'-p "{cnn_dir}" --unzip')

        # Handle nested folder if kaggle extracts into a subfolder
        for item in os.listdir(cnn_dir):
            item_path = os.path.join(cnn_dir, item)
            if os.path.isdir(item_path) and item not in ('Training', 'Testing'):
                # Move contents up one level
                for sub in os.listdir(item_path):
                    src = os.path.join(item_path, sub)
                    dst = os.path.join(cnn_dir, sub)
                    if not os.path.exists(dst):
                        import shutil
                        shutil.move(src, dst)
                        print(f'  Moved {sub} → {cnn_dir}')

        if os.path.exists(train_dir):
            classes = os.listdir(train_dir)
            counts = {c: len(os.listdir(os.path.join(train_dir, c))) for c in classes}
            print(f'[OK] Brain Tumor dataset saved to {cnn_dir}')
            print(f'     Classes: {counts}')
        else:
            print('[WARN] Training folder not found. Check datasets/brain_tumor/ manually.')

    # ── Summary ────────────────────────────────────────────────
    print('\n' + '='*55)
    print(' DATASET SETUP COMPLETE')
    print('='*55)
    print(f' ANN: {diabetes_csv}')
    print(f' CNN: {cnn_dir}')
    print('\nNext steps:')
    print('  1. Run the app:  python app.py')
    print('  2. ANN: Upload diabetes.csv via Dataset Explorer')
    print('         then go to ANN Prediction → Train')
    print('  3. CNN: Run notebooks/train_cnn.ipynb')
    print('='*55)


if __name__ == '__main__':
    main()
