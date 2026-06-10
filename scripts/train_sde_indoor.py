import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Train RetinexFormer on SDE indoor.')
    parser.add_argument('--gpus', default='0', help='CUDA_VISIBLE_DEVICES value.')
    args, train_args = parser.parse_known_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ['PYTHONPATH'] = str(root) + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    sys.argv = [
        'basicsr/train.py',
        '--opt',
        'Options/RetinexFormer_SDE_indoor.yml',
        *train_args,
    ]
    from basicsr.train import main as train_main
    train_main()


if __name__ == '__main__':
    main()
