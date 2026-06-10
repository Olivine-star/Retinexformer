import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Infer RetinexFormer on SDSD outdoor.')
    parser.add_argument('--weights', default='experiments/RetinexFormer_SDSD_outdoor/models/net_g_latest.pth')
    parser.add_argument('--gpus', default='0', help='CUDA_VISIBLE_DEVICES value.')
    parser.add_argument('--output_dir', default='results/SDSD_outdoor/enhanced')
    args, test_args = parser.parse_known_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ['PYTHONPATH'] = str(root) + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    sys.argv = [
        'Enhancement/test_sdsd.py',
        '--opt',
        'Options/RetinexFormer_SDSD_outdoor.yml',
        '--weights',
        args.weights,
        '--output_dir',
        args.output_dir,
        '--gpus',
        args.gpus,
        *test_args,
    ]
    from Enhancement.test_sdsd import main as test_main
    test_main()


if __name__ == '__main__':
    main()
