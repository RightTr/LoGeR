"""Run MovingDrone evaluation at both 16 and 64 frames."""
import argparse
from pathlib import Path
import subprocess
import sys

RUNNER = Path(__file__).with_name('run_movingdrone.py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--gpu', default='cuda:0')
    args = parser.parse_args()

    for frames in (16, 64):
        subprocess.run([
            sys.executable, str(RUNNER), '--scenes', str(args.scenes),
            '--output', str(args.output), '--gpu', args.gpu,
            '--frames', str(frames),
        ], check=True)


if __name__ == '__main__':
    main()
