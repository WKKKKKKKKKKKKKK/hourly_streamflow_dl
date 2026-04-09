#!/usr/bin/env python
import argparse
import json
import os
import sys


def load_record(path: str, idx: int):
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            if i == idx:
                return json.loads(line)
    raise SystemExit(f'No run spec for idx={idx} in {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, required=True)
    args = ap.parse_args()

    rec = load_record('grid_runs.jsonl', args.idx)
    idx = int(rec['idx'])
    params = rec.get('params', {})
    fixed = rec.get('fixed_args', [])
    combo = rec.get('combo_args', [])

    out_dir = os.path.abspath(os.path.join('runs', 'grid', f'idx_{idx:04d}'))
    os.makedirs(out_dir, exist_ok=True)

    done_path = os.path.join(out_dir, 'DONE')
    if os.path.exists(done_path):
        print('DONE marker exists; skipping index', idx)
        return

    exp = os.path.basename(os.getcwd())
    run_name = f'grid_{exp}_idx_{idx:04d}'

    # Override these no matter what sweep.yaml command contains
    skip_prefixes = (
        '--output-dir=',
        '--wandb-run-name=',
        '--wandb_run_name=',
        '--model-save-path=',
        '--best-model-path=',
        '--scaler-save-path=',
        '--checkpoint-path=',
    )
    filtered = [a for a in fixed if not a.startswith(skip_prefixes)]

    cmd = ['python', 'Train.py'] + filtered + combo + [
        f'--output-dir={out_dir}',
        f'--wandb-run-name={run_name}',
        '--resume',
    ]

    print('Index:', idx)
    print('Params:', params)
    print('Output dir:', out_dir)
    print('Command:', ' '.join(cmd))

    os.execvp(cmd[0], cmd)


if __name__ == '__main__':
    main()
