#!/usr/bin/env python
import json
import itertools
from pathlib import Path


def parse_sweep_yaml(path: Path):
    lines = path.read_text(encoding='utf-8').splitlines()

    def strip_comment(s: str) -> str:
        return s.split('#', 1)[0].rstrip('\n')

    params = {}
    in_params = False
    cur_key = None
    in_values = False

    for raw in lines:
        line = strip_comment(raw)
        if not line.strip():
            continue
        if line.startswith('parameters:'):
            in_params = True
            cur_key = None
            in_values = False
            continue

        if in_params and (not line.startswith(' ')) and (
            line.startswith('command:')
            or line.startswith('program:')
            or line.startswith('metric:')
            or line.startswith('method:')
        ):
            in_params = False
            cur_key = None
            in_values = False

        if in_params:
            if line.startswith('  ') and line.strip().endswith(':') and line.strip() != 'values:':
                cur_key = line.strip()[:-1]
                params.setdefault(cur_key, [])
                in_values = False
                continue
            if cur_key and line.strip() == 'values:':
                in_values = True
                continue
            if cur_key and in_values and line.strip().startswith('- '):
                params[cur_key].append(line.strip()[2:].strip())
                continue

    fixed_args = []
    in_cmd = False
    for raw in lines:
        line = strip_comment(raw)
        if line.startswith('command:'):
            in_cmd = True
            continue
        if in_cmd:
            if not line.startswith('- '):
                in_cmd = False
                continue
            item = line[2:].strip()
            if item in ('${env}', 'python', '${program}', '${args}'):
                continue
            fixed_args.append(item)

    return params, fixed_args


def main():
    exp_dir = Path(__file__).resolve().parent
    sweep = exp_dir / 'sweep.yaml'
    params, fixed_args = parse_sweep_yaml(sweep)

    keys = list(params.keys())
    values_lists = [params[k] for k in keys]
    combos = list(itertools.product(*values_lists)) if keys else [()]

    out_path = exp_dir / 'grid_runs.jsonl'
    count_path = exp_dir / 'grid_count.txt'
    meta_path = exp_dir / 'grid_meta.json'

    with out_path.open('w', encoding='utf-8') as f:
        for idx, combo in enumerate(combos, start=1):
            combo_params = {k: v for k, v in zip(keys, combo)}
            combo_args = [f"--{k}={v}" for k, v in combo_params.items()]
            rec = {
                'idx': idx,
                'params': combo_params,
                'fixed_args': fixed_args,
                'combo_args': combo_args,
            }
            f.write(json.dumps(rec, ensure_ascii=True) + '\n')

    count_path.write_text(str(len(combos)) + '\n', encoding='utf-8')
    meta_path.write_text(
        json.dumps({'keys': keys, 'fixed_args': fixed_args}, ensure_ascii=True, indent=2) + '\n',
        encoding='utf-8',
    )

    print(f"Wrote {out_path} with {len(combos)} runs")


if __name__ == '__main__':
    main()
