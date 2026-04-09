#!/usr/bin/env python3
"""Generate HydroDeepAI-style config_whole.txt from sweep.yaml.

Usage:
  python make_config_whole.py --sweep sweep.yaml --out config_whole.txt

Output format (tab-separated):
  ArrayTaskID  <param1> <param2> ...

Notes:
- Only supports grid sweeps with explicit `values` lists.
- Dict order in YAML is preserved (Python 3.7+), so columns follow sweep.yaml order.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required to generate config_whole.txt. "
        "Install it in your env: pip install pyyaml\n" + str(e)
    )


def _fmt(v) -> str:
    if isinstance(v, float):
        return format(v, '.10g')
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', default='sweep.yaml')
    ap.add_argument('--out', default='config_whole.txt')
    ap.add_argument('--count-out', default='config_count.txt')
    args = ap.parse_args()

    sweep_path = Path(args.sweep)
    data = yaml.safe_load(sweep_path.read_text(encoding='utf-8'))

    if str(data.get('method', '')).lower() != 'grid':
        raise SystemExit(f"Only method=grid is supported. Found: {data.get('method')!r}")

    params = data.get('parameters') or {}
    if not params:
        raise SystemExit('No parameters found in sweep.yaml')

    keys = list(params.keys())
    values_lists = []
    for k in keys:
        spec = params[k] or {}
        if 'values' not in spec:
            raise SystemExit(f"Parameter {k!r} must have explicit values for grid.")
        values_lists.append(list(spec['values']))

    combos = list(itertools.product(*values_lists))

    out_path = Path(args.out)
    lines = []
    lines.append('\t'.join(['ArrayTaskID'] + keys) + '\n')
    for i, combo in enumerate(combos, start=1):
        fields = [str(i)] + [_fmt(v) for v in combo]
        lines.append('\t'.join(fields) + '\n')

    out_path.write_text(''.join(lines), encoding='utf-8')
    Path(args.count_out).write_text(str(len(combos)) + '\n', encoding='utf-8')

    print(f"Wrote {out_path} with {len(combos)} combinations")


if __name__ == '__main__':
    main()
