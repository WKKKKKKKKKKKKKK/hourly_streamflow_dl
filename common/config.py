"""YAML config loading with dotted-key CLI overrides."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    """dict with attribute access, so cfg.train.epochs works."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _coerce(text: str) -> Any:
    """Turn a CLI string into the obvious Python value."""
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = copy.deepcopy(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_dotted(raw, key.strip(), _coerce(value))
    return Config(raw)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "phase1.yaml"),
        help="Path to the YAML config.",
    )
    # action="extend" so REPEATED --set flags accumulate. With plain nargs="*" the
    # last flag silently replaces every earlier one, so
    #   --set transfer.epochs=2 --set eval.max_batches_per_station=1
    # applied only the second override and dropped the first without a word.
    parser.add_argument(
        "--set",
        nargs="*",
        action="extend",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-key overrides, e.g. train.epochs=5 model.dropout=0.2. "
             "Repeatable; every occurrence is applied.",
    )
    return parser


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a config path relative to the repo root when it is not absolute."""
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path)
