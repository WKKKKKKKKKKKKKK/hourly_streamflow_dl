"""KGE / NSE and per-station metric aggregation.

compute_nse / compute_kge are byte-for-byte the definitions used in
MTSLSTM_100stations/code/Train.py, so numbers stay comparable with the
100-station experiment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[mask], sim[mask]
    if obs.size < 2:
        return float("nan")
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1 - np.sum((sim - obs) ** 2) / denom)


def compute_kge(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[mask], sim[mask]
    if obs.size < 2:
        return float("nan")
    mean_obs, std_obs = np.mean(obs), np.std(obs)
    if std_obs == 0 or mean_obs == 0:
        return float("nan")
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / std_obs
    beta = np.mean(sim) / mean_obs
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def kge_components(obs: np.ndarray, sim: np.ndarray) -> tuple[float, float, float, float]:
    """KGE split into the three things it penalises separately.

        KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
        r     correlation        -- is the timing right
        alpha std(sim)/std(obs)  -- is the variability right
        beta  mean(sim)/mean(obs) -- is the volume right

    Worth having because a drop in KGE says nothing about which failed. Daily-only
    supervision carries no sub-daily information, so if it makes predictions
    smoother the damage shows up in alpha while r holds -- a different conclusion
    from the model simply getting the timing wrong.
    """
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[mask], sim[mask]
    if obs.size < 2:
        return (float("nan"),) * 4
    mean_obs, std_obs = np.mean(obs), np.std(obs)
    if std_obs == 0 or mean_obs == 0:
        return (float("nan"),) * 4
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / std_obs
    beta = np.mean(sim) / mean_obs
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return float(kge), float(r), float(alpha), float(beta)


class StationAccumulator:
    """Collects (obs, sim) pairs per station across batches, in standardized space."""

    def __init__(self) -> None:
        self._sim: dict[str, list[np.ndarray]] = {}
        self._obs: dict[str, list[np.ndarray]] = {}

    def update(self, stations, sim: np.ndarray, obs: np.ndarray) -> None:
        stations = np.asarray(stations, dtype=object)
        sim = np.asarray(sim, dtype=np.float32).reshape(-1)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        # One np.unique pass beats a per-row dict lookup: batches are 512 rows and
        # usually a single station.
        uniques, inverse = np.unique(stations, return_inverse=True)
        for code, station in enumerate(uniques):
            sel = inverse == code
            key = str(station)
            self._sim.setdefault(key, []).append(sim[sel])
            self._obs.setdefault(key, []).append(obs[sel])

    def to_frame(self, y_mean: float, y_std: float, min_samples: int = 1) -> pd.DataFrame:
        """Per-station NSE/KGE in physical units (mm/h), one row per station."""
        rows = []
        for station in sorted(self._sim):
            sim = np.concatenate(self._sim[station]).astype(np.float64) * y_std + y_mean
            obs = np.concatenate(self._obs[station]).astype(np.float64) * y_std + y_mean
            row = {
                "station_id": station,
                "source": station.split("__")[0],
                "samples": int(obs.size),
                "score_status": "ok",
                "exclusion_reason": "",
                "nse": float("nan"),
                "kge": float("nan"),
                "obs_mean": float(np.nanmean(obs)) if obs.size else float("nan"),
                "obs_std": float(np.nanstd(obs)) if obs.size else float("nan"),
                "kge_r": float("nan"),
                "kge_alpha": float("nan"),
                "kge_beta": float("nan"),
                "sim_std": float("nan"),
            }
            if obs.size < max(2, min_samples):
                row["score_status"] = "excluded"
                row["exclusion_reason"] = "too_few_samples"
                rows.append(row)
                continue

            nse, kge = compute_nse(obs, sim), compute_kge(obs, sim)
            _, row["kge_r"], row["kge_alpha"], row["kge_beta"] = kge_components(obs, sim)
            if not (np.isfinite(nse) and np.isfinite(kge)):
                reasons = []
                if not np.isfinite(row["obs_std"]):
                    reasons.append("obs_std_nonfinite")
                elif row["obs_std"] == 0:
                    reasons.append("obs_std_zero")
                if row["obs_mean"] == 0:
                    reasons.append("obs_mean_zero")
                row["score_status"] = "excluded"
                row["exclusion_reason"] = "+".join(reasons) or "metric_nonfinite"
                # Keep whichever of the two is finite -- NSE often is when KGE is not.
                row["nse"] = float(nse) if np.isfinite(nse) else float("nan")
                row["kge"] = float(kge) if np.isfinite(kge) else float("nan")
            else:
                row["nse"], row["kge"] = float(nse), float(kge)
            row["sim_std"] = float(np.nanstd(sim))
            rows.append(row)
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self._sim)


def summarize(frame: pd.DataFrame, split: str = "") -> dict[str, float | int | str]:
    valid = frame.loc[frame["score_status"].eq("ok")] if len(frame) else frame
    return {
        "split": split,
        "n_stations": int(len(frame)),
        "n_valid_stations": int(len(valid)),
        "n_excluded_stations": int(len(frame) - len(valid)),
        "n_samples": int(frame["samples"].sum()) if len(frame) else 0,
        "median_kge": float(valid["kge"].median()) if len(valid) else float("nan"),
        "mean_kge": float(valid["kge"].mean()) if len(valid) else float("nan"),
        "median_nse": float(valid["nse"].median()) if len(valid) else float("nan"),
        "mean_nse": float(valid["nse"].mean()) if len(valid) else float("nan"),
        "frac_kge_gt_0": float((valid["kge"] > 0).mean()) if len(valid) else float("nan"),
        "frac_nse_gt_0": float((valid["nse"] > 0).mean()) if len(valid) else float("nan"),
    }
