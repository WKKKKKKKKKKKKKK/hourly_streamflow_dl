"""Shared plotting helpers: font scaling, coastlines and country borders, and the diverging
colour classes of the gain maps.

Collected here from the original repository's global_map.py, fig_map_adaptation.py and
fig_full_series.py, which drew figures that are not in the manuscript. Only the pieces the
manuscript figures use are kept, unchanged. Importing this module has no side effect on
matplotlib's rcParams; the figure scripts set their own fonts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Explicit font sizes are multiplied by FONT_SCALE. The section figures (sec41/42/43) set it
# to 1.0 (drawn at the 177 mm HESS text width); the Methods maps keep the default.
FONT_SCALE = 1.3


def fs(size: float) -> float:
    """An explicit font size, scaled for legibility at print size."""
    return round(size * FONT_SCALE, 2)


# Default map window (label_availability narrows it per figure).
XLIM = (-180.0, 180.0)
YLIM = (-60.0, 80.0)

_BORDER_CACHE: dict[str, object] = {}


def _borders():
    """Coastlines and country borders as line geometries, read once from cartopy's cached
    Natural Earth 1:110m shapefiles (no reprojection: the maps use plain lon/lat axes)."""
    if not _BORDER_CACHE:
        import geopandas as gpd
        found = {}
        for label, stem in (("coast", "ne_110m_coastline"),
                            ("country", "ne_110m_admin_0_boundary_lines_land")):
            hits = sorted(Path.home().glob(f".local/share/cartopy/shapefiles/**/{stem}.shp"))
            if not hits:
                raise SystemExit(
                    f"{stem}.shp is not in the cartopy cache (~/.local/share/cartopy). "
                    "Download the Natural Earth 110m coastline and admin-0 boundary lines first."
                )
            found[label] = gpd.read_file(hits[0]).geometry
        _BORDER_CACHE.update(found)
    return _BORDER_CACHE


def draw_borders(ax):
    """Coastlines (darker) and country borders at zorder 1, below every data layer."""
    b = _borders()
    b["coast"].plot(ax=ax, color="#5f5f5f", linewidth=0.45, zorder=1)
    b["country"].plot(ax=ax, color="#9a9a9a", linewidth=0.30, zorder=1)


# Diverging classes for a paired difference (M1 - M0, or M1 - daily LSTM), symmetric about 0
# and clipped at +-0.5 so that a few collapsed gauges do not set the scale.
GAIN_BOUNDS = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
RDBU = plt.get_cmap("RdBu_r")
GAIN_COLOURS = ([RDBU(x) for x in np.linspace(0.02, 0.42, 5)]
                + [RDBU(x) for x in np.linspace(0.58, 0.98, 5)])
