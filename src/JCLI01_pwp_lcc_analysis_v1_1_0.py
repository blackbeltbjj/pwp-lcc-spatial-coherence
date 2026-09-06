#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JCLI01_pwp_lcc_analysis_v1_1_0.py
==========================================================================
PURPOSE
    End-to-end, reproducible numerical-analysis workflow for the
    Journal of Climate manuscript on the largest connected component (LCC) of
    the Pacific Warm Pool (PWP).

SCIENTIFIC OBJECT
    For every valid daily NOAA OISST v2.1 field and SST threshold T in
    {28.0, 28.5, 29.0 degC}, the canonical PWP is all valid Pacific-mask cells
    with SST >= T. Eight-neighbor connected components are ranked by exact
    spherical area. The largest is the PWP-LCC.

INPUTS (relative to PROJECT_ROOT)
    data/raw/sst.day.mean.1981.nc ... sst.day.mean.2026.nc
    data/processed/pacific_mask_oisst.npy
    data/processed/grid_lat.npy
    data/processed/grid_lon.npy

OPTIONAL VALIDATED FAST-PATH INPUT
    outputs/tables/threshold_comparison/pwp_long_term_connectivity/
        pwp_daily_connectivity_diagnostics.csv

OUTPUT ROOT
    outputs/JCLI/
        figures/jcli_pwp_lcc/
        tables/jcli_pwp_lcc/
        reports/jcli_pwp_lcc/
        metadata/jcli_pwp_lcc/
        cache/jcli_pwp_lcc/

SEPARATION OF RESPONSIBILITIES
    This program performs the expensive scientific calculations once and
    writes immutable numerical products.  Publication figures are generated
    separately by JCLI02_figures_pwp_lcc_publication_v1_1_0.py, which never
    opens the raw OISST NetCDF collection.

NO SILENT REPROCESSING
    - The validated Pacific mask is required and never reconstructed.
    - No SST interpolation, spatial smoothing, morphology, or component-size
      filtering is applied.
    - Partial years 1981 and 2026 remain in daily diagnostics but are excluded
      from complete-calendar-year trend inference.
    - Canonical-area STL, spectral, wavelet, or occurrence products are never
      substituted for LCC-specific calculations.

PERFORMANCE
    Xarray opens each annual OISST file with Dask chunks. Data are loaded in
    bounded time blocks; connected-component labeling is then performed on
    in-memory two-dimensional daily masks. Reusable derived products are
    cached so figure-only reruns do not reread all NetCDF fields.

DEPENDENCIES
    Python >=3.10; numpy; pandas; xarray; dask[array]; netCDF4 or h5netcdf;
    scipy; statsmodels; matplotlib; cartopy; pycwt.

EXECUTION
    From the project root:
        python src/JCLI01_pwp_lcc_analysis_v1_1_0.py

    Useful options:
        --project-root PATH
        --rebuild-daily
        --rebuild-spatial
        --no-raw-maps
        --start-date YYYY-MM-DD --end-date YYYY-MM-DD
        --refresh-provenance-only
        --derive-detachment-only

AUTHOR
    Fabio Vieira Machado
VERSION
    1.1.0 (2026-09-05; disconnected-occurrence diagnostic)
==========================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage, signal, stats
from scipy.stats import theilslopes
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, ScalarFormatter

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError as exc:  # pragma: no cover - checked on user's workstation
    raise ImportError("Cartopy is required. Install with: conda install -c conda-forge cartopy") from exc

try:
    import pycwt as wavelet
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyCWT is required. Install with: pip install pycwt") from exc


PROGRAM_NAME = "JCLI01 PWP-LCC SCIENTIFIC ANALYSIS"
PROGRAM_VERSION = "1.1.0"
PROGRAM_DATE = "2026-09-05"
THRESHOLDS_C = (28.0, 28.5, 29.0)
PRIMARY_THRESHOLD_C = 28.0
EXPECTED_START = pd.Timestamp("1981-09-01")
EXPECTED_END = pd.Timestamp("2026-07-29")
EXPECTED_N_DAYS = 16_403
COMPLETE_YEAR_START = 1982
COMPLETE_YEAR_END = 2025
EARTH_RADIUS_KM = 6371.0088
CONNECTIVITY_LABEL = "8-neighbor"
CONNECTIVITY_STRUCTURE = np.ones((3, 3), dtype=np.uint8)
ROLLING_DAYS = 365
STL_PERIOD_DAYS = 365
WELCH_SEGMENT_DAYS = 2920
WELCH_OVERLAP_DAYS = 1460
WAVELET_DT_DAYS = 1.0
WAVELET_DJ = 1.0 / 12.0
WAVELET_OMEGA0 = 6.0
SIGNIFICANCE_LEVEL = 0.95
YEAR_TICK_START = 1982
YEAR_TICK_STEP = 4
MAP_EXTENT = (100.0, 290.0, -35.0, 35.0)
FIGURE_DPI = 400
RASTER_DPI = 600
BLOCK_DAYS = 31

# Bands are prespecified and intentionally allowed to overlap because each
# answers a different physical question. Fractions must not be summed.
SPECTRAL_BANDS_DAYS = {
    "annual_strict": (336.0, 378.0),
    "quasi_biennial": (550.0, 1096.0),
    "interannual_broad": (805.0, 2922.0),
}

THRESHOLD_STYLES = {
    28.0: dict(color="0.05", linestyle="-", marker="o"),
    28.5: dict(color="0.35", linestyle="--", marker="s"),
    29.0: dict(color="0.62", linestyle=":", marker="^"),
}


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    processed: Path
    mask: Path
    grid_lat: Path
    grid_lon: Path
    output_root: Path
    figures: Path
    tables: Path
    reports: Path
    metadata: Path
    cache: Path
    upstream_daily: Path
    daily: Path
    annual: Path
    trend: Path
    climatology: Path
    stl_daily: Path
    stl_summary: Path
    occurrence_npz: Path
    occurrence_summary: Path
    detachment_summary: Path
    detachment_report: Path
    spectral_summary: Path
    spectral_bands: Path
    wavelet_bands: Path
    spectral_npz: Path
    wavelet_npz: Path
    map_snapshots_npz: Path
    report: Path
    metadata_json: Path
    manifest: Path


def resolve_paths(root: Path) -> Paths:
    root = root.expanduser().resolve()
    out = root / "outputs" / "JCLI"
    fig = out / "figures" / "jcli_pwp_lcc"
    tab = out / "tables" / "jcli_pwp_lcc"
    rep = out / "reports" / "jcli_pwp_lcc"
    meta = out / "metadata" / "jcli_pwp_lcc"
    cache = out / "cache" / "jcli_pwp_lcc"
    proc = root / "data" / "processed"
    return Paths(
        root=root, raw=root / "data" / "raw", processed=proc,
        mask=proc / "pacific_mask_oisst.npy",
        grid_lat=proc / "grid_lat.npy", grid_lon=proc / "grid_lon.npy",
        output_root=out, figures=fig, tables=tab, reports=rep,
        metadata=meta, cache=cache,
        upstream_daily=root / "outputs" / "tables" / "threshold_comparison" /
            "pwp_long_term_connectivity" / "pwp_daily_connectivity_diagnostics.csv",
        daily=tab / "jcli_pwp_lcc_daily_diagnostics.csv",
        annual=tab / "JCLI_pwp_lcc_annual_summary.csv",
        trend=tab / "JCLI_pwp_lcc_trend_statistics.csv",
        climatology=tab / "JCLI_pwp_lcc_monthly_climatology.csv",
        stl_daily=tab / "JCLI_pwp_lcc_stl_daily.csv",
        stl_summary=tab / "JCLI_pwp_lcc_stl_seasonal_amplitude_summary.csv",
        occurrence_npz=proc / "jcli_pwp_lcc" / "JCLI_pwp_lcc_occurrence_persistence.npz",
        occurrence_summary=tab / "JCLI_pwp_lcc_occurrence_persistence_summary.csv",
        detachment_summary=tab / "JCLI_pwp_lcc_disconnected_occurrence_summary.csv",
        detachment_report=rep / "JCLI_pwp_lcc_disconnected_occurrence_report.txt",
        spectral_summary=tab / "JCLI_pwp_lcc_welch_summary.csv",
        spectral_bands=tab / "JCLI_pwp_lcc_welch_band_variance.csv",
        wavelet_bands=tab / "JCLI_pwp_lcc_wavelet_band_variance.csv",
        spectral_npz=cache / "JCLI_pwp_lcc_welch_products.npz",
        wavelet_npz=cache / "JCLI_pwp_lcc_wavelet_products.npz",
        map_snapshots_npz=cache / "JCLI_pwp_lcc_map_snapshots.npz",
        report=rep / "JCLI01_SCIENTIFIC_REPORT.txt",
        metadata_json=meta / "JCLI01_METADATA.json",
        manifest=meta / "JCLI01_OUTPUT_SHA256.csv",
    )


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "src" else Path.cwd()
    p = argparse.ArgumentParser(description=PROGRAM_NAME)
    p.add_argument("--project-root", type=Path, default=default_root)
    p.add_argument("--start-date", default=str(EXPECTED_START.date()))
    p.add_argument("--end-date", default=str(EXPECTED_END.date()))
    p.add_argument("--rebuild-daily", action="store_true")
    p.add_argument("--rebuild-spatial", action="store_true")
    p.add_argument("--no-raw-maps", action="store_true", help="Skip Figs. 1 and 6 if raw fields are unavailable.")
    p.add_argument(
        "--refresh-provenance-only", action="store_true",
        help=("Validate existing JCLI01 products and regenerate only JCLI01_METADATA.json "
              "and JCLI01_OUTPUT_SHA256.csv; do not repeat scientific calculations."),
    )
    p.add_argument(
        "--derive-detachment-only", action="store_true",
        help=("Derive the disconnected-occurrence table and report from the existing "
              "canonical/LCC occurrence cache; do not reopen OISST NetCDF files."),
    )
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.5, "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "lines.linewidth": 1.2,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "savefig.dpi": FIGURE_DPI, "figure.dpi": 120,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def ensure_directories(paths: Paths) -> None:
    for p in (paths.figures, paths.tables, paths.reports, paths.metadata, paths.cache, paths.occurrence_npz.parent):
        p.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, **kwargs)
    tmp.replace(path)


def save_json(obj: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def save_figure(fig: plt.Figure, paths: Paths, stem: str) -> list[Path]:
    png = paths.figures / f"{stem}.png"
    pdf = paths.figures / f"{stem}.pdf"
    tif = paths.figures / f"{stem}.tiff"
    fig.savefig(png, dpi=FIGURE_DPI)
    fig.savefig(pdf)
    fig.savefig(tif, dpi=RASTER_DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return [png, pdf, tif]


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.012, 0.975, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=11, fontweight="bold", zorder=100,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.2))


def grid(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="0.85", linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)


def format_long_time_axis(ax: plt.Axes) -> None:
    ax.set_xlim(EXPECTED_START, EXPECTED_END)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=YEAR_TICK_STEP, month=1, day=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.tick_params(axis="x", rotation=0)


def identify_name(names: Iterable[str], candidates: Sequence[str], kind: str) -> str:
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"Unable to identify {kind}; available names: {sorted(names)}")


def normalize_longitude(lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon360 = np.mod(np.asarray(lon, dtype=float), 360.0)
    order = np.argsort(lon360)
    out = lon360[order]
    if np.any(np.diff(out) <= 0):
        raise ValueError("Longitude must be unique after conversion to [0, 360).")
    return out, order


def coordinate_edges(coord: np.ndarray, periodic: bool = False) -> np.ndarray:
    x = np.asarray(coord, dtype=float)
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("Coordinates must be strictly increasing one-dimensional arrays.")
    mid = 0.5 * (x[1:] + x[:-1])
    edges = np.empty(x.size + 1)
    edges[1:-1] = mid
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    if not periodic:
        edges = np.clip(edges, -90.0, 90.0)
    return edges


def spherical_cell_areas_km2(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_e = np.deg2rad(coordinate_edges(lat, periodic=False))
    lon_e = np.deg2rad(coordinate_edges(lon, periodic=True))
    dlon = np.diff(lon_e)
    lat_factor = np.sin(lat_e[1:]) - np.sin(lat_e[:-1])
    area = EARTH_RADIUS_KM ** 2 * lat_factor[:, None] * dlon[None, :]
    if np.any(~np.isfinite(area)) or np.any(area <= 0):
        raise ValueError("Invalid spherical cell areas; inspect coordinate ordering.")
    return area


def load_reference_grid(paths: Paths) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for p in (paths.mask, paths.grid_lat, paths.grid_lon):
        if not p.is_file():
            raise FileNotFoundError(
                f"Required validated spatial input not found: {p}\n"
                "Do not reconstruct the Pacific mask silently. Copy the versioned mask/grid files into data/processed/."
            )
    mask = np.asarray(np.load(paths.mask), dtype=bool)
    lat = np.asarray(np.load(paths.grid_lat), dtype=float)
    lon_raw = np.asarray(np.load(paths.grid_lon), dtype=float)
    lon, order = normalize_longitude(lon_raw)
    mask = mask[:, order]
    if lat[0] > lat[-1]:
        lat = lat[::-1]; mask = mask[::-1, :]
    if mask.shape != (lat.size, lon.size):
        raise ValueError(f"Mask shape {mask.shape} differs from grid {(lat.size, lon.size)}.")
    area = spherical_cell_areas_km2(lat, lon)
    logging.info("Reference grid: %d lat × %d lon; Pacific cells=%d", lat.size, lon.size, mask.sum())
    return lat, lon, mask, area


def discover_oisst_files(paths: Paths, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    files = []
    for year in range(start.year, end.year + 1):
        p = paths.raw / f"sst.day.mean.{year}.nc"
        if not p.is_file():
            raise FileNotFoundError(f"Missing annual OISST file: {p}")
        files.append(p)
    return files


def open_annual_oisst(path: Path, target_lat: np.ndarray, target_lon: np.ndarray) -> xr.DataArray:
    # Open lazily before identifying coordinate names; some valid products use
    # ``date`` rather than ``time``, so hard-coding a chunk key here is unsafe.
    ds = xr.open_dataset(path, chunks={}, decode_times=True, mask_and_scale=True)
    sst_name = identify_name(ds.data_vars, ("sst", "SST", "sea_surface_temperature", "analysed_sst"), "SST variable")
    lat_name = identify_name(ds.coords, ("lat", "latitude", "y"), "latitude coordinate")
    lon_name = identify_name(ds.coords, ("lon", "longitude", "x"), "longitude coordinate")
    time_name = identify_name(ds.coords, ("time", "date"), "time coordinate")
    da = ds[sst_name]
    # NOAA OISST occasionally retains a singleton zlev coordinate.
    for dim in list(da.dims):
        if dim not in (time_name, lat_name, lon_name) and da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
    da = da.transpose(time_name, lat_name, lon_name)
    lon360 = np.mod(np.asarray(da[lon_name].values, dtype=float), 360.0)
    order = np.argsort(lon360)
    da = da.isel({lon_name: order}).assign_coords({lon_name: lon360[order]})
    if float(da[lat_name][0]) > float(da[lat_name][-1]):
        da = da.isel({lat_name: slice(None, None, -1)})
    if not np.allclose(da[lat_name].values, target_lat, atol=1e-8, rtol=0):
        raise ValueError(f"Latitude mismatch in {path}; no interpolation is permitted.")
    if not np.allclose(da[lon_name].values, target_lon, atol=1e-8, rtol=0):
        raise ValueError(f"Longitude mismatch in {path}; no interpolation is permitted.")
    da = da.rename({time_name: "time", lat_name: "lat", lon_name: "lon"})
    return da.chunk({"time": BLOCK_DAYS, "lat": -1, "lon": -1})


class UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n + 1, dtype=np.int32)
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = int(self.p[x])
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra


def label_periodic_8(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, n = ndimage.label(mask, structure=CONNECTIVITY_STRUCTURE)
    if n == 0: return labels.astype(np.int32, copy=False), 0
    uf = UnionFind(n)
    ny = mask.shape[0]
    # Join components touching the 0/360 seam using 8-neighbor adjacency.
    for i in range(ny):
        if not mask[i, 0]: continue
        for j in (i - 1, i, i + 1):
            if 0 <= j < ny and mask[j, -1]:
                uf.union(int(labels[i, 0]), int(labels[j, -1]))
    roots = np.arange(n + 1, dtype=np.int32)
    for k in range(1, n + 1): roots[k] = uf.find(k)
    merged = roots[labels]
    unique = np.unique(merged[merged > 0])
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[unique] = np.arange(1, unique.size + 1)
    return remap[merged], int(unique.size)


_SPHERICAL_BASIS_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def spherical_centroid(mask: np.ndarray, area: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    """Area-weighted spherical centroid, with immutable grid basis cached once."""
    w = np.where(mask, area, 0.0)
    total = float(w.sum())
    if total <= 0: return math.nan, math.nan
    key = (lat.size, lon.size, float(lat[0]), float(lat[-1]), float(lon[0]), float(lon[-1]))
    if key not in _SPHERICAL_BASIS_CACHE:
        lat2, lon2 = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
        _SPHERICAL_BASIS_CACHE[key] = (
            np.cos(lat2) * np.cos(lon2),
            np.cos(lat2) * np.sin(lon2),
            np.sin(lat2),
        )
    bx, by, bz = _SPHERICAL_BASIS_CACHE[key]
    x = float(np.sum(w * bx) / total)
    y = float(np.sum(w * by) / total)
    z = float(np.sum(w * bz) / total)
    return float(np.mod(np.rad2deg(np.arctan2(y, x)), 360.0)), float(np.rad2deg(np.arctan2(z, np.hypot(x, y))))


def great_circle_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    a1, a2 = np.deg2rad([lat1, lat2]); dl = np.deg2rad(((lon2 - lon1 + 180) % 360) - 180)
    x = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(dl / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(x, 0, 1))))


def daily_components(sst: np.ndarray, threshold: float, pacific: np.ndarray, area: np.ndarray,
                     lat: np.ndarray, lon: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    valid = pacific & np.isfinite(sst)
    canonical = valid & (sst >= threshold)
    labels, n = label_periodic_8(canonical)
    if n == 0: raise ValueError(f"No threshold-exceeding component for {threshold:g}°C.")
    component_area = np.bincount(labels.ravel(), weights=area.ravel(), minlength=n + 1)
    component_area[0] = 0.0
    largest_id = int(np.argmax(component_area))
    lcc = labels == largest_id
    total = float(area[canonical].sum()); largest = float(component_area[largest_id])
    full_lon, full_lat = spherical_centroid(canonical, area, lat, lon)
    lcc_lon, lcc_lat = spherical_centroid(lcc, area, lat, lon)
    # Program 31 convention: canonical centroid minus LCC centroid.
    dlon = float(((full_lon - lcc_lon + 180) % 360) - 180)
    out = {
        "connectivity": CONNECTIVITY_LABEL, "component_count": n,
        "total_area_km2": total, "largest_component_area_km2": largest,
        "largest_component_area_fraction": largest / total,
        "secondary_area_km2": total - largest, "fragmentation_index": 1 - largest / total,
        "full_centroid_lon_360": full_lon, "full_centroid_lat": full_lat,
        "largest_centroid_lon_360": lcc_lon, "largest_centroid_lat": lcc_lat,
        "delta_lon_deg": dlon, "delta_lat_deg": full_lat - lcc_lat,
        "centroid_distance_km": great_circle_km(full_lon, full_lat, lcc_lon, lcc_lat),
    }
    return out, lcc


def validate_daily(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    required = {"date", "threshold_c", "component_count", "total_area_km2",
                "largest_component_area_km2", "largest_component_area_fraction",
                "full_centroid_lon_360", "full_centroid_lat", "largest_centroid_lon_360",
                "largest_centroid_lat", "centroid_distance_km"}
    missing = required - set(frame.columns)
    if missing: raise ValueError(f"Daily diagnostics missing columns: {sorted(missing)}")
    f = frame.copy(); f["date"] = pd.to_datetime(f["date"])
    for c in required - {"date"}: f[c] = pd.to_numeric(f[c], errors="raise")
    f = f[(f.date >= start) & (f.date <= end)].sort_values(["date", "threshold_c"]).reset_index(drop=True)
    expected_dates = pd.date_range(start, end, freq="D")
    for t in THRESHOLDS_C:
        g = f[np.isclose(f.threshold_c, t)]
        if len(g) != len(expected_dates) or not pd.DatetimeIndex(g.date).equals(expected_dates):
            raise ValueError(f"Threshold {t:g}°C is not a complete consecutive daily record.")
    if "secondary_area_km2" not in f: f["secondary_area_km2"] = f.total_area_km2 - f.largest_component_area_km2
    if "fragmentation_index" not in f: f["fragmentation_index"] = 1 - f.largest_component_area_fraction
    # Normalize displacement signs even when reusing an older Program 31 table.
    f["delta_lon_deg"] = (
        (f.full_centroid_lon_360 - f.largest_centroid_lon_360 + 180.0) % 360.0
    ) - 180.0
    f["delta_lat_deg"] = f.full_centroid_lat - f.largest_centroid_lat
    f["year"] = f.date.dt.year
    return f


def build_daily_and_spatial(paths: Paths, files: list[Path], lat: np.ndarray, lon: np.ndarray,
                            pacific: np.ndarray, area: np.ndarray, start: pd.Timestamp,
                            end: pd.Timestamp, need_spatial: bool) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    records: list[dict[str, object]] = []
    shape = pacific.shape
    canonical_counts = {t: np.zeros(shape, dtype=np.uint32) for t in THRESHOLDS_C}
    counts = {t: np.zeros(shape, dtype=np.uint32) for t in THRESHOLDS_C}
    current = {t: np.zeros(shape, dtype=np.uint16) for t in THRESHOLDS_C}
    longest = {t: np.zeros(shape, dtype=np.uint16) for t in THRESHOLDS_C}
    denominators = np.zeros(shape, dtype=np.uint32)
    valid_days = 0
    snapshots: dict[str, np.ndarray] = {}
    for nc in files:
        logging.info("Processing %s", nc.name)
        da = open_annual_oisst(nc, lat, lon).sel(time=slice(start, end))
        for k0 in range(0, da.sizes["time"], BLOCK_DAYS):
            block = da.isel(time=slice(k0, min(k0 + BLOCK_DAYS, da.sizes["time"]))).load()
            for k in range(block.sizes["time"]):
                date = pd.Timestamp(block.time.values[k]).normalize()
                sst = np.asarray(block.isel(time=k).values, dtype=np.float64)
                valid = pacific & np.isfinite(sst)
                denominators += valid.astype(np.uint32); valid_days += 1
                for t in THRESHOLDS_C:
                    diag, lcc = daily_components(sst, t, pacific, area, lat, lon)
                    records.append({"date": date.date().isoformat(), "year": date.year,
                                    "threshold_c": t, **diag})
                    if need_spatial:
                        canonical = valid & (sst >= t)
                        canonical_counts[t] += canonical.astype(np.uint32)
                        counts[t] += lcc.astype(np.uint32)
                        current[t] = np.where(lcc, np.minimum(current[t].astype(np.uint32) + 1, 65535), 0).astype(np.uint16)
                        longest[t] = np.maximum(longest[t], current[t])
                if valid_days % 365 == 0: logging.info("Processed %d daily fields", valid_days)
        da.close()
    daily = validate_daily(pd.DataFrame.from_records(records), start, end)
    arrays: dict[str, np.ndarray] = {"lat": lat, "lon": lon, "pacific_mask": pacific,
        "valid_day_count": denominators, "n_dates": np.array(valid_days, dtype=np.int32)}
    for t in THRESHOLDS_C:
        code = str(t).replace(".", "p")
        arrays[f"canonical_occurrence_count_{code}"] = canonical_counts[t]
        arrays[f"lcc_occurrence_count_{code}"] = counts[t]
        arrays[f"lcc_longest_run_days_{code}"] = longest[t]
    return daily, arrays


def load_or_build_daily(paths: Paths, args: argparse.Namespace, files: list[Path], lat: np.ndarray,
                        lon: np.ndarray, pacific: np.ndarray, area: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
    start, end = pd.Timestamp(args.start_date), pd.Timestamp(args.end_date)
    spatial_exists = paths.occurrence_npz.is_file()
    if not args.rebuild_daily:
        candidate = paths.daily if paths.daily.is_file() else paths.upstream_daily
        if candidate.is_file():
            logging.info("Using validated daily diagnostics: %s", candidate)
            daily = validate_daily(pd.read_csv(candidate, low_memory=False), start, end)
            if spatial_exists and not args.rebuild_spatial:
                cached = dict(np.load(paths.occurrence_npz, allow_pickle=False))
                required = {
                    f"canonical_occurrence_count_{str(t).replace('.', 'p')}"
                    for t in THRESHOLDS_C
                }
                if required.issubset(cached):
                    return daily, cached
                logging.warning("Spatial cache lacks canonical occurrence; rebuilding it.")
            if args.no_raw_maps: return daily, None
            # Daily metrics are reused, but raw fields must still be scanned for LCC occurrence.
            _, arrays = build_daily_and_spatial(paths, files, lat, lon, pacific, area, start, end, True)
            return daily, arrays
    if args.no_raw_maps:
        raise RuntimeError("Daily diagnostics unavailable and --no-raw-maps prevents complete rebuilding.")
    return build_daily_and_spatial(paths, files, lat, lon, pacific, area, start, end, True)


def complete_year_annual(daily: pd.DataFrame) -> pd.DataFrame:
    f = daily[(daily.year >= COMPLETE_YEAR_START) & (daily.year <= COMPLETE_YEAR_END)].copy()
    metrics = ["total_area_km2", "largest_component_area_km2", "secondary_area_km2",
               "largest_component_area_fraction", "fragmentation_index", "component_count",
               "centroid_distance_km"]
    annual = f.groupby(["threshold_c", "year"], as_index=False)[metrics].mean()
    counts = f.groupby(["threshold_c", "year"]).size().rename("n_days").reset_index()
    annual = annual.merge(counts, on=["threshold_c", "year"])
    expected = annual.year.map(lambda y: 366 if pd.Timestamp(f"{y}-12-31").is_leap_year else 365)
    if not np.all(annual.n_days.to_numpy() == expected.to_numpy()):
        raise ValueError("Annual inference contains incomplete calendar years.")
    annual["date"] = pd.to_datetime(annual.year.astype(str) + "-07-01")
    return annual


def autocorrelation_fft(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, float); z = z - np.mean(z); n = len(z)
    f = np.fft.rfft(z, n=2*n); ac = np.fft.irfft(f*np.conjugate(f))[:n]
    ac /= np.arange(n, 0, -1); return ac / ac[0]


def effective_sample_size(x: np.ndarray) -> tuple[float, float]:
    """Bartlett/integral-timescale N_eff and df=N_eff-2 using positive ACF pairs."""
    rho = autocorrelation_fft(np.asarray(x, float)); n = len(rho)
    # Geyer's initial-positive-sequence stabilizes the truncation of noisy ACF tails.
    s = 0.0
    for k in range(1, n - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair <= 0: break
        s += (1 - k/n) * rho[k] + (1 - (k+1)/n) * rho[k+1]
    neff = float(np.clip(n / (1 + 2*s), 3.0, n))
    return neff, max(neff - 2.0, 1.0)


def trend_statistics(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["total_area_km2", "largest_component_area_km2", "secondary_area_km2",
               "largest_component_area_fraction", "fragmentation_index", "component_count"]
    for t, g in annual.groupby("threshold_c"):
        x = g.year.to_numpy(float)
        for m in metrics:
            y = g[m].to_numpy(float)
            slope, intercept, low, high = theilslopes(y, x, alpha=0.95)
            resid = y - (intercept + slope*x)
            neff, df = effective_sample_size(resid)
            # Linear slope with Newey-West/HAC covariance.  Statsmodels returns
            # the covariance estimate; inference is evaluated explicitly with
            # the autocorrelation-adjusted residual degrees of freedom.
            xc = x - np.mean(x)
            design = sm.add_constant(xc)
            hac_lags = max(1, int(np.floor(4.0 * (len(y) / 100.0) ** (2.0 / 9.0))))
            hac = sm.OLS(y, design, missing="raise").fit(
                cov_type="HAC", cov_kwds={"maxlags": hac_lags, "use_correction": True}
            )
            hac_slope = float(hac.params[1])
            hac_se = float(hac.bse[1])
            hac_t = hac_slope / hac_se
            hac_p = float(2.0 * stats.t.sf(abs(hac_t), df=df))
            hac_crit = float(stats.t.ppf(0.975, df=df))
            rows.append({"threshold_c": t, "metric": m, "n_years": len(y),
                "mean": float(np.mean(y)), "theil_sen_slope_per_decade": float(10*slope),
                "ci95_low_per_decade": float(10*low), "ci95_high_per_decade": float(10*high),
                "intercept": float(intercept), "effective_sample_size_residual": neff,
                "effective_degrees_of_freedom": df,
                "hac_maxlags": hac_lags,
                "hac_slope_per_decade": float(10.0 * hac_slope),
                "hac_se_per_decade": float(10.0 * hac_se),
                "hac_ci95_low_per_decade": float(10.0 * (hac_slope - hac_crit * hac_se)),
                "hac_ci95_high_per_decade": float(10.0 * (hac_slope + hac_crit * hac_se)),
                "hac_t_effective_df": float(hac_t),
                "hac_p_effective_df": hac_p,
                "relative_slope_pct_per_decade": float(100*10*slope/np.mean(y))})
    return pd.DataFrame(rows)


def monthly_climatology(daily: pd.DataFrame) -> pd.DataFrame:
    f = daily.copy(); f["month"] = f.date.dt.month
    cols = ["largest_component_area_km2", "total_area_km2", "secondary_area_km2",
            "largest_component_area_fraction", "component_count"]
    out = f.groupby(["threshold_c", "month"])[cols].agg(["mean", "std"])
    out.columns = [f"{variable}_{statistic}" for variable, statistic in out.columns]
    return out.reset_index()


def run_stl(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, summary = [], []
    for t, g in daily.groupby("threshold_c"):
        g = g.sort_values("date"); y = g.largest_component_area_km2.to_numpy(float)
        fit = STL(y, period=STL_PERIOD_DAYS, seasonal=365, trend=1095, robust=True).fit()
        temp = pd.DataFrame({"date": g.date.to_numpy(), "threshold_c": t,
            "observed_km2": y, "trend_km2": fit.trend, "seasonal_km2": fit.seasonal,
            "remainder_km2": fit.resid})
        temp["year"] = temp.date.dt.year
        amp = temp.groupby("year").seasonal_km2.agg(lambda v: float(v.max()-v.min())).rename("seasonal_amplitude_km2")
        mean = temp.groupby("year").observed_km2.mean().rename("annual_mean_lcc_area_km2")
        a = pd.concat([amp, mean], axis=1).reset_index()
        a["threshold_c"] = t; a["relative_amplitude_pct"] = 100*a.seasonal_amplitude_km2/a.annual_mean_lcc_area_km2
        valid = a[(a.year >= COMPLETE_YEAR_START) & (a.year <= COMPLETE_YEAR_END)]
        for metric in ("seasonal_amplitude_km2", "relative_amplitude_pct"):
            sl, itc, lo, hi = theilslopes(valid[metric], valid.year, alpha=.95)
            summary.append({"threshold_c":t, "metric":metric, "mean":valid[metric].mean(),
                "theil_sen_slope_per_decade":10*sl, "ci95_low_per_decade":10*lo,
                "ci95_high_per_decade":10*hi, "remainder_variance":float(np.var(fit.resid, ddof=1))})
        out.append(temp); summary.append({"threshold_c":t, "metric":"component_variance",
            "mean":math.nan, "theil_sen_slope_per_decade":math.nan, "ci95_low_per_decade":math.nan,
            "ci95_high_per_decade":math.nan, "remainder_variance":float(np.var(fit.resid,ddof=1)),
            "trend_variance":float(np.var(fit.trend,ddof=1)), "seasonal_variance":float(np.var(fit.seasonal,ddof=1))})
    return pd.concat(out, ignore_index=True), pd.DataFrame(summary)


def detrend(y: np.ndarray) -> tuple[np.ndarray, float]:
    z = signal.detrend(np.asarray(y, float), type="linear")
    r1 = float(np.corrcoef(z[:-1], z[1:])[0,1])
    return z, float(np.clip(r1, -0.99, 0.99))


def welch_effective_dof(window: np.ndarray, step: int, n_samples: int) -> float:
    """Equivalent chi-square DOF correcting overlapping-window dependence."""
    nper = len(window); k = 1 + (n_samples - nper)//step
    if k <= 1: return 2.0
    den = float(np.sum(window**2)); correction = 0.0
    for m in range(1, k):
        lag = m*step
        if lag >= nper: break
        rho_w = float(np.dot(window[:-lag], window[lag:]) / den)
        correction += (1 - m/k)*rho_w**2
    return float(2*k/(1 + 2*correction))


def ar1_background(freq: np.ndarray, variance: float, lag1: float) -> np.ndarray:
    raw = (1-lag1**2)/(1+lag1**2-2*lag1*np.cos(2*np.pi*freq))
    integral = np.trapezoid(raw, freq)
    return raw * variance / integral


def integrate_psd(freq: np.ndarray, psd: np.ndarray, min_days: float, max_days: float) -> float:
    sel = (freq >= 1/max_days) & (freq <= 1/min_days)
    if sel.sum() == 0: return math.nan
    df = np.gradient(freq)
    return float(np.sum(psd[sel]*df[sel]))


def run_welch(daily: pd.DataFrame) -> tuple[dict[float, dict[str, np.ndarray | float]], pd.DataFrame, pd.DataFrame]:
    products, summaries, bands = {}, [], []
    for t, g in daily.groupby("threshold_c"):
        y, lag1 = detrend(g.sort_values("date").largest_component_area_km2.to_numpy(float))
        nper = min(WELCH_SEGMENT_DAYS, len(y)); nover = min(WELCH_OVERLAP_DAYS, nper//2)
        win = signal.windows.hann(nper, sym=False)
        freq, psd = signal.welch(y, fs=1.0, window=win, nperseg=nper, noverlap=nover,
                                 detrend=False, scaling="density", return_onesided=True)
        keep = freq > 0; freq, psd = freq[keep], psd[keep]
        variance = float(np.var(y, ddof=1)); dof = welch_effective_dof(win, nper-nover, len(y))
        bg = ar1_background(freq, variance, lag1)
        sig = bg * stats.chi2.ppf(SIGNIFICANCE_LEVEL, dof)/dof
        period = 1/freq; dominant = float(period[np.argmax(psd)])
        summaries.append({"threshold_c":t, "n_days":len(y), "lag1":lag1,
            "welch_segment_days":nper, "overlap_days":nover, "effective_welch_dof":dof,
            "dominant_period_days":dominant, "detrended_variance_km4":variance})
        for name,(pmin,pmax) in SPECTRAL_BANDS_DAYS.items():
            power = integrate_psd(freq,psd,pmin,pmax)
            bands.append({"threshold_c":t,"band":name,"minimum_period_days":pmin,
                "maximum_period_days":pmax,"integrated_variance_km4":power,
                "fraction_of_detrended_variance_pct":100*power/variance})
        products[t] = {"frequency":freq,"period_days":period,"psd":psd,"ar1_background":bg,
                       "significance_95":sig,"dof":dof,"lag1":lag1}
    return products, pd.DataFrame(summaries), pd.DataFrame(bands)


def run_wavelet(daily: pd.DataFrame) -> tuple[dict[float, dict[str,np.ndarray | float]], pd.DataFrame]:
    products, band_rows = {}, []
    for t,g in daily.groupby("threshold_c"):
        g=g.sort_values("date"); y, lag1 = detrend(g.largest_component_area_km2.to_numpy(float))
        std=float(np.std(y,ddof=1)); z=y/std
        mother=wavelet.Morlet(WAVELET_OMEGA0); s0=2*WAVELET_DT_DAYS
        J=int(np.floor(np.log2(len(z)*WAVELET_DT_DAYS/s0)/WAVELET_DJ))
        W, scales, freqs, coi, _, _ = wavelet.cwt(z,WAVELET_DT_DAYS,WAVELET_DJ,s0,J,mother)
        power=np.abs(W)**2; periods=1/freqs
        signif,_=wavelet.significance(1.0,WAVELET_DT_DAYS,scales,0,lag1,
                                      significance_level=SIGNIFICANCE_LEVEL,wavelet=mother)
        ratio=power/signif[:,None]
        global_power=power.mean(axis=1)
        global_signif,_=wavelet.significance(1.0,WAVELET_DT_DAYS,scales,1,lag1,
            significance_level=SIGNIFICANCE_LEVEL,dof=len(z)-scales,wavelet=mother)
        for name,(pmin,pmax) in SPECTRAL_BANDS_DAYS.items():
            sel=(periods>=pmin)&(periods<=pmax)
            coi_valid=periods[:,None] <= coi[None,:]
            valid=sel[:,None]&coi_valid
            band_rows.append({"threshold_c":t,"band":name,"minimum_period_days":pmin,
                "maximum_period_days":pmax,"valid_coefficients":int(valid.sum()),
                "valid_fraction_pct":float(100*valid.sum()/(sel.sum()*len(z))),
                "mean_normalized_power_inside_coi":float(power[valid].mean()),
                "significant_fraction_inside_coi_pct":float(100*np.mean(ratio[valid]>=1))})
        products[t]={"dates":g.date.to_numpy(),"period_days":periods,"power":power,
            "coi_days":coi,"significance_ratio":ratio,"global_power":global_power,
            "global_significance_95":global_signif,"lag1":lag1,"std_km2":std}
    return products,pd.DataFrame(band_rows)


def select_representative_dates(daily: pd.DataFrame) -> tuple[pd.Timestamp,pd.Timestamp]:
    g=daily[np.isclose(daily.threshold_c,PRIMARY_THRESHOLD_C)].copy()
    med=float(g.largest_component_area_fraction.median())
    connected=g.iloc[(g.largest_component_area_fraction-med).abs().argsort().iloc[0]]
    fragmented=g.loc[g.largest_component_area_fraction.idxmin()]
    return pd.Timestamp(connected.date),pd.Timestamp(fragmented.date)


def load_daily_sst_for_dates(files: list[Path], dates: Sequence[pd.Timestamp], lat: np.ndarray, lon: np.ndarray) -> dict[pd.Timestamp,np.ndarray]:
    result={}
    by_year={d.year for d in dates}
    for f in files:
        year=int(f.stem.split(".")[-1])
        if year not in by_year: continue
        da=open_annual_oisst(f,lat,lon)
        for d in dates:
            if d.year==year:
                result[d]=np.asarray(da.sel(time=d).load().values,float)
        da.close()
    if len(result)!=len(dates): raise ValueError("Could not retrieve all representative SST fields.")
    return result


def save_product_dictionary(products: dict[float, dict[str, np.ndarray | float]], destination: Path) -> None:
    """Serialize threshold-keyed numerical products without Python pickles."""
    payload: dict[str, np.ndarray] = {}
    for threshold, variables in products.items():
        prefix = f"t{threshold:.1f}".replace(".", "p")
        for name, value in variables.items():
            payload[f"{prefix}__{name}"] = np.asarray(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def save_map_snapshots(paths: Paths, daily: pd.DataFrame, files: list[Path],
                       lat: np.ndarray, lon: np.ndarray, pacific: np.ndarray,
                       area: np.ndarray) -> None:
    """Save the two audited SST snapshots required by Figure 1."""
    dates = select_representative_dates(daily)
    fields = load_daily_sst_for_dates(files, dates, lat, lon)
    np.savez_compressed(
        paths.map_snapshots_npz,
        lat=lat,
        lon=lon,
        pacific_mask=pacific.astype(np.uint8),
        cell_area_km2=area,
        dates=np.asarray([d.strftime("%Y-%m-%d") for d in dates], dtype="U10"),
        sst=np.stack([fields[d] for d in dates]).astype(np.float32),
    )


def configure_map(ax, left_labels=True):
    ax.set_extent(MAP_EXTENT,crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,facecolor="0.78",edgecolor="0.2",linewidth=.35,zorder=20)
    ax.coastlines(resolution="110m",linewidth=.45,zorder=21)
    gl=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=.35,color="0.65",linestyle=":")
    gl.top_labels=False; gl.right_labels=False; gl.left_labels=left_labels
    gl.xlocator=mpl.ticker.FixedLocator(np.arange(100,301,30)); gl.ylocator=mpl.ticker.FixedLocator(np.arange(-30,31,15))
    gl.xlabel_style={"size":8}; gl.ylabel_style={"size":8}


def figure1(paths:Paths,daily:pd.DataFrame,files:list[Path],lat:np.ndarray,lon:np.ndarray,pacific:np.ndarray,area:np.ndarray)->list[Path]:
    dates=select_representative_dates(daily); fields=load_daily_sst_for_dates(files,dates,lat,lon)
    lon2,lat2=np.meshgrid(lon,lat); fig,axs=plt.subplots(2,2,figsize=(13,7.4),subplot_kw={"projection":ccrs.PlateCarree()})
    letters="ABCD"
    for row,(date,labelstate) in enumerate(zip(dates,("Representative connected state","Strongly fragmented state"))):
        sst=fields[date]; diag,lcc=daily_components(sst,PRIMARY_THRESHOLD_C,pacific,area,lat,lon)
        canonical=pacific&np.isfinite(sst)&(sst>=PRIMARY_THRESHOLD_C)
        for col in range(2):
            ax=axs[row,col]; configure_map(ax,left_labels=(col==0))
            ax.pcolormesh(lon2,lat2,np.ma.masked_where(~pacific,sst),cmap="Greys",vmin=18,vmax=32,
                          shading="auto",transform=ccrs.PlateCarree(),rasterized=True)
            if col==0:
                ax.contourf(lon2,lat2,canonical.astype(float),levels=[.5,1.5],colors=["none"],
                            hatches=["...."],transform=ccrs.PlateCarree(),zorder=5)
                ax.contour(lon2,lat2,canonical.astype(float),levels=[.5],colors="black",linewidths=1,
                           transform=ccrs.PlateCarree(),zorder=6)
                ax.set_title(f"{labelstate}: canonical PWP")
            else:
                secondary=canonical&~lcc
                ax.contourf(lon2,lat2,lcc.astype(float),levels=[.5,1.5],colors=["0.25"],alpha=.38,
                            transform=ccrs.PlateCarree(),zorder=5)
                ax.contourf(lon2,lat2,secondary.astype(float),levels=[.5,1.5],colors=["none"],hatches=["////"],
                            transform=ccrs.PlateCarree(),zorder=6)
                ax.contour(lon2,lat2,lcc.astype(float),levels=[.5],colors="black",linewidths=1.1,
                           transform=ccrs.PlateCarree(),zorder=7)
                ax.set_title(f"{labelstate}: LCC and secondary area")
            panel_label(ax,f"{letters[row*2+col]})")
            ax.text(.99,.02,f"{date:%d %b %Y}\nCanonical={diag['total_area_km2']/1e6:.2f}; "
                f"LCC={diag['largest_component_area_km2']/1e6:.2f} ×10⁶ km²\n"
                f"fLCC={diag['largest_component_area_fraction']:.3f}; Nc={diag['component_count']}",
                transform=ax.transAxes,ha="right",va="bottom",fontsize=7.7,
                bbox=dict(facecolor="white",edgecolor="0.2",alpha=.80,pad=2))
    fig.suptitle("Canonical and largest-connected 28°C Pacific Warm Pool",y=.995)
    fig.tight_layout()
    return save_figure(fig,paths,"Figure_01_Canonical_vs_LCC_maps")


def figure2(paths:Paths,daily:pd.DataFrame,annual:pd.DataFrame,trends:pd.DataFrame)->list[Path]:
    fig,axs=plt.subplots(3,1,figsize=(13,9),sharex=True)
    for i,(t,ax) in enumerate(zip(THRESHOLDS_C,axs)):
        g=daily[np.isclose(daily.threshold_c,t)].sort_values("date"); a=annual[np.isclose(annual.threshold_c,t)]
        roll=g.set_index("date").largest_component_area_km2.rolling(365,center=True,min_periods=183).mean()/1e6
        ax.plot(g.date,g.largest_component_area_km2/1e6,color="0.82",lw=.35,label="Daily")
        ax.plot(roll.index,roll,color="0.05",lw=1.1,label="365-day mean")
        ax.plot(a.date,a.largest_component_area_km2/1e6,"o",ms=3,mfc="white",mec="black",label="Annual mean")
        tr=trends[(np.isclose(trends.threshold_c,t))&(trends.metric=="largest_component_area_km2")].iloc[0]
        x=a.year.to_numpy(float); slope=tr.theil_sen_slope_per_decade/10
        intercept=np.median(a.largest_component_area_km2.to_numpy()-slope*x)
        ax.plot(a.date,(intercept+slope*x)/1e6,"k--",lw=1,
                label=f"Theil–Sen: {tr.theil_sen_slope_per_decade/1e6:.2f} ×10⁶ km² decade⁻¹")
        ax.set_ylabel(f"≥{t:g}°C LCC area\n(10⁶ km²)"); panel_label(ax,f"{chr(65+i)})"); grid(ax)
        ax.legend(frameon=False,loc="upper left",ncol=4)
    format_long_time_axis(axs[-1]); axs[-1].set_xlabel("Year")
    fig.suptitle("Daily evolution and long-term expansion of PWP-LCC area",y=.995)
    fig.tight_layout()
    return save_figure(fig,paths,"Figure_02_LCC_area_time_series")


def figure3(paths:Paths,daily:pd.DataFrame)->list[Path]:
    fig,axs=plt.subplots(2,2,figsize=(11,8)); metrics=[
        ("largest_component_area_fraction","Fraction in LCC",np.linspace(0,1,51)),
        ("fragmentation_index","Fragmentation index",np.linspace(0,1,51)),
        ("component_count","Number of components",60),
        ("centroid_distance_km","Canonical–LCC centroid distance (km)",60)]
    for j,(ax,(m,label,bins)) in enumerate(zip(axs.ravel(),metrics)):
        for t in THRESHOLDS_C:
            g=daily[np.isclose(daily.threshold_c,t)][m]
            if m=="centroid_distance_km": g=g.clip(upper=g.quantile(.995))
            st=THRESHOLD_STYLES[t]; ax.hist(g,bins=bins,density=True,histtype="step",lw=1.4,
                color=st["color"],linestyle=st["linestyle"],label=f"{t:g}°C")
        ax.set_xlabel(label); ax.set_ylabel("Probability density"); panel_label(ax,f"{chr(65+j)})"); grid(ax)
        ax.legend(frameon=False)
    fig.suptitle("Threshold dependence of PWP connectivity and fragmentation",y=.995)
    fig.tight_layout(); return save_figure(fig,paths,"Figure_03_Connectivity_fragmentation")


def figure4(paths:Paths,annual:pd.DataFrame,trends:pd.DataFrame)->list[Path]:
    specs=[("largest_component_area_km2","LCC area (10⁶ km²)",1e6),
           ("secondary_area_km2","Secondary area (10⁶ km²)",1e6),
           ("largest_component_area_fraction","Fraction in LCC",1),
           ("component_count","Number of components",1)]
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for j,(ax,(m,label,scale)) in enumerate(zip(axs.ravel(),specs)):
        for t in THRESHOLDS_C:
            g=annual[np.isclose(annual.threshold_c,t)]; st=THRESHOLD_STYLES[t]
            ax.plot(g.date,g[m]/scale,color=st["color"],ls=st["linestyle"],marker=st["marker"],
                    ms=2.8,mfc="white",label=f"{t:g}°C")
            tr=trends[(np.isclose(trends.threshold_c,t))&(trends.metric==m)].iloc[0]
            sl=tr.theil_sen_slope_per_decade/10; itc=np.median(g[m]-sl*g.year)
            ax.plot(g.date,(itc+sl*g.year)/scale,color=st["color"],ls="-",lw=.8)
        ax.set_ylabel(label); panel_label(ax,f"{chr(65+j)})"); grid(ax); format_long_time_axis(ax)
    axs[0,0].legend(frameon=False,ncol=3); axs[1,0].set_xlabel("Year"); axs[1,1].set_xlabel("Year")
    fig.suptitle("Annual expansion and spatial reorganization of the PWP-LCC",y=.995)
    fig.tight_layout(); return save_figure(fig,paths,"Figure_04_Annual_LCC_structural_trends")


def figure5(paths:Paths,clim:pd.DataFrame,stl:pd.DataFrame,stl_summary:pd.DataFrame)->list[Path]:
    # Flatten pandas multi-index columns from aggregated climatology if needed.
    c=clim.copy(); c.columns=["_".join([str(x) for x in col if str(x)]) if isinstance(col,tuple) else col for col in c.columns]
    fig,axs=plt.subplots(3,1,figsize=(12,9))
    for t in THRESHOLDS_C:
        st=THRESHOLD_STYLES[t]; g=c[np.isclose(c.threshold_c,t)]
        axs[0].plot(g.month,g.largest_component_area_km2_mean/1e6,color=st["color"],ls=st["linestyle"],
                    marker=st["marker"],mfc="white",label=f"{t:g}°C")
        s=stl[np.isclose(stl.threshold_c,t)]
        axs[1].plot(s.date,s.trend_km2/1e6,color=st["color"],ls=st["linestyle"],label=f"{t:g}°C")
        a=s.assign(year=s.date.dt.year).groupby("year").agg(amplitude=("seasonal_km2",lambda v:v.max()-v.min()),mean=("observed_km2","mean")).reset_index()
        a=a[(a.year>=COMPLETE_YEAR_START)&(a.year<=COMPLETE_YEAR_END)]; a["date"]=pd.to_datetime(a.year.astype(str)+"-07-01")
        axs[2].plot(a.date,100*a.amplitude/a["mean"],color=st["color"],ls=st["linestyle"],marker=st["marker"],ms=2.5,mfc="white",label=f"{t:g}°C")
    axs[0].set_xticks(range(1,13)); axs[0].set_xlabel("Month"); axs[0].set_ylabel("Climatological LCC area\n(10⁶ km²)")
    axs[1].set_ylabel("Robust STL trend\n(10⁶ km²)"); axs[2].set_ylabel("Seasonal amplitude\n(% of annual mean)"); axs[2].set_xlabel("Year")
    for i,ax in enumerate(axs): panel_label(ax,f"{chr(65+i)})"); grid(ax); ax.legend(frameon=False,ncol=3)
    format_long_time_axis(axs[1]); format_long_time_axis(axs[2])
    fig.suptitle("Seasonality and low-frequency evolution of PWP-LCC area",y=.995)
    fig.tight_layout(); return save_figure(fig,paths,"Figure_05_LCC_climatology_STL")


def figure6(paths:Paths,arrays:dict[str,np.ndarray])->list[Path]:
    lat=arrays["lat"]; lon=arrays["lon"]; denom=arrays["valid_day_count"].astype(float)
    lon2,lat2=np.meshgrid(lon,lat); fig,axs=plt.subplots(3,2,figsize=(13,9),subplot_kw={"projection":ccrs.PlateCarree()})
    mappable=None
    for i,t in enumerate(THRESHOLDS_C):
        code=str(t).replace(".","p"); count=arrays[f"lcc_occurrence_count_{code}"].astype(float)
        occurrence=np.divide(100*count,denom,out=np.full_like(count,np.nan),where=denom>0)
        longest=arrays[f"lcc_longest_run_days_{code}"].astype(float)
        for j,(field,title,cmap,vmin,vmax) in enumerate([(occurrence,"LCC occurrence (%)","Greys",0,100),(longest,"Longest continuous LCC membership (days)","Greys",0,np.nanpercentile(longest[longest>0],99))]):
            ax=axs[i,j]; configure_map(ax,left_labels=(j==0)); mappable=ax.pcolormesh(lon2,lat2,np.ma.masked_invalid(field),cmap=cmap,vmin=vmin,vmax=vmax,shading="auto",transform=ccrs.PlateCarree(),rasterized=True)
            ax.set_title(f"SST ≥ {t:g}°C — {title}"); panel_label(ax,f"{chr(65+i*2+j)})")
            cb=fig.colorbar(mappable,ax=ax,orientation="vertical",fraction=.025,pad=.025); cb.ax.tick_params(labelsize=8)
    fig.suptitle("Spatial occurrence and persistence of the Pacific Warm Pool LCC",y=.995)
    fig.tight_layout(); return save_figure(fig,paths,"Figure_06_LCC_occurrence_persistence")


def figure7(paths:Paths,products:dict[float,dict[str,np.ndarray|float]])->list[Path]:
    fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for i,(t,ax) in enumerate(zip(THRESHOLDS_C,axs)):
        p=products[t]; period=np.asarray(p["period_days"])/365.25
        ax.loglog(period,np.asarray(p["psd"]),color="0.05",label="Welch PSD")
        ax.loglog(period,np.asarray(p["significance_95"]),color="0.45",ls="--",label="95% AR(1)")
        for name,(lo,hi) in SPECTRAL_BANDS_DAYS.items(): ax.axvspan(lo/365.25,hi/365.25,color="0.75",alpha=.12)
        ax.set_ylabel(f"≥{t:g}°C PSD\n(km⁴ day)"); panel_label(ax,f"{chr(65+i)})"); grid(ax); ax.legend(frameon=False)
        ax.set_xlim(0.02,20)
    axs[-1].set_xlabel("Period (years)"); fig.suptitle("Welch spectra of PWP-LCC area",y=.995)
    fig.tight_layout(); return save_figure(fig,paths,"Figure_07_LCC_Welch_spectra")


def figure8(paths:Paths,products:dict[float,dict[str,np.ndarray|float]])->list[Path]:
    fig=plt.figure(figsize=(13,10)); gs=fig.add_gridspec(3,2,width_ratios=(5,1.25),hspace=.28,wspace=.12)
    levels=np.linspace(-2,2,17)
    for i,t in enumerate(THRESHOLDS_C):
        p=products[t]; dates=pd.to_datetime(np.asarray(p["dates"])); period=np.asarray(p["period_days"])/365.25
        power=np.asarray(p["power"]); ratio=np.asarray(p["significance_ratio"]); coi=np.asarray(p["coi_days"])/365.25
        ax=fig.add_subplot(gs[i,0]); im=ax.contourf(dates,period,np.log10(np.maximum(power,1e-6)),levels=levels,cmap="Greys",extend="both")
        ax.contour(dates,period,ratio,levels=[1],colors="black",linewidths=.55)
        ax.fill_between(dates,coi,period.max(),color="white",alpha=.55,hatch="//",edgecolor="0.5",linewidth=0)
        ax.set_yscale("log"); ax.set_ylim(16,.03); ax.set_ylabel(f"≥{t:g}°C period\n(years)"); panel_label(ax,f"{chr(65+i*2)})")
        format_long_time_axis(ax)
        if i<2: ax.tick_params(labelbottom=False)
        else: ax.set_xlabel("Year")
        ag=fig.add_subplot(gs[i,1],sharey=ax); ag.plot(np.asarray(p["global_power"]),period,"k-")
        ag.plot(np.asarray(p["global_significance_95"]),period,color="0.5",ls="--")
        ag.set_xscale("log"); ag.grid(True,color="0.85",ls=":",lw=.5); ag.tick_params(labelleft=False); ag.set_xlabel("Global power")
        panel_label(ag,f"{chr(66+i*2)})")
    cbar=fig.colorbar(im,ax=fig.axes,orientation="horizontal",fraction=.025,pad=.04,aspect=50); cbar.set_label("log₁₀ normalized wavelet power")
    fig.suptitle("Morlet wavelet variability of PWP-LCC area",y=.995)
    return save_figure(fig,paths,"Figure_08_LCC_Morlet_wavelet")


def occurrence_summary(arrays:dict[str,np.ndarray],area:np.ndarray)->pd.DataFrame:
    denom=arrays["valid_day_count"].astype(float); rows=[]
    for t in THRESHOLDS_C:
        code=str(t).replace(".","p"); c=arrays[f"lcc_occurrence_count_{code}"].astype(float)
        occ=np.divide(c,denom,out=np.full_like(c,np.nan),where=denom>0); run=arrays[f"lcc_longest_run_days_{code}"].astype(float)
        valid=np.isfinite(occ)&(denom>0)
        rows.append({"threshold_c":t,"n_common_days":int(arrays["n_dates"]),
            "valid_grid_cells":int(valid.sum()),"area_weighted_mean_occurrence_fraction":float(np.nansum(occ*area)/np.sum(area[valid])),
            "grid_cells_occurrence_ge_0p10":int(np.sum(occ>=.10)),"grid_cells_occurrence_ge_0p50":int(np.sum(occ>=.50)),
            "grid_cells_occurrence_ge_0p90":int(np.sum(occ>=.90)),"median_longest_run_days_positive_cells":float(np.median(run[run>0])),
            "p95_longest_run_days_positive_cells":float(np.quantile(run[run>0],.95)),"maximum_longest_run_days":int(np.max(run))})
    return pd.DataFrame(rows)


def disconnected_occurrence_summary(arrays: dict[str, np.ndarray],
                                    area: np.ndarray) -> pd.DataFrame:
    """Summarize warm occurrence outside the daily largest component.

    For threshold T and grid cell x, disconnected occurrence is
    100 * (N_canonical(x,T) - N_LCC(x,T)) / N_valid(x).  The difference is
    non-negative because the LCC is a subset of the canonical threshold mask.
    The area-time fraction is the spherical-area-weighted number of detached
    warm cell-days divided by all canonical warm cell-days.
    """
    denominator = arrays["valid_day_count"].astype(float)
    lat = np.asarray(arrays["lat"], dtype=float)
    lon = np.asarray(arrays["lon"], dtype=float)
    pacific = arrays["pacific_mask"].astype(bool)
    rows: list[dict[str, float | int]] = []
    for threshold in THRESHOLDS_C:
        code = str(threshold).replace(".", "p")
        canonical = arrays[f"canonical_occurrence_count_{code}"].astype(float)
        lcc = arrays[f"lcc_occurrence_count_{code}"].astype(float)
        detached = canonical - lcc
        if np.nanmin(detached[pacific]) < 0:
            raise ValueError(f"LCC count exceeds canonical count at {threshold:g}C.")
        probability = np.divide(
            detached, denominator, out=np.full_like(detached, np.nan),
            where=(denominator > 0) & pacific,
        )
        valid = pacific & np.isfinite(probability) & np.isfinite(area)
        canonical_area_days = float(np.sum(area[valid] * canonical[valid]))
        detached_area_days = float(np.sum(area[valid] * detached[valid]))
        maximum_flat = int(np.nanargmax(np.where(valid, probability, np.nan)))
        maximum_index = np.unravel_index(maximum_flat, probability.shape)
        row: dict[str, float | int] = {
            "threshold_c": threshold,
            "n_common_days": int(arrays["n_dates"]),
            "valid_grid_cells": int(valid.sum()),
            "area_weighted_mean_detached_occurrence_pct": float(
                100.0 * np.sum(area[valid] * probability[valid]) / np.sum(area[valid])
            ),
            "detached_area_time_fraction_of_canonical_pct": float(
                100.0 * detached_area_days / canonical_area_days
            ),
            "maximum_detached_occurrence_pct": float(100.0 * probability[maximum_index]),
            "maximum_latitude_deg_n": float(lat[maximum_index[0]]),
            "maximum_longitude_deg_e": float(lon[maximum_index[1]]),
        }
        for cutoff in (1, 5, 10, 25, 50):
            affected = valid & (100.0 * probability >= cutoff)
            row[f"ocean_area_detached_occurrence_ge_{cutoff}pct_km2"] = float(
                np.sum(area[affected])
            )
            row[f"grid_cells_detached_occurrence_ge_{cutoff}pct"] = int(
                np.sum(affected)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_detachment_report(paths: Paths, summary: pd.DataFrame) -> None:
    lines = [
        "PWP-LCC DISCONNECTED-OCCURRENCE DIAGNOSTIC",
        f"JCLI01 version: {PROGRAM_VERSION}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "DEFINITION",
        "Detached occurrence at a cell is the canonical occurrence count minus the LCC occurrence count, divided by the number of valid SST days.",
        "The area-time fraction is detached warm cell-area-days divided by all canonical warm cell-area-days at the same threshold.",
        "Cell areas are exact spherical grid-cell areas and all fields are restricted to pacific_mask_oisst.npy.",
        "No smoothing, interpolation, morphology, or component-size filter is applied.",
        "",
        "SUMMARY",
        summary.to_string(index=False),
    ]
    paths.detachment_report.write_text("\n".join(lines), encoding="utf-8")


def write_report(paths:Paths,daily:pd.DataFrame,trends:pd.DataFrame,stl_summary:pd.DataFrame,
                 spectral_summary:pd.DataFrame,spectral_bands:pd.DataFrame,wavelet_bands:pd.DataFrame,
                 occurrence:pd.DataFrame,generated:list[Path]) -> None:
    lines=[PROGRAM_NAME,f"Version: {PROGRAM_VERSION}",f"Generated: {datetime.now(timezone.utc).isoformat()}","",
        "RECORD",f"Dates: {daily.date.min().date()} to {daily.date.max().date()}",f"Rows: {len(daily):,}",
        f"Thresholds: {', '.join(map(str,THRESHOLDS_C))}","Connectivity: 8-neighbor with periodic-longitude seam union",
        "Component ranking: exact spherical area", "No interpolation, smoothing, morphology, or component filtering.","",
        "TREND STATISTICS",trends.to_string(index=False),"","STL SUMMARY",stl_summary.to_string(index=False),"",
        "WELCH SUMMARY",spectral_summary.to_string(index=False),"","WELCH BAND VARIANCE",spectral_bands.to_string(index=False),"",
        "WAVELET BAND DIAGNOSTICS",wavelet_bands.to_string(index=False),"","OCCURRENCE/PERSISTENCE",occurrence.to_string(index=False),"",
        "GENERATED FILES",*[str(p.resolve()) for p in generated]]
    paths.report.write_text("\n".join(lines),encoding="utf-8")


def build_manifest(paths:Paths) -> pd.DataFrame:
    files=[p for folder in (paths.tables,paths.reports,paths.metadata,paths.cache) for p in folder.rglob("*") if p.is_file() and p!=paths.manifest]
    if paths.occurrence_npz.is_file(): files.append(paths.occurrence_npz)
    manifest=pd.DataFrame([{"relative_path":str(p.relative_to(paths.root)).replace("\\","/"),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(files)])
    atomic_csv(manifest,paths.manifest); return manifest


def scientific_products(paths: Paths) -> list[Path]:
    """Return the authoritative JCLI01 products expected by JCLI02."""
    products = [
        paths.daily, paths.annual, paths.trend, paths.climatology,
        paths.stl_daily, paths.stl_summary, paths.occurrence_npz,
        paths.occurrence_summary, paths.detachment_summary,
        paths.detachment_report, paths.spectral_summary,
        paths.spectral_bands, paths.wavelet_bands, paths.spectral_npz,
        paths.wavelet_npz, paths.map_snapshots_npz, paths.report,
    ]
    return products


def make_analysis_metadata(paths: Paths, daily: pd.DataFrame,
                           oisst_files: Sequence[Path],
                           generated: Sequence[Path]) -> dict:
    """Build metadata for the separated, calculation-only JCLI01 workflow."""
    return {
        "metadata_schema": "JCLI-PWP-LCC-analysis-metadata/1.1",
        "program": PROGRAM_NAME,
        "version": PROGRAM_VERSION,
        "program_date": PROGRAM_DATE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(paths.root),
        "program_file": str(Path(__file__).resolve()),
        "program_basename": Path(__file__).name,
        "program_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "thresholds_c": THRESHOLDS_C,
        "record": {
            "start": str(pd.to_datetime(daily["date"]).min().date()),
            "end": str(pd.to_datetime(daily["date"]).max().date()),
            "n_days_per_threshold": int(daily.groupby("threshold_c").size().min()),
        },
        "methods": {
            "connectivity": CONNECTIVITY_LABEL,
            "component_ranking": "exact spherical area",
            "periodic_longitude": True,
            "interpolation": False,
            "smoothing": False,
            "morphology": False,
            "complete_years": [COMPLETE_YEAR_START, COMPLETE_YEAR_END],
            "stl_period_days": STL_PERIOD_DAYS,
            "welch_segment_days": WELCH_SEGMENT_DAYS,
            "welch_overlap_days": WELCH_OVERLAP_DAYS,
            "wavelet": "Morlet",
            "omega0": WAVELET_OMEGA0,
            "dj": WAVELET_DJ,
            "bands_days": SPECTRAL_BANDS_DAYS,
        },
        "workflow_role": "scientific calculations only; figures are generated by JCLI02",
        "inputs": {
            "mask": str(paths.mask),
            "grid_lat": str(paths.grid_lat),
            "grid_lon": str(paths.grid_lon),
            "annual_oisst_files": [str(path) for path in oisst_files],
        },
        "outputs": [str(path) for path in generated if path.is_file()],
    }


def refresh_provenance(paths: Paths, start: pd.Timestamp,
                       end: pd.Timestamp) -> None:
    """Refresh provenance without altering any scientific product.

    This operation is intentionally read-only with respect to numerical and
    spatial results. It verifies the existing product inventory, validates the
    daily record, then replaces only the JCLI01 metadata and checksum manifest.
    """
    required = scientific_products(paths)
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Cannot refresh provenance because JCLI01 products are missing:\n" + formatted
        )
    daily = pd.read_csv(paths.daily, parse_dates=["date"], low_memory=False)
    required_columns = {"date", "threshold_c"}
    absent_columns = required_columns.difference(daily.columns)
    if absent_columns:
        raise ValueError(f"Daily diagnostics lack columns: {sorted(absent_columns)}")
    observed_thresholds = tuple(sorted(daily["threshold_c"].astype(float).unique()))
    if observed_thresholds != THRESHOLDS_C:
        raise ValueError(
            f"Daily thresholds {observed_thresholds} differ from {THRESHOLDS_C}."
        )
    counts = daily.groupby("threshold_c").size()
    if counts.nunique() != 1 or int(counts.iloc[0]) != EXPECTED_N_DAYS:
        raise ValueError(f"Unexpected daily counts by threshold: {counts.to_dict()}")
    if daily["date"].min() != EXPECTED_START or daily["date"].max() != EXPECTED_END:
        raise ValueError(
            f"Unexpected record {daily['date'].min()} to {daily['date'].max()}."
        )
    oisst_files = discover_oisst_files(paths, start, end)
    metadata = make_analysis_metadata(paths, daily, oisst_files, required)
    save_json(metadata, paths.metadata_json)
    build_manifest(paths)
    logging.info("Refreshed JCLI01 provenance only; scientific products were unchanged.")
    logging.info("Metadata: %s", paths.metadata_json)
    logging.info("Manifest: %s", paths.manifest)


def main() -> int:
    args=parse_args(); configure_logging(args.log_level); configure_matplotlib()
    paths=resolve_paths(args.project_root); ensure_directories(paths)
    start,end=pd.Timestamp(args.start_date),pd.Timestamp(args.end_date)
    if start>end: raise ValueError("start-date must not exceed end-date")
    if args.refresh_provenance_only:
        refresh_provenance(paths, start, end)
        return 0
    lat,lon,pacific,area=load_reference_grid(paths)
    if args.derive_detachment_only:
        if not paths.occurrence_npz.is_file():
            raise FileNotFoundError(
                f"Required occurrence cache not found: {paths.occurrence_npz}"
            )
        arrays = dict(np.load(paths.occurrence_npz, allow_pickle=False))
        summary = disconnected_occurrence_summary(arrays, area)
        atomic_csv(summary, paths.detachment_summary, float_format="%.10g")
        write_detachment_report(paths, summary)
        daily = pd.read_csv(paths.daily, parse_dates=["date"], low_memory=False)
        oisst_files = discover_oisst_files(paths, start, end)
        generated = scientific_products(paths)
        metadata = make_analysis_metadata(paths, daily, oisst_files, generated)
        save_json(metadata, paths.metadata_json)
        build_manifest(paths)
        logging.info("Disconnected-occurrence products written without reopening OISST.")
        logging.info("Table: %s", paths.detachment_summary)
        logging.info("Report: %s", paths.detachment_report)
        logging.info("JCLI01 metadata and checksum manifest were reconciled to v%s.", PROGRAM_VERSION)
        return 0
    files=[] if args.no_raw_maps else discover_oisst_files(paths,start,end)
    daily,arrays=load_or_build_daily(paths,args,files,lat,lon,pacific,area)
    atomic_csv(daily,paths.daily,float_format="%.10g")
    if arrays is not None:
        np.savez_compressed(paths.occurrence_npz,**arrays)
    annual=complete_year_annual(daily); atomic_csv(annual,paths.annual,float_format="%.10g")
    trends=trend_statistics(annual); atomic_csv(trends,paths.trend,float_format="%.10g")
    clim=monthly_climatology(daily); atomic_csv(clim,paths.climatology,float_format="%.10g")
    stl,stl_summary=run_stl(daily); atomic_csv(stl,paths.stl_daily,float_format="%.10g"); atomic_csv(stl_summary,paths.stl_summary,float_format="%.10g")
    welch_products,welch_summary,welch_bands=run_welch(daily)
    atomic_csv(welch_summary,paths.spectral_summary,float_format="%.10g"); atomic_csv(welch_bands,paths.spectral_bands,float_format="%.10g")
    save_product_dictionary(welch_products,paths.spectral_npz)
    wave_products,wave_bands=run_wavelet(daily); atomic_csv(wave_bands,paths.wavelet_bands,float_format="%.10g")
    save_product_dictionary(wave_products,paths.wavelet_npz)
    if arrays is not None:
        occ=occurrence_summary(arrays,area); atomic_csv(occ,paths.occurrence_summary,float_format="%.10g")
        detached=disconnected_occurrence_summary(arrays,area)
        atomic_csv(detached,paths.detachment_summary,float_format="%.10g")
        write_detachment_report(paths,detached)
    else: occ=pd.DataFrame()
    if not args.no_raw_maps:
        save_map_snapshots(paths,daily,files,lat,lon,pacific,area)
    generated=[paths.daily,paths.annual,paths.trend,paths.climatology,paths.stl_daily,
        paths.stl_summary,paths.spectral_summary,paths.spectral_bands,paths.wavelet_bands,
        paths.spectral_npz,paths.wavelet_npz]
    if arrays is not None: generated.extend([
        paths.occurrence_npz,paths.occurrence_summary,
        paths.detachment_summary,paths.detachment_report,
    ])
    if paths.map_snapshots_npz.is_file(): generated.append(paths.map_snapshots_npz)
    metadata=make_analysis_metadata(paths,daily,files,generated)
    save_json(metadata,paths.metadata_json)
    write_report(paths,daily,trends,stl_summary,welch_summary,welch_bands,wave_bands,occ,generated)
    build_manifest(paths)
    logging.info("Completed %s v%s",PROGRAM_NAME,PROGRAM_VERSION)
    logging.info("Numerical analysis completed. Tables: %s",paths.tables)
    logging.info("Run JCLI02_figures_pwp_lcc_publication_v1_1_0.py to create figures.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Fatal error")
        raise
