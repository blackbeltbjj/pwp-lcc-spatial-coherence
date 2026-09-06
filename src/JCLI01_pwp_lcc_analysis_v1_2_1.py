#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JCLI01_pwp_lcc_analysis_v1_2_1.py
==============================================================================
PURPOSE
    Additive scientific extension for the Journal of Climate Pacific Warm Pool
    Largest Connected Component (PWP-LCC) study.

    This version performs the historical-definition comparison motivated by
    Yan et al. (1992) and explicitly documented by Yan et al. (1993):

        Yan-style WPWP = SST > 28 degC inside the rectangular box
                         20 degS-20 degN, 120 degE-150 degW.

    The program compares, on the SAME NOAA OISST v2.1 fields:

        1) PWP-LCC at SST >= 28 degC inside the validated Pacific mask;
        2) Canonical PWP at SST >= 28 degC inside the validated Pacific mask;
        3) Yan-style fixed-domain WPWP at SST > 28 degC inside
           20 degS-20 degN, 120 degE-150 degW;
        4) Yan-style >=28 degC sensitivity using the same fixed box.

    For each definition it calculates daily area and area-weighted mean SST.
    Spherical centroids are calculated for LCC, canonical PWP, and the primary
    Yan-style (>28 degC) definition. It additionally quantifies geometric
    overlap and the amount of the LCC lying outside the Yan fixed box.

SCIENTIFIC MOTIVATION
    Yan et al. (1993), responding to a technical comment on Yan et al. (1992),
    state that their WPWP was defined as SST higher than 28 degC inside a
    rectangular box from 120 degE to 150 degW and 20 degS to 20 degN.
    They also state that yearly mean MCSST of the WPWP was used to track
    variations in WPWP temperature and size.

    The present diagnostic therefore tests how diagnosed PWP area, temperature,
    and location depend on a fixed geographical window versus an explicit
    topological criterion (the daily largest connected component).

IMPORTANT
    - This is a NEW scientific version. It does NOT overwrite JCLI01 v1.1.0
      numerical products.
    - Existing JCLI01 v1.1.0 daily diagnostics are used only as an independent
      QC target when available.
    - The validated Pacific mask is required for canonical/LCC calculations and
      is never reconstructed.
    - The Yan-style diagnostic intentionally uses the historical rectangular
      window itself (ocean cells with finite OISST inside the box), rather than
      using the Pacific mask to define the historical box.
    - No interpolation, smoothing, morphology, or small-component filtering.
    - Eight-neighbor connectivity includes periodic 0/360 longitude continuity.
    - Exact spherical grid-cell areas use R = 6371.0088 km.
    - Complete-year summaries use 1982-2025.
    - A dedicated historical comparison summary uses 1982-1991.

INPUTS (relative to PROJECT_ROOT)
    data/raw/sst.day.mean.1981.nc ... sst.day.mean.2026.nc
    data/processed/pacific_mask_oisst.npy
    data/processed/grid_lat.npy
    data/processed/grid_lon.npy

OPTIONAL QC INPUT
    outputs/JCLI/tables/jcli_pwp_lcc/jcli_pwp_lcc_daily_diagnostics.csv
    or
    outputs/tables/threshold_comparison/pwp_long_term_connectivity/
        pwp_daily_connectivity_diagnostics.csv

OUTPUTS
    outputs/JCLI/tables/jcli_pwp_lcc/historical_yan1992_v1_2_1/
        JCLI_pwp_lcc_yan1992_daily_comparison_v1_2_1.csv
        JCLI_pwp_lcc_yan1992_annual_summary_v1_2_1.csv
        JCLI_pwp_lcc_yan1992_period_summary_v1_2_1.csv
        JCLI_pwp_lcc_yan1992_trend_summary_v1_2_1.csv
        JCLI_pwp_lcc_yan1992_qc_summary_v1_2_1.csv

    outputs/JCLI/reports/jcli_pwp_lcc/
        JCLI01_v1_2_1_YAN1992_COMPARISON_REPORT.txt

    outputs/JCLI/metadata/jcli_pwp_lcc/
        JCLI01_v1_2_1_YAN1992_METADATA.json
        JCLI01_v1_2_1_YAN1992_SHA256.csv

    outputs/JCLI/cache/jcli_pwp_lcc/yan1992_v1_2_0/
        year_YYYY.csv

REFERENCES DEFINING THE HISTORICAL WINDOW
    Yan, X.-H., C.-R. Ho, Q. Zheng, and V. Klemas, 1992:
        Temperature and size variabilities of the western Pacific warm pool.
        Science, 258, 1643-1645.

    Yan, X.-H., C.-R. Ho, Q. Zheng, and V. Klemas, 1993:
        Response: Using Satellite Infrared Data in Studies of Variabilities of
        the Western Pacific Warm Pool. Science, 262, 441.
        The response explicitly states: SST > 28 degC inside
        120 degE-150 degW, 20 degS-20 degN.

AUTHOR
    Fabio Vieira Machado

VERSION
    1.2.1 (2026-09-06; QC-serialization tolerance correction; numerical science unchanged)
==============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from scipy.stats import theilslopes


PROGRAM_NAME = "JCLI01 PWP-LCC HISTORICAL DEFINITION COMPARISON"
PROGRAM_VERSION = "1.2.1"
PROGRAM_DATE = "2026-09-06"

THRESHOLD_C = 28.0
EXPECTED_START = pd.Timestamp("1981-09-01")
EXPECTED_END = pd.Timestamp("2026-07-29")
EXPECTED_N_DAYS = 16_403

COMPLETE_YEAR_START = 1982
COMPLETE_YEAR_END = 2025
YAN_PERIOD_START = 1982
YAN_PERIOD_END = 1991

YAN_LAT_MIN = -20.0
YAN_LAT_MAX = 20.0
YAN_LON_MIN_E = 120.0
YAN_LON_MAX_E = 210.0       # 150 degW = 210 degE

EARTH_RADIUS_KM = 6371.0088
CONNECTIVITY_STRUCTURE = np.ones((3, 3), dtype=np.uint8)
BLOCK_DAYS = 31

INTERNAL_QC_ABS_TOL_KM2 = 1e-4
REFERENCE_QC_ABS_TOL_KM2 = 1e-2
QC_REL_TOL = 1e-11

# JCLI01 v1.1.0 stores daily CSV values with float_format="%.10g".
# At PWP areas of order 10^7-10^8 km2, this serialization can round the
# stored reference value by about 0.005 km2. The 0.01-km2 reference
# tolerance therefore reflects the precision preserved in the authoritative
# CSV only; it does not relax the scientific calculation. Internal QC remains
# at 1e-4 km2.


def parse_args() -> argparse.Namespace:
    default_root = (
        Path(__file__).resolve().parents[1]
        if Path(__file__).resolve().parent.name == "src"
        else Path.cwd()
    )
    p = argparse.ArgumentParser(description=PROGRAM_NAME)
    p.add_argument("--project-root", type=Path, default=default_root)
    p.add_argument("--start-date", default=str(EXPECTED_START.date()))
    p.add_argument("--end-date", default=str(EXPECTED_END.date()))
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore completed yearly checkpoints and recompute all requested years.",
    )
    p.add_argument(
        "--no-qc-reference",
        action="store_true",
        help="Do not compare recomputed 28C canonical/LCC area against existing JCLI01/Program31 daily diagnostics.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, **kwargs)
    tmp.replace(path)


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def identify_name(names: Iterable[str], candidates: tuple[str, ...], kind: str) -> str:
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"Unable to identify {kind}; available names: {sorted(names)}")


def coordinate_edges(coord: np.ndarray) -> np.ndarray:
    x = np.asarray(coord, dtype=float)
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("Coordinates must be strictly increasing one-dimensional arrays.")
    mid = 0.5 * (x[1:] + x[:-1])
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = mid
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges


def spherical_cell_areas_km2(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_edges = np.clip(coordinate_edges(lat), -90.0, 90.0)
    lon_edges = coordinate_edges(lon)
    lat_e = np.deg2rad(lat_edges)
    lon_e = np.deg2rad(lon_edges)
    dlon = np.diff(lon_e)
    lat_factor = np.sin(lat_e[1:]) - np.sin(lat_e[:-1])
    area = EARTH_RADIUS_KM**2 * lat_factor[:, None] * dlon[None, :]
    if np.any(~np.isfinite(area)) or np.any(area <= 0):
        raise ValueError("Invalid spherical cell areas.")
    return area


def resolve_project_paths(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    proc = root / "data" / "processed"
    out = root / "outputs" / "JCLI"
    table_dir = out / "tables" / "jcli_pwp_lcc" / "historical_yan1992_v1_2_1"
    report_dir = out / "reports" / "jcli_pwp_lcc"
    metadata_dir = out / "metadata" / "jcli_pwp_lcc"
    cache_dir = out / "cache" / "jcli_pwp_lcc" / "yan1992_v1_2_0"
    return {
        "root": root,
        "raw": root / "data" / "raw",
        "mask": proc / "pacific_mask_oisst.npy",
        "grid_lat": proc / "grid_lat.npy",
        "grid_lon": proc / "grid_lon.npy",
        "table_dir": table_dir,
        "report_dir": report_dir,
        "metadata_dir": metadata_dir,
        "cache_dir": cache_dir,
        "daily": table_dir / "JCLI_pwp_lcc_yan1992_daily_comparison_v1_2_1.csv",
        "annual": table_dir / "JCLI_pwp_lcc_yan1992_annual_summary_v1_2_1.csv",
        "period": table_dir / "JCLI_pwp_lcc_yan1992_period_summary_v1_2_1.csv",
        "trend": table_dir / "JCLI_pwp_lcc_yan1992_trend_summary_v1_2_1.csv",
        "qc": table_dir / "JCLI_pwp_lcc_yan1992_qc_summary_v1_2_1.csv",
        "report": report_dir / "JCLI01_v1_2_1_YAN1992_COMPARISON_REPORT.txt",
        "metadata": metadata_dir / "JCLI01_v1_2_1_YAN1992_METADATA.json",
        "manifest": metadata_dir / "JCLI01_v1_2_1_YAN1992_SHA256.csv",
        "jcli11_daily": out / "tables" / "jcli_pwp_lcc" / "jcli_pwp_lcc_daily_diagnostics.csv",
        "program31_daily": (
            root / "outputs" / "tables" / "threshold_comparison"
            / "pwp_long_term_connectivity" / "pwp_daily_connectivity_diagnostics.csv"
        ),
    }


def ensure_dirs(paths: dict[str, Path]) -> None:
    for key in ("table_dir", "report_dir", "metadata_dir", "cache_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


def load_reference_grid(paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for key in ("mask", "grid_lat", "grid_lon"):
        if not paths[key].is_file():
            raise FileNotFoundError(
                f"Required validated spatial input missing: {paths[key]}\n"
                "Do not reconstruct the Pacific mask."
            )
    mask = np.asarray(np.load(paths["mask"]), dtype=bool)
    lat = np.asarray(np.load(paths["grid_lat"]), dtype=float)
    lon = np.mod(np.asarray(np.load(paths["grid_lon"]), dtype=float), 360.0)

    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    mask = mask[:, lon_order]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        mask = mask[::-1, :]

    if mask.shape != (lat.size, lon.size):
        raise ValueError(f"Mask shape {mask.shape} != grid {(lat.size, lon.size)}")
    if np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0):
        raise ValueError("Reference grid must be strictly increasing.")

    area = spherical_cell_areas_km2(lat, lon)
    logging.info(
        "Reference grid: %d x %d; Pacific-mask ocean cells=%d",
        lat.size, lon.size, int(mask.sum())
    )
    return lat, lon, mask, area


def discover_oisst_files(paths: dict[str, Path], start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    files: list[Path] = []
    for year in range(start.year, end.year + 1):
        p = paths["raw"] / f"sst.day.mean.{year}.nc"
        if not p.is_file():
            raise FileNotFoundError(f"Missing OISST file: {p}")
        files.append(p)
    return files


def open_annual_oisst(path: Path, target_lat: np.ndarray, target_lon: np.ndarray) -> xr.DataArray:
    ds = xr.open_dataset(path, chunks={}, decode_times=True, mask_and_scale=True)
    sst_name = identify_name(
        tuple(ds.data_vars),
        ("sst", "SST", "sea_surface_temperature", "analysed_sst"),
        "SST variable",
    )
    lat_name = identify_name(tuple(ds.coords), ("lat", "latitude", "y"), "latitude")
    lon_name = identify_name(tuple(ds.coords), ("lon", "longitude", "x"), "longitude")
    time_name = identify_name(tuple(ds.coords), ("time", "date"), "time")

    da = ds[sst_name]
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
        raise ValueError(f"Latitude mismatch in {path}; interpolation is forbidden.")
    if not np.allclose(da[lon_name].values, target_lon, atol=1e-8, rtol=0):
        raise ValueError(f"Longitude mismatch in {path}; interpolation is forbidden.")

    da = da.rename({time_name: "time", lat_name: "lat", lon_name: "lon"})
    return da.chunk({"time": BLOCK_DAYS, "lat": -1, "lon": -1})


class UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n + 1, dtype=np.int32)

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = int(self.p[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def label_periodic_8(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, n = ndimage.label(mask, structure=CONNECTIVITY_STRUCTURE)
    if n == 0:
        return labels.astype(np.int32, copy=False), 0

    uf = UnionFind(n)
    ny = mask.shape[0]
    for i in range(ny):
        if not mask[i, 0]:
            continue
        for j in (i - 1, i, i + 1):
            if 0 <= j < ny and mask[j, -1]:
                uf.union(int(labels[i, 0]), int(labels[j, -1]))

    roots = np.arange(n + 1, dtype=np.int32)
    for k in range(1, n + 1):
        roots[k] = uf.find(k)

    merged = roots[labels]
    unique = np.unique(merged[merged > 0])
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[unique] = np.arange(1, unique.size + 1)
    return remap[merged], int(unique.size)


def largest_component(mask: np.ndarray, area: np.ndarray) -> tuple[np.ndarray, int, float]:
    labels, n = label_periodic_8(mask)
    if n == 0:
        raise ValueError("No connected component found at 28C.")
    component_area = np.bincount(
        labels.ravel(), weights=area.ravel(), minlength=n + 1
    )
    component_area[0] = 0.0
    largest_id = int(np.argmax(component_area))
    return labels == largest_id, n, float(component_area[largest_id])


def build_spherical_basis(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat2, lon2 = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    return (
        np.cos(lat2) * np.cos(lon2),
        np.cos(lat2) * np.sin(lon2),
        np.sin(lat2),
    )


def mask_metrics(
    mask: np.ndarray,
    sst: np.ndarray,
    area: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    if not np.any(mask):
        return {
            "area_km2": math.nan,
            "sst_area_integral_c_km2": math.nan,
            "mean_sst_c": math.nan,
            "centroid_lon_360": math.nan,
            "centroid_lat": math.nan,
        }

    a = area[mask]
    v = sst[mask]
    total = float(a.sum())
    sst_integral = float(np.dot(v, a))
    out = {
        "area_km2": total,
        "sst_area_integral_c_km2": sst_integral,
        "mean_sst_c": sst_integral / total,
        "centroid_lon_360": math.nan,
        "centroid_lat": math.nan,
    }

    if basis is not None:
        bx, by, bz = basis
        x = float(np.sum(area[mask] * bx[mask]) / total)
        y = float(np.sum(area[mask] * by[mask]) / total)
        z = float(np.sum(area[mask] * bz[mask]) / total)
        out["centroid_lon_360"] = float(np.mod(np.rad2deg(np.arctan2(y, x)), 360.0))
        out["centroid_lat"] = float(np.rad2deg(np.arctan2(z, np.hypot(x, y))))
    return out


def great_circle_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    if not all(np.isfinite([lon1, lat1, lon2, lat2])):
        return math.nan
    a1, a2 = np.deg2rad([lat1, lat2])
    dl = np.deg2rad(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    x = (
        np.sin((a2 - a1) / 2.0) ** 2
        + np.cos(a1) * np.cos(a2) * np.sin(dl / 2.0) ** 2
    )
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(x, 0.0, 1.0))))


def yan_geographic_box(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_sel = (lat >= YAN_LAT_MIN - 1e-10) & (lat <= YAN_LAT_MAX + 1e-10)
    lon_sel = (lon >= YAN_LON_MIN_E - 1e-10) & (lon <= YAN_LON_MAX_E + 1e-10)
    box = lat_sel[:, None] & lon_sel[None, :]
    if not np.any(box):
        raise ValueError("Yan-style geographic box contains no grid cells.")
    return box


def flatten(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def process_year(
    nc: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    pacific: np.ndarray,
    area: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    yan_box: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    da = open_annual_oisst(nc, lat, lon).sel(time=slice(start, end))

    for k0 in range(0, da.sizes["time"], BLOCK_DAYS):
        block = da.isel(
            time=slice(k0, min(k0 + BLOCK_DAYS, da.sizes["time"]))
        ).load()

        for k in range(block.sizes["time"]):
            date = pd.Timestamp(block.time.values[k]).normalize()
            sst = np.asarray(block.isel(time=k).values, dtype=np.float64)

            valid_pacific = pacific & np.isfinite(sst)
            canonical = valid_pacific & (sst >= THRESHOLD_C)
            lcc, component_count, lcc_area_from_label = largest_component(canonical, area)

            # Historical Yan-style definition:
            # authors explicitly said "higher than 28C" inside the fixed box.
            yan_gt28 = yan_box & np.isfinite(sst) & (sst > THRESHOLD_C)

            # Sensitivity using the manuscript's inclusive threshold convention.
            yan_ge28 = yan_box & np.isfinite(sst) & (sst >= THRESHOLD_C)

            canonical_m = mask_metrics(canonical, sst, area, basis)
            lcc_m = mask_metrics(lcc, sst, area, basis)
            yan_gt_m = mask_metrics(yan_gt28, sst, area, basis)
            yan_ge_m = mask_metrics(yan_ge28, sst, area, None)

            if not np.isclose(
                lcc_m["area_km2"],
                lcc_area_from_label,
                atol=INTERNAL_QC_ABS_TOL_KM2,
                rtol=QC_REL_TOL,
            ):
                raise RuntimeError(
                    f"LCC area internal QC failed on {date.date()}: "
                    f"{lcc_m['area_km2']} vs {lcc_area_from_label}"
                )

            intersection = lcc & yan_gt28
            union = lcc | yan_gt28
            intersection_area = float(area[intersection].sum())
            union_area = float(area[union].sum())
            lcc_outside_box_area = float(area[lcc & ~yan_box].sum())
            yan_outside_lcc_area = float(area[yan_gt28 & ~lcc].sum())
            canonical_outside_box_area = float(area[canonical & ~yan_box].sum())

            row: dict[str, object] = {
                "date": date.date().isoformat(),
                "year": date.year,
                "threshold_c": THRESHOLD_C,
                "component_count_28c": component_count,
                **flatten("canonical", canonical_m),
                **flatten("lcc", lcc_m),
                **flatten("yan_gt28", yan_gt_m),
                **flatten("yan_ge28_sensitivity", yan_ge_m),
                "lcc_minus_yan_gt28_area_km2": lcc_m["area_km2"] - yan_gt_m["area_km2"],
                "lcc_minus_yan_gt28_mean_sst_c": lcc_m["mean_sst_c"] - yan_gt_m["mean_sst_c"],
                "canonical_minus_yan_gt28_area_km2": canonical_m["area_km2"] - yan_gt_m["area_km2"],
                "lcc_yan_gt28_centroid_distance_km": great_circle_km(
                    lcc_m["centroid_lon_360"], lcc_m["centroid_lat"],
                    yan_gt_m["centroid_lon_360"], yan_gt_m["centroid_lat"],
                ),
                "lcc_yan_gt28_intersection_area_km2": intersection_area,
                "lcc_yan_gt28_union_area_km2": union_area,
                "lcc_yan_gt28_area_jaccard": intersection_area / union_area if union_area > 0 else math.nan,
                "lcc_area_outside_yan_box_km2": lcc_outside_box_area,
                "lcc_area_outside_yan_box_fraction": (
                    lcc_outside_box_area / lcc_m["area_km2"]
                    if lcc_m["area_km2"] > 0 else math.nan
                ),
                "yan_gt28_area_outside_lcc_km2": yan_outside_lcc_area,
                "yan_gt28_area_outside_lcc_fraction": (
                    yan_outside_lcc_area / yan_gt_m["area_km2"]
                    if yan_gt_m["area_km2"] > 0 else math.nan
                ),
                "canonical_area_outside_yan_box_km2": canonical_outside_box_area,
                "canonical_area_outside_yan_box_fraction": (
                    canonical_outside_box_area / canonical_m["area_km2"]
                    if canonical_m["area_km2"] > 0 else math.nan
                ),
                "yan_operator_sensitivity_area_km2": (
                    yan_ge_m["area_km2"] - yan_gt_m["area_km2"]
                ),
                "yan_operator_sensitivity_mean_sst_c": (
                    yan_ge_m["mean_sst_c"] - yan_gt_m["mean_sst_c"]
                ),
            }
            records.append(row)

        logging.info(
            "%s | processed %d/%d days",
            nc.name,
            min(k0 + BLOCK_DAYS, da.sizes["time"]),
            da.sizes["time"],
        )

    da.close()
    return pd.DataFrame.from_records(records)


def expected_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="D")


def validate_daily(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    f = frame.copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values("date").reset_index(drop=True)
    exp = expected_dates(start, end)
    if len(f) != len(exp) or not pd.DatetimeIndex(f["date"]).equals(exp):
        raise ValueError(
            f"Historical comparison is not a complete daily record: "
            f"got {len(f)}, expected {len(exp)}."
        )

    numerical_checks = [
        "canonical_area_km2",
        "lcc_area_km2",
        "yan_gt28_area_km2",
        "yan_ge28_sensitivity_area_km2",
        "canonical_mean_sst_c",
        "lcc_mean_sst_c",
        "yan_gt28_mean_sst_c",
    ]
    if f[numerical_checks].isna().any().any():
        bad = f[numerical_checks].isna().sum()
        raise ValueError(f"Unexpected missing daily metrics:\n{bad[bad > 0]}")

    if np.any(f["lcc_area_km2"] > f["canonical_area_km2"] + INTERNAL_QC_ABS_TOL_KM2):
        raise ValueError("LCC area exceeds canonical area.")
    if np.any(
        f["yan_gt28_area_km2"]
        > f["yan_ge28_sensitivity_area_km2"] + INTERNAL_QC_ABS_TOL_KM2
    ):
        raise ValueError("Strict Yan >28 area exceeds Yan >=28 sensitivity area.")

    return f


def load_qc_reference(paths: dict[str, Path]) -> pd.DataFrame | None:
    candidate = None
    if paths["jcli11_daily"].is_file():
        candidate = paths["jcli11_daily"]
    elif paths["program31_daily"].is_file():
        candidate = paths["program31_daily"]
    if candidate is None:
        return None

    ref = pd.read_csv(candidate, low_memory=False)
    ref["date"] = pd.to_datetime(ref["date"])
    if "threshold_c" not in ref:
        raise ValueError(f"QC reference lacks threshold_c: {candidate}")
    ref = ref[np.isclose(pd.to_numeric(ref["threshold_c"]), THRESHOLD_C)].copy()
    required = {"date", "total_area_km2", "largest_component_area_km2"}
    if not required.issubset(ref.columns):
        raise ValueError(f"QC reference missing {sorted(required - set(ref.columns))}: {candidate}")
    logging.info("Independent area-QC reference: %s", candidate)
    return ref[list(required)].sort_values("date")


def qc_against_reference(
    daily: pd.DataFrame,
    reference: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    rows.append({
        "test": "daily_record_complete",
        "value": int(len(daily)),
        "tolerance": EXPECTED_N_DAYS if (
            daily["date"].min() == EXPECTED_START and daily["date"].max() == EXPECTED_END
        ) else len(daily),
        "pass": True,
    })
    rows.append({
        "test": "lcc_never_exceeds_canonical",
        "value": float((daily["lcc_area_km2"] - daily["canonical_area_km2"]).max()),
        "tolerance": INTERNAL_QC_ABS_TOL_KM2,
        "pass": bool(np.all(daily["lcc_area_km2"] <= daily["canonical_area_km2"] + INTERNAL_QC_ABS_TOL_KM2)),
    })
    rows.append({
        "test": "yan_gt28_never_exceeds_yan_ge28",
        "value": float((daily["yan_gt28_area_km2"] - daily["yan_ge28_sensitivity_area_km2"]).max()),
        "tolerance": INTERNAL_QC_ABS_TOL_KM2,
        "pass": bool(np.all(daily["yan_gt28_area_km2"] <= daily["yan_ge28_sensitivity_area_km2"] + INTERNAL_QC_ABS_TOL_KM2)),
    })

    if reference is None:
        rows.append({
            "test": "independent_jcli11_or_program31_area_qc",
            "value": math.nan,
            "tolerance": REFERENCE_QC_ABS_TOL_KM2,
            "pass": True,
            "note": "Reference not found; internal QC only.",
        })
        return pd.DataFrame(rows)

    merged = daily[["date", "canonical_area_km2", "lcc_area_km2"]].merge(
        reference,
        on="date",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(daily):
        raise ValueError(
            f"QC reference covers {len(merged)} of {len(daily)} requested days."
        )

    dcan = np.abs(merged["canonical_area_km2"] - merged["total_area_km2"])
    dlcc = np.abs(merged["lcc_area_km2"] - merged["largest_component_area_km2"])

    can_tol = REFERENCE_QC_ABS_TOL_KM2 + QC_REL_TOL * np.maximum(
        np.abs(merged["canonical_area_km2"]),
        np.abs(merged["total_area_km2"]),
    )
    lcc_tol = REFERENCE_QC_ABS_TOL_KM2 + QC_REL_TOL * np.maximum(
        np.abs(merged["lcc_area_km2"]),
        np.abs(merged["largest_component_area_km2"]),
    )

    rows.extend([
        {
            "test": "canonical_area_vs_existing_authority",
            "value": float(dcan.max()),
            "tolerance": float(can_tol.max()),
            "pass": bool(np.all(dcan <= can_tol)),
        },
        {
            "test": "lcc_area_vs_existing_authority",
            "value": float(dlcc.max()),
            "tolerance": float(lcc_tol.max()),
            "pass": bool(np.all(dlcc <= lcc_tol)),
        },
    ])
    return pd.DataFrame(rows)


DEFINITIONS = {
    "canonical": "Pacific-mask canonical SST >= 28C",
    "lcc": "Largest 8-neighbor component of Pacific-mask SST >= 28C",
    "yan_gt28": "Yan-style fixed box SST > 28C; 20S-20N, 120E-150W",
    "yan_ge28_sensitivity": "Yan-style fixed box SST >= 28C sensitivity",
}


def annual_summary(daily: pd.DataFrame) -> pd.DataFrame:
    f = daily[
        (daily["year"] >= COMPLETE_YEAR_START)
        & (daily["year"] <= COMPLETE_YEAR_END)
    ].copy()
    rows: list[dict[str, object]] = []

    for year, gy in f.groupby("year"):
        expected_n = 366 if pd.Timestamp(f"{year}-12-31").is_leap_year else 365
        if len(gy) != expected_n:
            raise ValueError(f"Incomplete year {year}: {len(gy)} days.")

        for key, description in DEFINITIONS.items():
            area_col = f"{key}_area_km2"
            mean_col = f"{key}_mean_sst_c"
            integral_col = f"{key}_sst_area_integral_c_km2"

            # The >=28 sensitivity does not have a centroid by design.
            row = {
                "year": int(year),
                "definition": key,
                "definition_description": description,
                "n_days": int(len(gy)),
                "mean_daily_area_km2": float(gy[area_col].mean()),
                "median_daily_area_km2": float(gy[area_col].median()),
                "mean_daily_mean_sst_c": float(gy[mean_col].mean()),
                "area_time_weighted_mean_sst_c": float(
                    gy[integral_col].sum() / gy[area_col].sum()
                ),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def period_summary(
    daily: pd.DataFrame,
    start_year: int,
    end_year: int,
    label: str,
) -> list[dict[str, object]]:
    f = daily[(daily["year"] >= start_year) & (daily["year"] <= end_year)].copy()
    rows: list[dict[str, object]] = []

    for key, description in DEFINITIONS.items():
        area_col = f"{key}_area_km2"
        mean_col = f"{key}_mean_sst_c"
        integral_col = f"{key}_sst_area_integral_c_km2"
        rows.append({
            "period": label,
            "start_year": start_year,
            "end_year": end_year,
            "definition": key,
            "definition_description": description,
            "n_days": int(len(f)),
            "mean_daily_area_km2": float(f[area_col].mean()),
            "median_daily_area_km2": float(f[area_col].median()),
            "mean_daily_mean_sst_c": float(f[mean_col].mean()),
            "area_time_weighted_mean_sst_c": float(
                f[integral_col].sum() / f[area_col].sum()
            ),
        })

    # Direct method-comparison row.
    yan = f["yan_gt28_area_km2"]
    lcc = f["lcc_area_km2"]
    rows.append({
        "period": label,
        "start_year": start_year,
        "end_year": end_year,
        "definition": "LCC_MINUS_YAN_GT28",
        "definition_description": "Direct daily comparison: LCC minus historical Yan-style fixed-domain definition",
        "n_days": int(len(f)),
        "mean_daily_area_km2": float((lcc - yan).mean()),
        "median_daily_area_km2": float((lcc - yan).median()),
        "mean_daily_mean_sst_c": float(
            f["lcc_mean_sst_c"].mean() - f["yan_gt28_mean_sst_c"].mean()
        ),
        "area_time_weighted_mean_sst_c": math.nan,
        "mean_centroid_distance_km": float(
            f["lcc_yan_gt28_centroid_distance_km"].mean()
        ),
        "mean_lcc_area_outside_yan_box_fraction": float(
            f["lcc_area_outside_yan_box_fraction"].mean()
        ),
        "mean_yan_area_outside_lcc_fraction": float(
            f["yan_gt28_area_outside_lcc_fraction"].mean()
        ),
        "mean_area_jaccard": float(
            f["lcc_yan_gt28_area_jaccard"].mean()
        ),
        "mean_lcc_minus_yan_area_pct_of_yan": float(
            100.0 * (lcc - yan).mean() / yan.mean()
        ),
        "mean_operator_sensitivity_area_km2": float(
            f["yan_operator_sensitivity_area_km2"].mean()
        ),
        "mean_operator_sensitivity_sst_c": float(
            f["yan_operator_sensitivity_mean_sst_c"].mean()
        ),
    })
    return rows


def trend_summary(annual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ("mean_daily_area_km2", "area_time_weighted_mean_sst_c")
    for definition, g in annual.groupby("definition"):
        x = g["year"].to_numpy(float)
        for metric in metrics:
            y = g[metric].to_numpy(float)
            slope, intercept, low, high = theilslopes(y, x, alpha=0.95)
            rows.append({
                "definition": definition,
                "metric": metric,
                "n_years": len(g),
                "theil_sen_slope_per_decade": float(10.0 * slope),
                "ci95_low_per_decade": float(10.0 * low),
                "ci95_high_per_decade": float(10.0 * high),
                "intercept": float(intercept),
            })
    return pd.DataFrame(rows)


def write_report(
    paths: dict[str, Path],
    daily: pd.DataFrame,
    annual: pd.DataFrame,
    periods: pd.DataFrame,
    trends: pd.DataFrame,
    qc: pd.DataFrame,
) -> None:
    hist = periods[periods["period"] == "Yan historical interval 1982-1991"]
    full = periods[periods["period"] == "Complete-year climate interval 1982-2025"]
    comparison_hist = hist[hist["definition"] == "LCC_MINUS_YAN_GT28"]
    comparison_full = full[full["definition"] == "LCC_MINUS_YAN_GT28"]

    lines = [
        PROGRAM_NAME,
        f"Version: {PROGRAM_VERSION}",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "HISTORICAL DEFINITION",
        "Yan et al. (1993) explicitly state that the Yan et al. (1992) WPWP was",
        "defined as SST higher than 28C inside 120E-150W and 20S-20N.",
        "Primary reconstruction therefore uses SST > 28C in that fixed box.",
        "An >=28C operator sensitivity is also reported.",
        "",
        "PRESENT-DAY TOPOLOGICAL DEFINITION",
        "PWP-LCC = largest 8-neighbor connected component of SST >= 28C",
        "inside the validated Pacific mask, ranked by exact spherical area.",
        "",
        "RECORD",
        f"{daily['date'].min().date()} to {daily['date'].max().date()} | N={len(daily):,}",
        f"Complete-year inference: {COMPLETE_YEAR_START}-{COMPLETE_YEAR_END}",
        f"Historical Yan comparison: {YAN_PERIOD_START}-{YAN_PERIOD_END}",
        "",
        "QC",
        qc.to_string(index=False),
        "",
        "PERIOD SUMMARY",
        periods.to_string(index=False),
        "",
        "TREND SUMMARY",
        trends.to_string(index=False),
        "",
        "SCIENTIFIC INTERPRETATION RULE",
        "A fixed geographical window is a valid historical/regional framework,",
        "but threshold-exceeding water within that window need not be connected",
        "to the principal western Pacific warm body. The LCC provides an explicit",
        "topological criterion. Results should be described as a methodological",
        "extension/complement, not as evidence that earlier work was invalid.",
    ]

    if not comparison_hist.empty:
        r = comparison_hist.iloc[0]
        lines += [
            "",
            "DIRECT LCC VS YAN-STYLE COMPARISON, 1982-1991",
            f"Mean LCC-minus-Yan area: {r['mean_daily_area_km2']/1e6:.6f} x10^6 km2",
            f"Mean LCC-minus-Yan SST: {r['mean_daily_mean_sst_c']:.6f} degC",
            f"Mean centroid separation: {r.get('mean_centroid_distance_km', math.nan):.3f} km",
            f"Mean LCC fraction outside Yan box: {100*r.get('mean_lcc_area_outside_yan_box_fraction', math.nan):.3f}%",
            f"Mean Yan warm-area fraction outside LCC: {100*r.get('mean_yan_area_outside_lcc_fraction', math.nan):.3f}%",
            f"Mean area Jaccard: {r.get('mean_area_jaccard', math.nan):.6f}",
        ]
    if not comparison_full.empty:
        r = comparison_full.iloc[0]
        lines += [
            "",
            "DIRECT LCC VS YAN-STYLE COMPARISON, 1982-2025",
            f"Mean LCC-minus-Yan area: {r['mean_daily_area_km2']/1e6:.6f} x10^6 km2",
            f"Mean LCC-minus-Yan SST: {r['mean_daily_mean_sst_c']:.6f} degC",
            f"Mean centroid separation: {r.get('mean_centroid_distance_km', math.nan):.3f} km",
            f"Mean LCC fraction outside Yan box: {100*r.get('mean_lcc_area_outside_yan_box_fraction', math.nan):.3f}%",
            f"Mean Yan warm-area fraction outside LCC: {100*r.get('mean_yan_area_outside_lcc_fraction', math.nan):.3f}%",
            f"Mean area Jaccard: {r.get('mean_area_jaccard', math.nan):.6f}",
        ]

    paths["report"].write_text("\n".join(lines), encoding="utf-8")


def write_metadata_and_manifest(
    paths: dict[str, Path],
    start: pd.Timestamp,
    end: pd.Timestamp,
    files: list[Path],
    outputs: list[Path],
    qc_reference: Path | None,
) -> None:
    metadata = {
        "metadata_schema": "JCLI-PWP-LCC-historical-comparison/1.2",
        "program": PROGRAM_NAME,
        "version": PROGRAM_VERSION,
        "program_date": PROGRAM_DATE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "program_file": str(Path(__file__).resolve()),
        "program_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "record": {
            "start": str(start.date()),
            "end": str(end.date()),
            "n_days": int((end - start).days + 1),
        },
        "historical_definition": {
            "source": "Yan et al. (1993) response describing Yan et al. (1992)",
            "threshold_operator_primary": ">",
            "threshold_c": THRESHOLD_C,
            "latitude_min": YAN_LAT_MIN,
            "latitude_max": YAN_LAT_MAX,
            "longitude_min_e": YAN_LON_MIN_E,
            "longitude_max_e": YAN_LON_MAX_E,
            "longitude_max_original": "150W",
            "sensitivity_threshold_operator": ">=",
        },
        "lcc_definition": {
            "threshold_operator": ">=",
            "threshold_c": THRESHOLD_C,
            "connectivity": "8-neighbor with periodic longitude seam union",
            "component_ranking": "exact spherical area",
            "pacific_mask": str(paths["mask"]),
        },
        "methods": {
            "earth_radius_km": EARTH_RADIUS_KM,
            "reference_qc_note": "JCLI01 v1.1.0 daily CSV stored with %.10g; external area-QC absolute tolerance = 0.01 km2 for serialization rounding only",
            "mean_sst": "exact spherical-area-weighted over the daily definition mask",
            "annual_mean_daily_sst": "arithmetic mean of daily area-weighted mean SST",
            "annual_area_time_weighted_sst": "sum(SST*cell_area over all mask-days) / sum(cell_area over all mask-days)",
            "interpolation": False,
            "smoothing": False,
            "morphology": False,
        },
        "inputs": {
            "mask": str(paths["mask"]),
            "grid_lat": str(paths["grid_lat"]),
            "grid_lon": str(paths["grid_lon"]),
            "annual_oisst_files": [str(p) for p in files],
            "independent_qc_reference": str(qc_reference) if qc_reference else None,
        },
        "outputs": [str(p) for p in outputs],
    }
    save_json(metadata, paths["metadata"])

    manifest_targets = [p for p in outputs + [paths["metadata"]] if p.is_file()]
    manifest = pd.DataFrame([
        {
            "relative_path": str(p.relative_to(paths["root"])).replace("\\", "/"),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        }
        for p in manifest_targets
    ])
    atomic_csv(manifest, paths["manifest"])


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    paths = resolve_project_paths(args.project_root)
    ensure_dirs(paths)

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if start > end:
        raise ValueError("start-date must not exceed end-date")
    if start < EXPECTED_START or end > EXPECTED_END:
        raise ValueError(
            f"Requested dates must lie within frozen OISST record "
            f"{EXPECTED_START.date()} to {EXPECTED_END.date()}."
        )

    lat, lon, pacific, area = load_reference_grid(paths)
    basis = build_spherical_basis(lat, lon)
    yan_box = yan_geographic_box(lat, lon)

    box_area_total = float(area[yan_box].sum())
    logging.info(
        "Yan geographic box: %.1fS-%.1fN, %.1fE-%.1fE (150W); grid cells=%d; spherical grid area=%.3f x10^6 km2",
        abs(YAN_LAT_MIN), YAN_LAT_MAX, YAN_LON_MIN_E, YAN_LON_MAX_E,
        int(yan_box.sum()), box_area_total / 1e6,
    )

    files = discover_oisst_files(paths, start, end)

    yearly_frames: list[pd.DataFrame] = []
    for nc in files:
        year = int(nc.stem.split(".")[-1])
        year_start = max(start, pd.Timestamp(f"{year}-01-01"))
        year_end = min(end, pd.Timestamp(f"{year}-12-31"))
        checkpoint = paths["cache_dir"] / f"year_{year}.csv"

        if checkpoint.is_file() and not args.rebuild:
            cached = pd.read_csv(checkpoint, parse_dates=["date"], low_memory=False)
            exp = pd.date_range(year_start, year_end, freq="D")
            if len(cached) == len(exp) and pd.DatetimeIndex(cached["date"]).equals(exp):
                logging.info("Checkpoint PASS; reusing %s", checkpoint.name)
                yearly_frames.append(cached)
                continue
            logging.warning("Checkpoint invalid; rebuilding %s", checkpoint.name)

        logging.info("Starting year %d (%s to %s)", year, year_start.date(), year_end.date())
        frame = process_year(
            nc, lat, lon, pacific, area, basis, yan_box, year_start, year_end
        )
        frame["date"] = pd.to_datetime(frame["date"])
        atomic_csv(frame, checkpoint, float_format="%.12g")
        yearly_frames.append(frame)
        logging.info("Checkpoint written: %s", checkpoint)

    daily = pd.concat(yearly_frames, ignore_index=True)
    daily = validate_daily(daily, start, end)
    atomic_csv(daily, paths["daily"], float_format="%.12g")

    reference = None if args.no_qc_reference else load_qc_reference(paths)
    qc = qc_against_reference(daily, reference)
    atomic_csv(qc, paths["qc"], float_format="%.12g")

    if not bool(qc["pass"].all()):
        raise RuntimeError(
            "One or more scientific QC tests failed:\n"
            + qc.to_string(index=False)
        )

    annual = annual_summary(daily)
    atomic_csv(annual, paths["annual"], float_format="%.12g")

    period_rows: list[dict[str, object]] = []
    if start.year <= YAN_PERIOD_START and end.year >= YAN_PERIOD_END:
        period_rows.extend(
            period_summary(
                daily,
                YAN_PERIOD_START,
                YAN_PERIOD_END,
                "Yan historical interval 1982-1991",
            )
        )
    if start.year <= COMPLETE_YEAR_START and end.year >= COMPLETE_YEAR_END:
        period_rows.extend(
            period_summary(
                daily,
                COMPLETE_YEAR_START,
                COMPLETE_YEAR_END,
                "Complete-year climate interval 1982-2025",
            )
        )
    periods = pd.DataFrame(period_rows)
    atomic_csv(periods, paths["period"], float_format="%.12g")

    trends = trend_summary(annual)
    atomic_csv(trends, paths["trend"], float_format="%.12g")

    write_report(paths, daily, annual, periods, trends, qc)

    qc_reference_path = None
    if reference is not None:
        if paths["jcli11_daily"].is_file():
            qc_reference_path = paths["jcli11_daily"]
        elif paths["program31_daily"].is_file():
            qc_reference_path = paths["program31_daily"]

    outputs = [
        paths["daily"],
        paths["annual"],
        paths["period"],
        paths["trend"],
        paths["qc"],
        paths["report"],
    ]
    write_metadata_and_manifest(
        paths, start, end, files, outputs, qc_reference_path
    )

    logging.info("=" * 78)
    logging.info("COMPLETED %s v%s", PROGRAM_NAME, PROGRAM_VERSION)
    logging.info("All QC tests PASS.")
    logging.info("Daily table: %s", paths["daily"])
    logging.info("Annual table: %s", paths["annual"])
    logging.info("Period summary: %s", paths["period"])
    logging.info("Trend summary: %s", paths["trend"])
    logging.info("QC summary: %s", paths["qc"])
    logging.info("Report: %s", paths["report"])
    logging.info("Metadata: %s", paths["metadata"])
    logging.info("Manifest: %s", paths["manifest"])
    logging.info("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Fatal error")
        raise
