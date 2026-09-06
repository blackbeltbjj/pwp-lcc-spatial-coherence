#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JCLI02_figures_pwp_lcc_publication_v1_1_0.py
============================================================
Fast, figure-only workflow for the Journal of Climate PWP-LCC manuscript.

This program NEVER opens the NOAA OISST NetCDF files and NEVER recalculates
connected components, trends, STL, Welch spectra, or wavelets.  It reads the
audited products written by JCLI01_pwp_lcc_analysis_v1_1_0.py and produces
camera-ready Figures 1--9 in grayscale (PNG, PDF, and TIFF).

Place JCLI01 and JCLI02 together in PROJECT_ROOT/src, then run:

    python src/JCLI02_figures_pwp_lcc_publication_v1_1_0.py \
        --project-root PROJECT_ROOT

Adjusting layout, labels, line widths, or captions requires rerunning only
this program.  If numerical methods, thresholds, mask, or input dates change,
rerun JCLI01 first.

Author: Fabio Vieira Machado
Version: 1.1.0 (2026-09-05; disconnected-occurrence Figure 2)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import JCLI01_pwp_lcc_analysis_v1_1_0 as core


PROGRAM_NAME = "JCLI02 PWP-LCC PUBLICATION FIGURES"
PROGRAM_VERSION = "1.1.0"
REQUIRED_JCLI01_VERSION = "1.1.0"
REQUIRED_JCLI01_BASENAME = "JCLI01_pwp_lcc_analysis_v1_1_0.py"


def format_paper_time_axis(ax: plt.Axes, step_years: int = 4) -> None:
    """Use a fixed 1982-origin manuscript tick convention."""
    ax.set_xlim(core.EXPECTED_START, core.EXPECTED_END)
    years = np.arange(core.YEAR_TICK_START, 2027, step_years)
    ticks = pd.to_datetime([f"{year}-01-01" for year in years])
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.tick_params(axis="x", rotation=0)


def format_biennial_time_axes(axes: Iterable[plt.Axes]) -> None:
    """Apply explicit even-year ticks to every shared time-series axis.

    Matplotlib can silently replace locators on shared axes when only the last
    panel is formatted.  Setting the same fixed locator and formatter on every
    panel makes the two-year convention deterministic.  Labels remain visible
    only on the bottom panel, while tick marks are retained on all panels.
    """
    axes = list(axes)
    years = np.arange(core.YEAR_TICK_START, 2027, 2)
    ticks = pd.to_datetime([f"{year}-01-01" for year in years])
    for index, ax in enumerate(axes):
        ax.set_xlim(core.EXPECTED_START, core.EXPECTED_END)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
        ax.tick_params(axis="x", which="major", bottom=True,
                       labelbottom=(index == len(axes) - 1), rotation=0)


def configure_pacific_map(ax: plt.Axes, pacific: np.ndarray,
                          lat: np.ndarray, lon: np.ndarray,
                          left_labels: bool = True,
                          bottom_labels: bool = True) -> None:
    """Apply the exact analysis-mask footprint and a dateline-safe Pacific view."""
    active_rows, active_cols = np.where(pacific)
    if active_rows.size == 0:
        raise ValueError("The validated Pacific mask contains no active cells.")
    lon_min = max(100.0, float(np.nanmin(lon[active_cols])))
    lon_max = min(290.0, float(np.nanmax(lon[active_cols])))
    lat_min = max(-35.0, float(np.nanmin(lat[active_rows])))
    lat_max = min(35.0, float(np.nanmax(lat[active_rows])))
    # Input coordinates are geographical 0--360 degrees east.  Cartopy
    # performs the transformation to the Pacific-centred plotting CRS.
    ax.set_extent((lon_min, lon_max, lat_min, lat_max),
                  crs=ccrs.PlateCarree())
    # Make cells outside pacific_mask_oisst.npy visually distinct from valid
    # Pacific cells with zero occurrence (which are white in the Greys map).
    ax.set_facecolor("0.90")
    ax.add_feature(cfeature.LAND, facecolor="0.78", edgecolor="0.2",
                   linewidth=0.35, zorder=20)
    ax.coastlines(resolution="110m", linewidth=0.45, zorder=21)
    longitude_ticks = np.arange(100.0, 281.0, 20.0)
    latitude_ticks = np.arange(-30.0, 31.0, 15.0)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                 xlocs=longitude_ticks, ylocs=latitude_ticks,
                 linewidth=0.35, color="0.65", linestyle=":")

    # GeoAxes stores x tick locations in its rotated PlateCarree coordinates.
    # Transform the requested east longitudes explicitly, then assign literal
    # 0--360°E labels.  This avoids Cartopy converting values >180° to °W.
    projected_x = ax.projection.transform_points(
        ccrs.PlateCarree(), longitude_ticks,
        np.zeros_like(longitude_ticks),
    )[:, 0]
    ax.set_xticks(projected_x)
    ax.set_xticklabels([f"{value:.0f}°E" for value in longitude_ticks])
    ax.set_yticks(latitude_ticks)
    ax.set_yticklabels([
        "0°" if value == 0 else f"{abs(value):.0f}°{'N' if value > 0 else 'S'}"
        for value in latitude_ticks
    ])
    ax.tick_params(axis="x", bottom=bottom_labels, labelbottom=bottom_labels,
                   top=False, labeltop=False, labelsize=8)
    ax.tick_params(axis="y", left=left_labels, labelleft=left_labels,
                   right=False, labelright=False, labelsize=8)


def draw_pacific_mask_boundary(ax: plt.Axes, pacific: np.ndarray,
                               lat: np.ndarray, lon: np.ndarray) -> None:
    """Draw the exact boundary of pacific_mask_oisst.npy."""
    lon2, lat2 = np.meshgrid(lon, lat)
    ax.contour(
        lon2, lat2, pacific.astype(float), levels=[0.5], colors="0.35",
        linewidths=0.45, linestyles="-", transform=ccrs.PlateCarree(),
        zorder=19,
    )


def masked_pacific_field(field: np.ndarray, pacific: np.ndarray,
                         field_name: str) -> np.ma.MaskedArray:
    """Return a field restricted strictly to the validated Pacific mask."""
    values = np.asarray(field, dtype=float)
    if values.shape != pacific.shape:
        raise ValueError(
            f"{field_name} shape {values.shape} differs from Pacific mask {pacific.shape}."
        )
    return np.ma.array(values, mask=(~pacific) | (~np.isfinite(values)), copy=False)


def load_authoritative_pacific_mask(paths: core.Paths,
                                    expected_lat: np.ndarray,
                                    expected_lon: np.ndarray) -> np.ndarray:
    """Load data/processed/pacific_mask_oisst.npy as the sole map mask."""
    lat, lon, pacific, _ = core.load_reference_grid(paths)
    if not np.array_equal(lat, np.asarray(expected_lat, dtype=float)):
        raise ValueError(
            "Latitude in pacific_mask_oisst.npy does not match the JCLI01 spatial cache."
        )
    if not np.array_equal(lon, np.asarray(expected_lon, dtype=float)):
        raise ValueError(
            "Longitude in pacific_mask_oisst.npy does not match the JCLI01 spatial cache."
        )
    return pacific


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_root = here.parents[1] if here.parent.name == "src" else Path.cwd()
    parser = argparse.ArgumentParser(description=PROGRAM_NAME)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def read_csv(path: Path, date_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JCLI01 product not found: {path}\n"
            "Run JCLI01_pwp_lcc_analysis_v1_1_0.py before JCLI02."
        )
    frame = pd.read_csv(path, low_memory=False)
    for column in date_columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def load_product_dictionary(path: Path) -> dict[float, dict[str, np.ndarray | float]]:
    """Reverse JCLI01's object-free threshold-keyed NPZ serialization."""
    if not path.is_file():
        raise FileNotFoundError(f"Required JCLI01 binary product not found: {path}")
    products: dict[float, dict[str, np.ndarray | float]] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            prefix, variable = key.split("__", 1)
            threshold = float(prefix[1:].replace("p", "."))
            value = archive[key]
            products.setdefault(threshold, {})[variable] = value.item() if value.ndim == 0 else value
    missing = set(core.THRESHOLDS_C).difference(products)
    if missing:
        raise ValueError(f"NPZ {path} lacks thresholds: {sorted(missing)}")
    return products


def validate_analysis_metadata(paths: core.Paths) -> dict:
    if not paths.metadata_json.is_file():
        raise FileNotFoundError(f"JCLI01 metadata not found: {paths.metadata_json}")
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    if metadata.get("version") != REQUIRED_JCLI01_VERSION:
        raise RuntimeError(
            f"JCLI01 version {metadata.get('version')!r} is incompatible with "
            f"JCLI02; expected {REQUIRED_JCLI01_VERSION}."
        )
    if metadata.get("program_basename") != REQUIRED_JCLI01_BASENAME:
        raise RuntimeError(
            f"JCLI01 metadata identifies {metadata.get('program_basename')!r}; "
            f"expected {REQUIRED_JCLI01_BASENAME!r}. Run JCLI01 with "
            "--refresh-provenance-only before JCLI02."
        )
    if tuple(float(v) for v in metadata.get("thresholds_c", [])) != core.THRESHOLDS_C:
        raise RuntimeError("Thresholds recorded by JCLI01 do not match JCLI02.")
    return metadata


def figure1_occurrence_climatology(paths: core.Paths) -> list[Path]:
    """Plot canonical and LCC mean-occurrence maps as two stacked panels.

    The grayscale field is the period-mean occurrence of the conventional
    28 degC definition.  Thin contours delimit locations satisfying each
    threshold definition on at least 50% of valid days.  Using the same
    occurrence contour for all thresholds makes their nested spatial
    contraction directly comparable without mixing three filled fields.
    """
    source = paths.occurrence_npz
    if not source.is_file():
        raise FileNotFoundError(
            f"Figure 1 source is missing: {source}. Run JCLI01 with --rebuild-spatial."
        )
    with np.load(source, allow_pickle=False) as archive:
        lat = archive["lat"]
        lon = archive["lon"]
        cached_pacific = archive["pacific_mask"].astype(bool)
        denominator = archive["valid_day_count"].astype(float)
        available = set(archive.files)
        fields: dict[tuple[float, str], np.ndarray] = {}
        for threshold in core.THRESHOLDS_C:
            code = str(threshold).replace(".", "p")
            for definition, prefix in (("canonical", "canonical"), ("lcc", "lcc")):
                key = f"{prefix}_occurrence_count_{code}"
                if key not in available:
                    raise RuntimeError(
                        f"Missing {key}. Rerun JCLI01 with --rebuild-spatial."
                    )
                count = archive[key].astype(float)
                fields[(threshold, definition)] = np.divide(
                    100.0 * count, denominator, out=np.full_like(count, np.nan),
                    where=denominator > 0,
                )

    pacific = load_authoritative_pacific_mask(paths, lat, lon)
    if not np.array_equal(pacific, cached_pacific):
        raise ValueError(
            "The cached mask differs from data/processed/pacific_mask_oisst.npy; "
            "refusing to plot an inconsistent spatial product."
        )
    lon2, lat2 = np.meshgrid(lon, lat)
    # Cartopy changes the final GeoAxes width to preserve map aspect.  The
    # colorbar is therefore positioned only after the canvas has been drawn,
    # using the exact final left edge and width of the map panels.
    fig = plt.figure(figsize=(11.5, 8.0))
    grid = fig.add_gridspec(
        2, 1, height_ratios=(1.0, 1.0),
        left=0.075, right=0.985, top=0.925, bottom=0.165,
        hspace=0.22,
    )
    axes = np.asarray([
        fig.add_subplot(grid[0, 0], projection=ccrs.PlateCarree(central_longitude=180)),
        fig.add_subplot(grid[1, 0], projection=ccrs.PlateCarree(central_longitude=180)),
    ])
    levels = np.arange(0, 101, 10)
    image = None
    line_styles = {
        28.0: dict(color="black", linestyle="-", linewidth=0.65),
        28.5: dict(color="black", linestyle="--", linewidth=0.65),
        29.0: dict(color="white", linestyle="-", linewidth=0.65),
    }
    for row, definition in enumerate(("canonical", "lcc")):
        ax = axes[row]
        configure_pacific_map(
            ax, pacific, lat, lon, left_labels=True, bottom_labels=(row == 1)
        )
        draw_pacific_mask_boundary(ax, pacific, lat, lon)

        background = masked_pacific_field(
            fields[(28.0, definition)], pacific,
            f"Figure 1 {definition} occurrence 28C",
        )
        image = ax.contourf(
            lon2, lat2, background, levels=levels, cmap="Greys",
            vmin=0, vmax=100, transform=ccrs.PlateCarree(), extend="neither",
        )

        legend_handles = []
        for threshold in core.THRESHOLDS_C:
            occurrence = masked_pacific_field(
                fields[(threshold, definition)], pacific,
                f"Figure 1 {definition} occurrence {threshold:g}C",
            )
            style = line_styles[float(threshold)]
            ax.contour(
                lon2, lat2, occurrence, levels=[50.0],
                colors=[style["color"]], linestyles=[style["linestyle"]],
                linewidths=[style["linewidth"]],
                transform=ccrs.PlateCarree(), zorder=18,
            )
            legend_handles.append(mpl.lines.Line2D(
                [], [], label=f"PWP {threshold:g}°C (50% occurrence)",
                **style,
            ))

        ax.set_title("A) Canonical PWP" if row == 0 else "B) PWP LCC")
        # Repeat the threshold key in both maps at the lower-right Pacific
        # margin, adjacent to South America.  The white 29 degC key receives
        # a fine grey outline only in the legend so it remains visible.
        legend_handles[-1].set_path_effects([
            path_effects.Stroke(linewidth=1.35, foreground="0.45"),
            path_effects.Normal(),
        ])
        ax.legend(
            handles=legend_handles, loc="lower right",
            bbox_to_anchor=(0.985, 0.025), ncol=1,
            frameon=False, fontsize=8.0, handlelength=3.4,
            borderaxespad=0.0, labelspacing=0.35,
        )
    # Force Cartopy to finalize the axes geometry, then make the colorbar
    # exactly as wide as panel B (and panel A, which shares the same extent).
    fig.canvas.draw()
    map_position = axes[1].get_position()
    colorbar_axis = fig.add_axes([
        map_position.x0, 0.060, map_position.width, 0.024
    ])
    cbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    cbar.set_label("Occurrence of the PWP ≥28°C (% of valid days)")
    fig.suptitle(
        "Mean occurrence of the canonical and largest-connected Pacific Warm Pool",
        y=0.985,
    )
    return core.save_figure(
        fig, paths, "Figure_01_Canonical_vs_LCC_mean_occurrence"
    )


def figure2_disconnected_occurrence(paths: core.Paths,
                                    arrays: dict[str, np.ndarray]) -> list[Path]:
    """Map canonical warm occurrence that is disconnected from the daily LCC."""
    lat = np.asarray(arrays["lat"], dtype=float)
    lon = np.asarray(arrays["lon"], dtype=float)
    cached_pacific = arrays["pacific_mask"].astype(bool)
    pacific = load_authoritative_pacific_mask(paths, lat, lon)
    if not np.array_equal(pacific, cached_pacific):
        raise ValueError(
            "The occurrence cache and pacific_mask_oisst.npy are inconsistent."
        )
    denominator = arrays["valid_day_count"].astype(float)
    lon2, lat2 = np.meshgrid(lon, lat)
    fig = plt.figure(figsize=(11.5, 10.0))
    grid = fig.add_gridspec(
        3, 1, left=0.075, right=0.985, top=0.925, bottom=0.135,
        hspace=0.16,
    )
    axes: list[plt.Axes] = []
    image = None
    levels = np.arange(0.0, 55.0, 5.0)
    for row, threshold in enumerate(core.THRESHOLDS_C):
        code = str(threshold).replace(".", "p")
        canonical = arrays[f"canonical_occurrence_count_{code}"].astype(float)
        lcc = arrays[f"lcc_occurrence_count_{code}"].astype(float)
        detached_count = canonical - lcc
        if np.nanmin(detached_count[pacific]) < 0:
            raise ValueError(f"LCC count exceeds canonical count at {threshold:g}C.")
        detached_pct = np.divide(
            100.0 * detached_count, denominator,
            out=np.full_like(detached_count, np.nan),
            where=(denominator > 0) & pacific,
        )
        lcc_pct = np.divide(
            100.0 * lcc, denominator, out=np.full_like(lcc, np.nan),
            where=(denominator > 0) & pacific,
        )
        ax = fig.add_subplot(
            grid[row, 0], projection=ccrs.PlateCarree(central_longitude=180)
        )
        axes.append(ax)
        configure_pacific_map(
            ax, pacific, lat, lon, left_labels=True, bottom_labels=(row == 2)
        )
        draw_pacific_mask_boundary(ax, pacific, lat, lon)
        image = ax.contourf(
            lon2, lat2,
            masked_pacific_field(detached_pct, pacific, "detached occurrence"),
            levels=levels, cmap="Greys", vmin=0.0, vmax=50.0,
            extend="max", transform=ccrs.PlateCarree(),
        )
        # The fine white contour locates the persistent core of the daily LCC.
        ax.contour(
            lon2, lat2,
            masked_pacific_field(lcc_pct, pacific, "LCC occurrence"),
            levels=[50.0], colors="white", linewidths=0.65,
            transform=ccrs.PlateCarree(), zorder=18,
        )
        ax.set_title(
            f"{chr(65 + row)}) SST ≥ {threshold:g}°C: canonical occurrence outside the LCC"
        )
    fig.canvas.draw()
    position = axes[-1].get_position()
    colorbar_axis = fig.add_axes([position.x0, 0.055, position.width, 0.022])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(
        "Detached warm-pool occurrence (% of valid days; white contour = 50% LCC occurrence)"
    )
    fig.suptitle(
        "Spatial occurrence of warm water disconnected from the principal Pacific Warm Pool",
        y=0.985,
    )
    return core.save_figure(
        fig, paths, "Figure_02_Disconnected_warm_pool_occurrence"
    )


def figure2(paths: core.Paths, daily: pd.DataFrame, annual: pd.DataFrame,
            trends: pd.DataFrame,
            common_y_scale: bool = False) -> list[Path]:
    """Create Figure 3 and its common-y-scale comparison Figure 03A."""
    fig, axes = plt.subplots(3, 1, figsize=(14.5, 10.2), sharex=True)
    for index, (threshold, ax) in enumerate(zip(core.THRESHOLDS_C, axes)):
        series = daily[np.isclose(daily.threshold_c, threshold)].sort_values("date")
        yearly = annual[np.isclose(annual.threshold_c, threshold)]
        rolling = (series.set_index("date").largest_component_area_km2
                   .rolling(365, center=True, min_periods=183).mean() / 1e6)
        ax.plot(series.date, series.largest_component_area_km2 / 1e6,
                color="0.82", lw=0.45, label="Daily LCC area")
        ax.plot(rolling.index, rolling, color="0.05", lw=1.55, label="365-day moving mean")
        ax.plot(yearly.date, yearly.largest_component_area_km2 / 1e6, "o",
                ms=3.8, mfc="white", mec="black", label="Complete-year mean")
        trend = trends[(np.isclose(trends.threshold_c, threshold)) &
                       (trends.metric == "largest_component_area_km2")].iloc[0]
        years = yearly.year.to_numpy(float)
        slope = trend.theil_sen_slope_per_decade / 10
        intercept = np.median(yearly.largest_component_area_km2.to_numpy() - slope * years)
        ax.plot(yearly.date, (intercept + slope * years) / 1e6, "k--", lw=1,
                label=f"Theil–Sen: {trend.theil_sen_slope_per_decade/1e6:.2f} "
                      "×10⁶ km² decade⁻¹")
        ax.set_ylabel(f"≥{threshold:g}°C LCC area\n(10⁶ km²)")
        if common_y_scale:
            ax.set_ylim(0.0, 60.0)
        core.panel_label(ax, f"{chr(65 + index)})")
        core.grid(ax)
        ax.legend(frameon=False, loc="lower right", ncol=4)
    format_biennial_time_axes(axes)
    axes[-1].set_xlabel("Year")
    if common_y_scale:
        fig.suptitle(
            "Daily evolution and long-term expansion of PWP-LCC area — common y-axis scale",
            y=0.995,
        )
        fig.text(
            0.99, 0.006,
            "Note: all panels use the same 0–60 × 10⁶ km² y-axis scale.",
            ha="right", va="bottom", fontsize=8,
        )
        output_name = "Figure_03A_LCC_area_time_series"
    else:
        fig.suptitle("Daily evolution and long-term expansion of PWP-LCC area", y=0.995)
        fig.text(
            0.99, 0.006,
            "Note: y-axis scales differ among panels to preserve within-threshold variability.",
            ha="right", va="bottom", fontsize=8,
        )
        output_name = "Figure_03_LCC_area_time_series"
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.985))
    return core.save_figure(fig, paths, output_name)


def figure3(paths: core.Paths, daily: pd.DataFrame) -> list[Path]:
    """RBMet-style four-panel connectivity and fragmentation time series."""
    fig, axes = plt.subplots(4, 1, figsize=(14.5, 11.8), sharex=True)
    specifications = [
        ("largest_component_area_fraction", "Fraction of canonical area\nin the LCC", 1.0),
        ("secondary_area_km2", "Area outside the LCC\n(10⁶ km²)", 1e6),
        ("component_count", "Number of connected\ncomponents", 1.0),
        ("fragmentation_index", "Fragmentation index", 1.0),
    ]
    for index, (ax, (metric, label, scale)) in enumerate(zip(axes, specifications)):
        for threshold in core.THRESHOLDS_C:
            series = daily[np.isclose(daily.threshold_c, threshold)].sort_values("date")
            rolling = (series.set_index("date")[metric]
                       .rolling(365, center=True, min_periods=183).mean() / scale)
            style = core.THRESHOLD_STYLES[threshold]
            ax.plot(rolling.index, rolling, color=style["color"],
                    linestyle=style["linestyle"], lw=1.35,
                    label=f"{threshold:g}°C")
        ax.set_ylabel(label)
        core.panel_label(ax, f"{chr(65 + index)})")
        core.grid(ax)
        ax.legend(frameon=False, loc="best", ncol=3)
    format_biennial_time_axes(axes)
    axes[-1].set_xlabel("Year")
    fig.suptitle("Threshold dependence of PWP connectivity and fragmentation", y=0.995)
    fig.tight_layout()
    return core.save_figure(fig, paths, "Figure_04_Connectivity_fragmentation")


def figure4(paths: core.Paths, annual: pd.DataFrame, trends: pd.DataFrame) -> list[Path]:
    specifications = [
        ("largest_component_area_km2", "LCC area (10⁶ km²)", 1e6),
        ("secondary_area_km2", "Secondary area (10⁶ km²)", 1e6),
        ("largest_component_area_fraction", "Fraction in LCC", 1),
        ("component_count", "Number of components", 1),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for index, (ax, (metric, label, scale)) in enumerate(zip(axes.ravel(), specifications)):
        for threshold in core.THRESHOLDS_C:
            yearly = annual[np.isclose(annual.threshold_c, threshold)]
            style = core.THRESHOLD_STYLES[threshold]
            ax.plot(yearly.date, yearly[metric] / scale, color=style["color"],
                    ls=style["linestyle"], marker=style["marker"], ms=2.8,
                    mfc="white", label=f"{threshold:g}°C")
            trend = trends[(np.isclose(trends.threshold_c, threshold)) &
                           (trends.metric == metric)].iloc[0]
            slope = trend.theil_sen_slope_per_decade / 10
            intercept = np.median(yearly[metric] - slope * yearly.year)
            ax.plot(yearly.date, (intercept + slope * yearly.year) / scale,
                    color=style["color"], ls="-", lw=0.8)
        ax.set_ylabel(label)
        core.panel_label(ax, f"{chr(65 + index)})")
        core.grid(ax)
        format_paper_time_axis(ax)
    axes[0, 0].legend(frameon=False, ncol=3)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    fig.suptitle("Annual expansion and spatial reorganization of the PWP-LCC", y=0.995)
    fig.tight_layout()
    return core.save_figure(fig, paths, "Figure_05_Annual_LCC_structural_trends")


def figure5(paths: core.Paths, climatology: pd.DataFrame,
            stl: pd.DataFrame) -> list[Path]:
    """Four uncluttered diagnostics; one metric and one y-axis per panel."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    climatology_ax, trend_ax, amplitude_ax, remainder_ax = axes.ravel()
    for threshold in core.THRESHOLDS_C:
        style = core.THRESHOLD_STYLES[threshold]
        monthly = climatology[np.isclose(climatology.threshold_c, threshold)]
        climatology_ax.plot(
            monthly.month, monthly.largest_component_area_km2_mean / 1e6,
            color=style["color"], ls=style["linestyle"],
            marker=style["marker"], mfc="white", label=f"{threshold:g}°C",
        )
        decomposition = stl[np.isclose(stl.threshold_c, threshold)].copy()
        trend_ax.plot(
            decomposition.date, decomposition.trend_km2 / 1e6,
            color=style["color"], ls=style["linestyle"], lw=1.25,
            label=f"{threshold:g}°C",
        )
        amplitude = (decomposition.assign(year=decomposition.date.dt.year)
                     .groupby("year").agg(
                         amplitude=("seasonal_km2", lambda values: values.max() - values.min()),
                         mean=("observed_km2", "mean"),
                         remainder_rms=("remainder_km2",
                                        lambda values: np.sqrt(np.nanmean(np.square(values))))
                     ).reset_index())
        amplitude = amplitude[(amplitude.year >= core.COMPLETE_YEAR_START) &
                              (amplitude.year <= core.COMPLETE_YEAR_END)]
        amplitude["date"] = pd.to_datetime(amplitude.year.astype(str) + "-07-01")
        amplitude_ax.plot(
            amplitude.date, 100 * amplitude.amplitude / amplitude["mean"],
            color=style["color"], ls=style["linestyle"], lw=1.05,
            label=f"{threshold:g}°C",
        )
        remainder_ax.plot(
            amplitude.date, amplitude.remainder_rms / 1e6,
            color=style["color"], ls=style["linestyle"], lw=1.05,
            label=f"{threshold:g}°C",
        )

    climatology_ax.set_xticks(range(1, 13))
    climatology_ax.set_xticklabels(("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
                                   rotation=0)
    climatology_ax.set_xlabel("Month")
    climatology_ax.set_ylabel("Climatological LCC area\n(10⁶ km²)")
    trend_ax.set_ylabel("Robust STL trend\n(10⁶ km²)")
    amplitude_ax.set_ylabel("Seasonal amplitude\n(% of annual mean)")
    remainder_ax.set_ylabel("Annual remainder RMS\n(10⁶ km²)")
    amplitude_ax.set_xlabel("Year")
    remainder_ax.set_xlabel("Year")
    for index, ax in enumerate(axes.ravel()):
        core.panel_label(ax, f"{chr(65 + index)})")
        core.grid(ax)
        ax.legend(frameon=False, ncol=3, loc="best", fontsize=8)
    for ax in (trend_ax, amplitude_ax, remainder_ax):
        format_paper_time_axis(ax)
    trend_ax.tick_params(labelbottom=False)
    fig.suptitle("Seasonality and low-frequency evolution of PWP-LCC area", y=0.995)
    fig.tight_layout()
    return core.save_figure(fig, paths, "Figure_06_LCC_climatology_STL")


def figure6(paths: core.Paths, arrays: dict[str, np.ndarray]) -> list[Path]:
    lat, lon = arrays["lat"], arrays["lon"]
    cached_pacific = arrays["pacific_mask"].astype(bool)
    pacific = load_authoritative_pacific_mask(paths, lat, lon)
    if not np.array_equal(pacific, cached_pacific):
        raise ValueError(
            "The cached mask differs from data/processed/pacific_mask_oisst.npy; "
            "refusing to plot Figure 7."
        )
    denominator = arrays["valid_day_count"].astype(float)
    lon2, lat2 = np.meshgrid(lon, lat)
    # A wider canvas increases the physical height of each undistorted
    # PlateCarree map.  The reduced hspace keeps the three threshold rows
    # visually associated without allowing titles or tick labels to collide.
    fig = plt.figure(figsize=(14.5, 10.2))
    grid = fig.add_gridspec(
        3, 2, left=0.060, right=0.945, bottom=0.070, top=0.945,
        hspace=0.12, wspace=0.19,
    )
    axes = np.empty((3, 2), dtype=object)
    colorbar_requests: list[tuple[plt.Axes, mpl.cm.ScalarMappable]] = []
    for row, threshold in enumerate(core.THRESHOLDS_C):
        code = str(threshold).replace(".", "p")
        count = arrays[f"lcc_occurrence_count_{code}"].astype(float)
        occurrence = np.divide(100 * count, denominator,
                               out=np.full_like(count, np.nan), where=denominator > 0)
        longest = arrays[f"lcc_longest_run_days_{code}"].astype(float)
        positive = longest[longest > 0]
        persistence_max = np.nanpercentile(positive, 99) if positive.size else 1.0
        panels = ((occurrence, "LCC occurrence (%)", 100),
                  (longest, "Longest continuous LCC membership (days)", persistence_max))
        for column, (field, title, maximum) in enumerate(panels):
            ax = fig.add_subplot(
                grid[row, column],
                projection=ccrs.PlateCarree(central_longitude=180),
            )
            axes[row, column] = ax
            configure_pacific_map(
                ax, pacific, lat, lon,
                left_labels=(column == 0), bottom_labels=(row == 2),
            )
            draw_pacific_mask_boundary(ax, pacific, lat, lon)
            masked = masked_pacific_field(field, pacific, f"Figure 7 {title} {threshold:g}C")
            image = ax.pcolormesh(lon2, lat2, masked, cmap="Greys",
                                  vmin=0, vmax=maximum, shading="auto",
                                  transform=ccrs.PlateCarree(), rasterized=True)
            ax.set_title(f"SST ≥ {threshold:g}°C — {title}")
            core.panel_label(ax, f"{chr(65 + row * 2 + column)})")
            colorbar_requests.append((ax, image))
    fig.suptitle("Spatial occurrence and persistence of the Pacific Warm Pool LCC", y=0.995)

    # Add the six colorbars only after Cartopy has finalized every map's
    # physical bounding box.  Each bar copies the exact y0 and height of its
    # associated map, so saving with bbox_inches='tight' cannot alter their
    # one-to-one vertical alignment.
    fig.canvas.draw()
    for ax, image in colorbar_requests:
        position = ax.get_position()
        colorbar_axis = fig.add_axes([
            position.x1 + 0.006, position.y0, 0.011, position.height
        ])
        colorbar = fig.colorbar(image, cax=colorbar_axis,
                                orientation="vertical")
        colorbar.ax.tick_params(labelsize=8)
        # Reapply the map height after Colorbar has initialized its axis;
        # this is deterministic across Matplotlib backends and avoids brittle
        # floating-point equality checks.
        colorbar_axis.set_position([
            position.x1 + 0.006, position.y0, 0.011, position.height
        ])
    return core.save_figure(fig, paths, "Figure_07_LCC_occurrence_persistence")


def figure7(paths: core.Paths,
            products: dict[float, dict[str, np.ndarray | float]]) -> list[Path]:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9.5), sharex=True)
    for index, (threshold, ax) in enumerate(zip(core.THRESHOLDS_C, axes)):
        product = products[threshold]
        period = np.asarray(product["period_days"]) / 365.25
        bands = list(core.SPECTRAL_BANDS_DAYS.items())
        shades = ("0.92", "0.96", "0.88")
        for (name, (low, high)), shade in zip(bands, shades):
            ax.axvspan(low / 365.25, high / 365.25, color=shade,
                       label=name.replace("_", " ").title() if index == 0 else None)
        ax.plot(period, np.asarray(product["psd"]), color="0.05", lw=1.35,
                label="Welch PSD")
        ax.plot(period, np.asarray(product["significance_95"]), color="0.45",
                ls="--", lw=1.0, label="95% AR(1)")
        ax.set_ylabel(f"≥{threshold:g}°C PSD\n(km⁴ day)")
        core.panel_label(ax, f"{chr(65 + index)})")
        core.grid(ax)
        ax.legend(frameon=False)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.25, 8)
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    axes[-1].set_xlabel("Period (years)")
    fig.suptitle("Welch spectra of PWP-LCC area", y=0.995)
    fig.text(
        0.99, 0.006,
        "Note: y-axis scales differ among panels because spectral power decreases with threshold.",
        ha="right", va="bottom", fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.985))
    return core.save_figure(fig, paths, "Figure_08_LCC_Welch_spectra")


def figure8(paths: core.Paths,
            products: dict[float, dict[str, np.ndarray | float]]) -> list[Path]:
    fig = plt.figure(figsize=(13, 10))
    grid_spec = fig.add_gridspec(
        4, 2, width_ratios=(5, 1.25),
        height_ratios=(1, 1, 1, 0.075),
        left=0.075, right=0.975, top=0.94, bottom=0.075,
        hspace=0.34, wspace=0.12,
    )
    levels = np.linspace(-2, 2, 17)
    image = None
    wavelet_axes: list[plt.Axes] = []
    for row, threshold in enumerate(core.THRESHOLDS_C):
        product = products[threshold]
        dates = pd.to_datetime(np.asarray(product["dates"]))
        period = np.asarray(product["period_days"]) / 365.25
        power = np.asarray(product["power"])
        ratio = np.asarray(product["significance_ratio"])
        coi = np.asarray(product["coi_days"]) / 365.25
        ax = fig.add_subplot(grid_spec[row, 0])
        wavelet_axes.append(ax)
        image = ax.contourf(dates, period, np.log10(np.maximum(power, 1e-6)),
                            levels=levels, cmap="Greys", extend="both")
        ax.contour(dates, period, ratio, levels=[1], colors="black", linewidths=0.55)
        ax.fill_between(dates, coi, period.max(), color="white", alpha=0.55,
                        hatch="//", edgecolor="0.5", linewidth=0)
        ax.set_yscale("log", base=2)
        period_ticks = np.asarray([0.125, 0.25, 0.5, 1, 2, 4, 8, 10])
        period_ticks = period_ticks[
            (period_ticks >= np.nanmin(period)) & (period_ticks <= np.nanmax(period))
        ]
        ax.set_yticks(period_ticks)
        ax.set_yticklabels([f"{value:g}" for value in period_ticks])
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_ylim(min(10.0, float(np.nanmax(period))),
                    max(0.125, float(np.nanmin(period))))
        ax.set_ylabel(f"≥{threshold:g}°C period\n(years)")
        core.panel_label(ax, f"{chr(65 + row * 2)})")
        format_paper_time_axis(ax)
        if row < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Year")
        global_ax = fig.add_subplot(grid_spec[row, 1], sharey=ax)
        global_ax.plot(np.asarray(product["global_power"]), period, "k-")
        global_ax.plot(np.asarray(product["global_significance_95"]), period,
                       color="0.5", ls="--")
        global_ax.set_xscale("log")
        global_ax.set_yticks(period_ticks)
        global_ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        global_ax.grid(True, color="0.85", ls=":", lw=0.5)
        # Hide labels only on the global-spectrum axes.  Do not call
        # set_yticklabels([]): with sharey that replaces the formatter on the
        # wavelet axes too and erases their decimal period labels.
        global_ax.tick_params(axis="y", which="both", labelleft=False)
        global_ax.set_xlabel("Global power")
        core.panel_label(global_ax, f"{chr(66 + row * 2)})")
    # The colorbar occupies the fourth GridSpec row under the wavelet column.
    # Consequently it has exactly the same left/right boundaries as every
    # wavelet panel and remains immediately below them after export.
    colorbar_axis = fig.add_subplot(grid_spec[3, 0])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(r"$\log_{10}$ normalized wavelet power")
    spacer_axis = fig.add_subplot(grid_spec[3, 1])
    spacer_axis.set_axis_off()
    # Matplotlib may alter a supplied colorbar axis by a few floating-point
    # units while drawing.  Enforce the final left edge and width directly
    # instead of aborting on an inconsequential numerical difference.
    fig.canvas.draw()
    wavelet_position = wavelet_axes[-1].get_position()
    colorbar_position = colorbar_axis.get_position()
    colorbar_axis.set_position([
        wavelet_position.x0,
        colorbar_position.y0,
        wavelet_position.width,
        colorbar_position.height,
    ])
    fig.suptitle("Morlet wavelet variability of PWP-LCC area", y=0.985)
    return core.save_figure(fig, paths, "Figure_09_LCC_Morlet_wavelet")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    core.configure_logging(args.log_level)
    core.configure_matplotlib()
    paths = core.resolve_paths(args.project_root)
    # Never overwrite figures from an earlier audited run.  Besides preserving
    # provenance, a version-specific directory prevents Windows/Pillow errors
    # when an existing PNG is open in Photos, Word, Explorer Preview, or a
    # synchronization client.  Numerical inputs remain in the standard JCLI
    # tables/metadata directories; only figure output is versioned here.
    version_directory = f"v{PROGRAM_VERSION.replace('.', '_')}"
    paths = replace(paths, figures=paths.figures / version_directory)
    core.ensure_directories(paths)
    logging.info("Versioned figure directory: %s", paths.figures)
    # Remove only the known obsolete snapshot-map products from pre-v1.0.1.
    # They are not part of the present 3 x 2 full-period occurrence Figure 1.
    for suffix in ("png", "pdf", "tiff"):
        obsolete = paths.figures / f"Figure_01_Canonical_vs_LCC_maps.{suffix}"
        if obsolete.is_file():
            obsolete.unlink()
            logging.info("Removed obsolete figure: %s", obsolete)
    analysis_metadata = validate_analysis_metadata(paths)

    daily = read_csv(paths.daily, ("date",))
    annual = read_csv(paths.annual, ("date",))
    trends = read_csv(paths.trend)
    climatology = read_csv(paths.climatology)
    stl = read_csv(paths.stl_daily, ("date",))
    read_csv(paths.stl_summary)  # Required provenance product; values are not recalculated here.
    read_csv(paths.detachment_summary)  # Quantitative source for the new Figure 2 interpretation.
    welch = load_product_dictionary(paths.spectral_npz)
    wavelet = load_product_dictionary(paths.wavelet_npz)

    if not paths.occurrence_npz.is_file():
        raise FileNotFoundError(
            f"Spatial-figure source is missing: {paths.occurrence_npz}. "
            "Rerun JCLI01 without --no-raw-maps."
        )
    with np.load(paths.occurrence_npz, allow_pickle=False) as archive:
        occurrence = {name: archive[name] for name in archive.files}

    generated: list[Path] = []
    generated += figure1_occurrence_climatology(paths)
    generated += figure2_disconnected_occurrence(paths, occurrence)
    generated += figure2(paths, daily, annual, trends)
    generated += figure2(paths, daily, annual, trends, common_y_scale=True)
    generated += figure3(paths, daily)
    generated += figure4(paths, annual, trends)
    generated += figure5(paths, climatology, stl)
    generated += figure6(paths, occurrence)
    generated += figure7(paths, welch)
    generated += figure8(paths, wavelet)

    metadata_path = paths.metadata / "JCLI02_FIGURE_METADATA.json"
    metadata = {
        "program": PROGRAM_NAME,
        "version": PROGRAM_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "program_file": str(Path(__file__).resolve()),
        "program_sha256": sha256(Path(__file__).resolve()),
        "source_analysis_version": analysis_metadata["version"],
        "source_analysis_program": analysis_metadata.get("program_basename"),
        "source_analysis_sha256": analysis_metadata.get("program_sha256"),
        "raw_netcdf_accessed": False,
        "figures": [str(path.resolve()) for path in generated],
    }
    core.save_json(metadata, metadata_path)

    manifest_path = paths.metadata / "JCLI02_FIGURE_SHA256.csv"
    manifest = pd.DataFrame([
        {
            "relative_path": str(path.relative_to(paths.root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in generated + [metadata_path]
    ])
    core.atomic_csv(manifest, manifest_path)
    logging.info("Completed %s v%s", PROGRAM_NAME, PROGRAM_VERSION)
    logging.info("Figures written to %s", paths.figures)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Fatal error")
        raise
