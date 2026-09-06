# Data

## Primary observational dataset

The analysis uses NOAA/NCEI Daily Optimum Interpolation Sea Surface
Temperature Version 2.1 (OISST v2.1), daily 0.25-degree resolution.

Dataset DOI:
https://doi.org/10.25921/RE9P-PT57

Analysis interval:
1981-09-01 through 2026-07-29.

The complete record contains 16,403 consecutive daily observations per
threshold.

The repository does not contain the 46 annual NOAA NetCDF files. Their
filenames, byte sizes, SHA-256 hashes and source information are recorded in:

validation/JCLI_INPUT_SHA256.csv

## Spatial inputs

The repository contains the three validated spatial inputs used by the
scientific workflow:

- data/masks/pacific_mask_oisst.npy
- data/masks/grid_lat.npy
- data/masks/grid_lon.npy

The Pacific mask is an authoritative numerical input. It must not be
reconstructed from the geographic extent displayed in publication figures.

## Derived data

data/derived/JCLI_pwp_lcc_occurrence_persistence.npz contains the compact
occurrence/persistence product required by the final workflow.

Additional machine-readable derived products are stored under outputs/tables/.