# Methods and Reproduction

## Scientific chain

Primary analytical authority:

src/JCLI01_pwp_lcc_analysis_v1_1_0.py

Publication figure authority:

src/JCLI02_figures_pwp_lcc_publication_v1_1_0.py

Historical Yan comparison extension:

src/JCLI01_pwp_lcc_analysis_v1_2_1.py

## Dataset

NOAA OISST v2.1
0.25 degree daily SST
1981-09-01 to 2026-07-29
16,403 consecutive days

Thresholds:
28.0, 28.5 and 29.0 degrees C.

## Connected-component definition

The canonical threshold field contains all finite SST cells inside the
validated Pacific mask satisfying SST >= threshold.

The largest connected component (LCC) is selected independently each day
using eight-neighbor connectivity, including periodic longitude seam
connectivity. Components are ranked by exact spherical grid-cell area.

No smoothing, morphological filtering, hole filling or minimum-area filter is
applied before component labeling.

## Annual inference

Only complete calendar years 1982-2025 enter annual trend inference.
Partial 1981 and 2026 remain in daily descriptive analyses.

## Trend methods

Theil-Sen is the principal robust trend estimator.
OLS with Newey-West HAC covariance is used as an independent parametric check.

## STL

Daily LCC area is decomposed into trend, seasonal and remainder components
using robust STL with annual periodicity.

## Welch

Daily LCC area is linearly detrended while retaining the annual cycle.

Welch uses 2920-day Hann windows with 50 percent overlap.

Diagnostic bands overlap and their variance fractions must not be added.

## Wavelet

Morlet mother wavelet:
omega0 = 6
dt = 1 day
dj = 1/12

Inference is interpreted with the cone of influence and AR(1) significance.

## Reproduction order

A full scientific rebuild requires the externally obtained annual NOAA OISST
files plus the validated spatial inputs.

For the completed audited dataset, disconnected occurrence can be regenerated
from validated cached products without reopening OISST using:

python src/JCLI01_pwp_lcc_analysis_v1_1_0.py --project-root . --derive-detachment-only

Publication figures are generated with:

python src/JCLI02_figures_pwp_lcc_publication_v1_1_0.py --project-root .

The repository package contains compact derived products rather than the
46 large NOAA annual NetCDF files.