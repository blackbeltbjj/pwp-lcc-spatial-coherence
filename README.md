# Pacific Warm Pool Largest Connected Component

Scientific and computational reproducibility package supporting:

**Machado, F. V.**

*Expansion and Increasing Spatial Coherence of the Largest Connected Pacific Warm Pool, 1981-2026.*

Journal of Climate manuscript.

## Scientific Release

**Current release:** v1.0.0 - Scientific/Reproducibility Freeze

[View v1.0.0 release](https://github.com/blackbeltbjj/pwp-lcc-spatial-coherence/releases/tag/v1.0.0)

The release audit passed and the reproducibility freeze was approved before publication of the GitHub release.

## Primary Scientific Chain

`JCLI01 v1.1.0 -> audited numerical products -> JCLI02 v1.1.0 -> publication figures`

JCLI01 v1.2.1 is preserved separately as the historical Yan (1992) methodological-comparison extension. It does not supersede the primary scientific chain.

## Data

- NOAA OISST v2.1 daily
- Spatial resolution: 0.25 degrees
- Analysis interval: 1981-09-01 through 2026-07-29
- 16,403 consecutive daily observations in the common analysis record
- Multiple SST thresholds

The 46 annual NOAA OISST NetCDF files are not distributed in this repository.

Their filenames, byte sizes, SHA-256 checksums, and source information are recorded in:

`validation/JCLI_INPUT_SHA256.csv`

NOAA OISST dataset DOI:

[10.25921/RE9P-PT57](https://doi.org/10.25921/RE9P-PT57)

## Scientific Scope

The repository supports analyses of:

- largest-connected-component geometry
- spatial coherence
- long-term expansion
- robust trend estimation
- seasonal and interannual variability
- STL decomposition
- Welch power spectral density
- continuous wavelet analysis
- spatial occurrence and persistence
- historical methodological comparison
- numerical integrity
- provenance and reproducibility auditing

## Analytical Workflow

`NOAA OISST v2.1 -> Pacific mask -> threshold field -> connected-component analysis -> LCC geometry -> temporal diagnostics -> spectral analysis -> occurrence/persistence -> audited products -> publication figures`

## Repository Structure

`src/`

Authoritative scientific-analysis and publication-figure programs.

`data/masks/`

Validated Pacific mask and reference coordinate arrays.

`data/derived/`

Compact derived spatial products required by the workflow.

`outputs/`

Final figures, analytical tables, and scientific reports.

`documentation/`

Methods, mask provenance, version reconciliation, and variable definitions.

`validation/`

Input provenance, environment capture, integrity records, and release-audit material.

## Reproducibility Status

- `JCLI_RELEASE_AUDIT = PASS`
- `JCLI REPRODUCIBILITY FREEZE = APPROVED`
- GitHub release `v1.0.0 = PUBLIC`

## Archival DOI

The GitHub v1.0.0 release is public.

A Zenodo DOI will be added here only after the archive identifier has been independently verified against this released software version.

## Author

**Fabio Vieira Machado**

ORCID: [0000-0003-0723-075X](https://orcid.org/0000-0003-0723-075X)

## License

See repository licensing information.
