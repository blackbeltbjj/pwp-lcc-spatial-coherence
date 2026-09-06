# Pacific Warm Pool Largest Connected Component

Reproducibility package for:

Expansion and Increasing Spatial Coherence of the Largest Connected Pacific
Warm Pool, 1981-2026

Target journal: Journal of Climate

Author: Fabio Vieira Machado
ORCID: 0000-0003-0723-075X
Affiliation: Independent Researcher

## Status

This repository tree is a reproducibility release candidate.

No public GitHub release, v1.0.0 tag or Zenodo DOI is authorized until the
scientific supervisor approves the reproducibility freeze.

## Primary scientific chain

JCLI01 v1.1.0 -> audited numerical products -> JCLI02 v1.1.0 -> figures

JCLI01 v1.2.1 is preserved separately as the historical Yan (1992)
methodological comparison extension. It does not supersede the main chain.

## Repository contents

src/
Authoritative scientific and publication-figure programs.

data/masks/
Validated Pacific mask and reference coordinate arrays.

data/derived/
Compact derived spatial product required by the workflow.

outputs/
Final figures, analytical tables and scientific reports.

documentation/
Methods, mask provenance, version reconciliation and variable definitions.

validation/
Stage-level provenance, input inventory, environment capture and final
release-audit records.

## Raw NOAA data

The 46 annual NOAA OISST NetCDF files are not committed to the repository.

Their filenames, byte sizes, SHA-256 checksums and source information are
recorded in validation/JCLI_INPUT_SHA256.csv.

Dataset DOI:
https://doi.org/10.25921/RE9P-PT57