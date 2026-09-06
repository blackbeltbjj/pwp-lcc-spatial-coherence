# Version and Provenance Reconciliation

## Primary frozen scientific chain

The authoritative main workflow for the Journal of Climate PWP-LCC article is:

JCLI01_pwp_lcc_analysis_v1_1_0.py
    ->
audited machine-readable scientific products
    ->
JCLI02_figures_pwp_lcc_publication_v1_1_0.py
    ->
final publication figures

JCLI01 v1.1.0 is the numerical authority.

JCLI02 v1.1.0 is the publication-figure authority and does not redefine the
science.

The following analytical provenance records are preserved:

- JCLI01_METADATA.json
- JCLI01_OUTPUT_SHA256.csv

The following figure provenance records are preserved:

- JCLI02_FIGURE_METADATA.json
- JCLI02_FIGURE_SHA256.csv

These are stage-level provenance records and are not replacements for the
final repository release manifest.

## Historical Yan (1992) comparison extension

JCLI01_pwp_lcc_analysis_v1_2_1.py is retained as a separately identified
historical/methodological extension used for the Yan (1992) comparison that
was incorporated later into the manuscript.

It does not replace or supersede the frozen JCLI01 v1.1.0 -> JCLI02 v1.1.0
scientific chain.

Its provenance is preserved independently through:

- JCLI01_v1_2_1_YAN1992_METADATA.json
- JCLI01_v1_2_1_YAN1992_SHA256.csv
- JCLI01_v1_2_1_YAN1992_COMPARISON_REPORT.txt
- outputs/tables/historical_yan1992_v1_2_1/

## JCLI03

No JCLI03 program is required for this article.

## Excluded versions

Earlier JCLI01/JCLI02 development versions, historical Yan v1.2.0 products,
cache products, RBMet material, JTECH material and unrelated PWP programs are
not part of this release candidate.

No scientific definition has been changed during repository assembly.
## Yan-extension provenance note

JCLI01 v1.2.1 is the authoritative archived version of the
Yan (1992) methodological-comparison extension used by the
Journal of Climate manuscript.

A corresponding product generated under v1.2.0 was found to
have the same SHA-256 digest as the v1.2.1 product retained in
this release. The v1.2.0 copy is therefore not duplicated in
the release.

This byte identity applies to the verified product only and
does not imply that the complete v1.2.0 and v1.2.1 workflows
are byte-identical.

The v1.2.1 source, metadata, validation records, comparison
report, and derived tables constitute the authoritative
release record for the Yan (1992) extension.
