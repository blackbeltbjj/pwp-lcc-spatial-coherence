# Pacific Mask Provenance

The PWP-LCC calculations use a precomputed Boolean Pacific Ocean mask on the
native NOAA OISST v2.1 720 x 1440 grid.

Authoritative files:

- pacific_mask_oisst.npy
- grid_lat.npy
- grid_lon.npy

The audited Boolean mask contains 318,496 active grid positions.

The scientific programs do not reconstruct the mask. They require the stored
mask and reference latitude/longitude arrays and stop if these inputs are
absent or inconsistent with the OISST grid.

Project history associates this mask with the validated Pacific-domain
preprocessing workflow, including OISST_Pacific_Ocean_v1.2.3_EN.py. That
historical preprocessing program is not part of the frozen JCLI scientific
chain and is therefore not reclassified as JCLI source code.

The map plotting extent used in the manuscript is not the mathematical
definition of the Pacific domain and must never be used to recreate the mask.

Frozen SHA-256 values:

pacific_mask_oisst.npy
59b7100eb2360a57f26436e180b812fc64e0d32a8e4d6ad03c06a41c08612ad5

grid_lat.npy
c56849bc0192ea096ee93a4d1f7957e7582f0645a264618839656846c54bdac0

grid_lon.npy
ab9dc2ce281ec6aa33d40d72a5f13a163584d4a6b4107b26c048af86412cf74c