# Data

Large raw files are **not stored in this repository** (GitHub limit: 100 MB per file).
Download them as described below and place them in `data/raw/`. They are listed in
`.gitignore` so they are never committed by accident.

## 1. Dutch Distribution Centres 2021 Geodata  (logistics buildings and estates)
- Author: Merten Nefs (2022), 4TU.ResearchData
- DOI: https://doi.org/10.4121/19361018
- Page: https://data.4tu.nl/datasets/61e28011-f96d-4b01-900e-15145b77ee59/1
- Licence: CC BY-SA 4.0 (attribution + share alike)
- Files used:
  - `buildings2021NL_repo.csv` (attributes only, ~6 MB; **kept in this repo**, used by Analysis 1)
  - `distributioncenters2021NL_repo.gpkg` (layers `buildings2021NL`, `estates2021NL`,
    `corridor2021NL`; CRS EPSG:28992; used by Analysis 2)
- Also read: the dataset `README.txt` and `Compilation_method_of_Dutch_Distribution_Centres_2021_Geodata.pdf`

## 2. CBS square statistics, 500 m x 500 m grid (population)
- Source: Statistics Netherlands (CBS), file `cbs_vk500_2025_v1.gpkg`
  (layer `cbs_vk500_2025`, CRS EPSG:28992, column `aantal_inwoners` = inhabitants)
- Page: https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/ruimtelijke-statistieken/vierkanten
- Licence: check the CBS page and cite accordingly (TODO: confirm the licence of the 500 m 2025 file).
- The value `-99997` means "no data / suppressed" and is **not** a population. In this
  project such cells are treated as not residential.

## Folders
- `data/raw/`        original downloads (never edited by hand)
- `data/processed/`  files produced by the scripts in `src/`
