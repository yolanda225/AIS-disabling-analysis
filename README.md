# AIS-disabling-analysis


A temporal comparison of AIS ("Automatic Identification System") disabling by fishing
vessels between 2017–2019 and 2023–2025. 
The comparison is built on the methodology from:

> Welch, H., Clavelle, T., White, T.D., et al. **"Hot spots of unseen fishing vessels."**
> *Science Advances* 8, eabq2109 (2022). https://doi.org/10.1126/sciadv.abq2109

which built the first global dataset of suspected intentional AIS disabling and modeled
where it happens as a function of behavioral drivers (loitering by carrier vessels,
distance to shore, distance to marine protected areas, distance to piracy events) and
environmental drivers (SST, SST variability, chlorophyll-a, eddy kinetic energy), using
boosted regression trees.

This project runs that pipeline over two periods:

- **`2017_2019`** — the paper's original study window. Reproducing it as closely as
  possible against the published results is the calibration step. The goal is that the
  models and driver results match the paper's.
- **`2023_2025`** — a current window, run through the identical pipeline, to see
  whether disabling hotspots and their drivers have shifted since
  the original study.

## Data sources

- **AIS gap and loitering events, vessel gear type, gridded fishing effort** — Global
  Fishing Watch API (`gfwapiclient`).
- **SST, chlorophyll-a, sea surface height anomalies (for eddy kinetic energy)** —
  Copernicus Marine Service (CMEMS) API.
- **Marine protected areas** — World Database on Protected Areas (WDPA), downloaded
  separately as shapefiles.
- **Piracy events** — NGA's Anti-Shipping Activity Messages (ASAM) database,
  downloaded separately as a shapefile.
- **Distance to shore** — separately downloaded PacIOOS `dist2coast` raster.

## Pipeline and File descriptions

- **`config.py`** — credentials (GFW token, CMEMS username/password, loaded from
  `.env`) and the two study periods, `2017_2019` and `2023_2025`.
- **`data_download.py`** — pulls AIS gap events, loitering events, and gridded
  fishing-effort hours (the source of absences) from the GFW API for both periods.
- **`environmental_driver_download.py`** — downloads SST, chlorophyll-a, and geostrophic velocity anomalies (`ugosa`/`vgosa`, for eddy kinetic
  energy) from CMEMS, for both periods.
- **`environmental_driver_processing.py`** — turns the downloaded environmental data
  into 0.25° GeoTIFFs per period: distance to coast (clipped to the paper's [50nm,
  200nm EEZ + 400km] bounds), eddy kinetic energy, SST and
  chlorophyll-a climatologies (chl log10-transformed), and SST variability
- **`behavioral_driver_processing.py`** — turns the WDPA and ASAM shapefiles into
  distance-to-nearest-feature rasters (distance to MPA, distance to piracy event),
  both capped at 400km.
- **`gap_processing.py`** — filters each period's raw AIS gaps to the paper's
  criteria (≥12 hours, starting >50nm offshore), attaches gear type, and builds the
  loitering-hours raster (log-transformed to remove left skew, like the paper).
- **`raster_utils.py`** — shared helpers used by the two driver-processing scripts
  above: writing a GeoTIFF on a fixed lon/lat grid, and computing a haversine
  distance-to-nearest-point raster.
- **`brt_modeling.py`** — the modeling step: for a given period, builds a balanced
  1:1 presence/absence dataset per gear type, samples all eight driver rasters at
  each cell, fits an XGBoost model (tuning the learning rate to reach the paper's
  ≥1,000-tree target), and saves out performance metrics (explained deviance, AUC,
  TSS) and feature importances.

## Known differences from the paper (calibration run, `2017_2019`)

- **BRT implementation**: Here XGBoost is used which can differ from the library used in the paper.
- **Data**: even for the `2017_2019` period, AIS gap/fishing data and the
  ASAM piracy database have been reprocessed/extended since.
- **CMEMS products**: some of the exact dataset IDs the paper cites (e.g. the
  now-retired `SST_GLO_SST_L4_NRT_OBSERVATIONS_010_005`) are no longer available.


## Setup

Requires a `.env` file with:

```
TOKEN=<GFW API token>
CMEMS_USERNAME=<CMEMS username>
CMEMS_PASSWORD=<CMEMS password>
```