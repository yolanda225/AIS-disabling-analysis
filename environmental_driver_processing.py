"""
Processes downloaded environmental driver data (distance to coast, eddy kinetic
energy, SST, SST variability, chlorophyll-a) into 0.25deg GeoTIFFs for the BRT models.
"""

import numpy as np
import xarray as xr
import rioxarray

RAW_DIR = "data/raw/driver_data/environmental"
OUT_DIR = "data/processed"


def save_raster(da_mean, path, res=0.25):
    """Regrid a DataArray onto 0.25deg grid and write it as a GeoTIFF.

    Args:
        da_mean (xarray.DataArray): Values to write, with 'longitude'/'latitude' dims.
        path (str): Output path for the GeoTIFF.
        res (float, optional): Target grid resolution in degrees. Defaults to 0.25.

    Returns:
        None: Writes path.
    """
    lon_t = np.arange(-180 + res / 2, 180, res)
    lat_t = np.arange(-90 + res / 2, 90, res)
    da_mean = da_mean.interp(longitude=lon_t, latitude=lat_t, method="linear")
    da_mean = da_mean.sortby('latitude', ascending=False)  # correct orientation
    raster = da_mean.rio.write_crs("EPSG:4326")
    raster.rio.to_raster(path)


def process_dist2coast(min_km=92.6, max_km=770.4):
    """Clip the distance-to-coast raster to the paper's bounds.

    Args:
        min_km (float, optional): Lower clip bound (50nm). Defaults to 92.6.
        max_km (float, optional): Upper clip bound (200nm EEZ + 400km). Defaults to 770.4.

    Returns:
        None: Writes dist2coast.tif to OUT_DIR.
    """
    ds = xr.load_dataset(f"{RAW_DIR}/dist2coast_0.25deg_global.nc")
    da_mean = ds["dist"]
    # clip to [50nm, 200nm EEZ boundary + 400km] as in the paper
    da_mean = da_mean.clip(min_km, max_km)
    save_raster(da_mean, f"{OUT_DIR}/dist2coast.tif")


def process_eke(period):
    """Compute climatological eddy kinetic energy from geostrophic velocity anomalies.

    Args:
        period (str): Which period's ugosa_{period}.nc/vgosa_{period}.nc to process.

    Returns:
        None: Writes eke_{period}.tif to OUT_DIR.
    """
    # Eddy kinetic Energy
    da_ugosa = xr.open_dataset(
        f"{RAW_DIR}/ugosa_{period}.nc",
        chunks={"time": 50}  # process in chunks along time, not all at once
    ).ugosa

    da_vgosa = xr.open_dataset(
        f"{RAW_DIR}/vgosa_{period}.nc",
        chunks={"time": 50}
    ).vgosa

    da_eddy = 0.5 * (da_ugosa**2 + da_vgosa**2)

    eke_mean = da_eddy.mean(dim="time", skipna=True)
    eke_mean = eke_mean.compute()

    save_raster(eke_mean, f"{OUT_DIR}/eke_{period}.tif")


def process_sst_chl():
    """Compute SST and chlorophyll-a for every period.

    Returns:
        None: Writes sst_{period}.tif and chl_{period}.tif to OUT_DIR for each period.
    """
    for variable, period in [("sst", "2017_2019"), ("sst", "2023_2025"),
                               ("chl", "2017_2019"), ("chl", "2023_2025")]:
        ds = xr.load_dataset(f"{RAW_DIR}/{variable}_{period}.nc")
        da_mean = ds[variable].mean(dim="time", skipna=True)
        if variable == "chl":
            da_mean = np.log10(da_mean)
        save_raster(da_mean, f"{OUT_DIR}/{variable}_{period}.tif")


def process_sst_variability(period):
    """Compute the temporal standard deviation of daily SST for given period.

    Args:
        period (str): Which period's sst_{period}.nc to process.

    Returns:
        None: Writes sst_variability_{period}.tif to OUT_DIR.
    """
    ds = xr.load_dataset(f"{RAW_DIR}/sst_{period}.nc")
    da_mean = ds["sst"].std(dim="time", skipna=True, ddof=1)

    save_raster(da_mean, f"{OUT_DIR}/sst_variability_{period}.tif")


if __name__ == "__main__":
    process_dist2coast()
    process_eke("2017_2019")
    process_eke("2023_2025")
    process_sst_chl()
    process_sst_variability("2017_2019")
    process_sst_variability("2023_2025")