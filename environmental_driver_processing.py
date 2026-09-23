import numpy as np
import xarray as xr
import rioxarray

RAW_DIR = "data/raw/driver_data/environmental"
OUT_DIR = "data/processed"


def save_raster(da_mean, path):
    da_mean = da_mean.sortby('latitude', ascending=False)  # correct orientaion
    raster = da_mean.rio.write_crs("EPSG:4326")
    raster.rio.to_raster(path)


def process_dist2coast(threshold_km=600):
    ds = xr.load_dataset(f"{RAW_DIR}/dist2coast_0.25deg_global.nc")
    da_mean = ds["dist"]

    # constant value beyond 400km from 200km dist2coast (boundary of EEZs)
    da_mean = da_mean.where(da_mean <= threshold_km, threshold_km)

    save_raster(da_mean, f"{OUT_DIR}/dist2coast.tif")


def process_eke(period):
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
    for variable, period in [("sst", "2017_2019"), ("sst", "2023_2025"),
                               ("chl", "2017_2019"), ("chl", "2023_2025")]:
        ds = xr.load_dataset(f"{RAW_DIR}/{variable}_{period}.nc")
        da_mean = ds[variable].mean(dim="time", skipna=True)
        if variable == "chl":
            da_mean = np.log10(da_mean)
        save_raster(da_mean, f"{OUT_DIR}/{variable}_{period}.tif")


def process_sst_variability(period):
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