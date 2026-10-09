"""
Downloads environmental driver data (SST, chlorophyll, sea surface height anomalies)
from the Copernicus Marine Service (CMEMS).
"""
import gc
import os

import pandas as pd
import xarray as xr
import copernicusmarine

from config import PERIODS, CMEMS_USERNAME, CMEMS_PASSWORD

OUTPUT_DIR = "data/raw/driver_data/environmental"


def month_chunks(start_date, end_date):
    """Split a date range into month chunks.

    Args:
        start_date (str): Start of the range ("YYYY-MM-DD").
        end_date (str): End of the range ("YYYY-MM-DD").

    Yields:
        tuple[str, str]: (chunk_start, chunk_end)
    """
    starts = pd.date_range(start_date, end_date, freq="MS")
    if pd.Timestamp(start_date) < starts[0]:
        starts = starts.insert(0, pd.Timestamp(start_date))
    for chunk_start in starts:
        chunk_end = min(chunk_start + pd.offsets.MonthEnd(0), pd.Timestamp(end_date))
        yield chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")


def download_sst(period, res=0.25):
    """Downloads SST one month at a time (113GB total at native 0.05deg),
    regridding each month to 0.25deg and combines the monthly files into one full-period NetCDF.

    Args:
        period (str): Key into PERIODS dict identifying which date range to download.
        res (float, optional): Target grid resolution in degrees. Defaults to 0.25.

    Returns:
        None: Writes sst_{period}.nc to OUTPUT_DIR.
    """
    start_date, end_date = PERIODS[period]
    native_res = 0.05
    factor = round(res / native_res)
    chunk_file = f"{OUTPUT_DIR}/_sst_chunk.nc"
    out_file = f"{OUTPUT_DIR}/sst_{period}.nc"
    
    temp_files = []

    for i, (chunk_start, chunk_end) in enumerate(month_chunks(start_date, end_date)):
        copernicusmarine.subset(
            dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
            variables=["analysed_sst"],
            minimum_longitude=-179.975,
            maximum_longitude=179.975,
            minimum_latitude=-89.975,
            maximum_latitude=89.975,
            start_datetime=f"{chunk_start}T00:00:00",
            end_datetime=f"{chunk_end}T00:00:00",
            output_filename="_sst_chunk.nc",
            output_directory=OUTPUT_DIR,
            username=CMEMS_USERNAME,
            password=CMEMS_PASSWORD,
        )

        with xr.open_dataset(chunk_file, chunks={"time": 1}) as ds:
            da = (ds.analysed_sst - 273.15).astype("float32")
            da = da.coarsen(longitude=factor, latitude=factor, boundary="exact").mean()
            da = da.astype("float32").compute()
            month_ds = da.to_dataset(name="sst")
            
            # save to temporary per-month file
            temp_out = f"{OUTPUT_DIR}/_sst_temp_{chunk_start}.nc"
            month_ds.to_netcdf(temp_out)
            temp_files.append(temp_out)

        os.remove(chunk_file)
        gc.collect()
        print(f"sst {period}: downloaded through {chunk_end}")

    # put the temporary chunks into one NetCDF file
    print(f"Combining temporary files into {out_file}...")
    with xr.open_mfdataset(temp_files, combine="nested", concat_dim="time") as full_ds:
        full_ds.to_netcdf(out_file)
        
    # clean up
    for f in temp_files:
        os.remove(f)


def download_chl(period):
    """Downloads the chlorophyll-a source data for one period.

    Args:
        period (str): Key into PERIODS dict identifying which date range to download.

    Returns:
        None: Writes chl_{period}.nc to OUTPUT_DIR.
    """
    start_date, end_date = PERIODS[period]
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc_my_0.25deg_P1M-m",
        variables=["chl"],
        minimum_longitude=-180,
        maximum_longitude=179.75,
        minimum_latitude=-80,
        maximum_latitude=90,
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T00:00:00",
        minimum_depth=0.5057600140571594,
        maximum_depth=0.5057600140571594,
        output_filename=f"chl_{period}.nc",
        output_directory=OUTPUT_DIR,
        username=CMEMS_USERNAME,
        password=CMEMS_PASSWORD,
    )


def download_altimetry(variable, period):
    """Downloads one geostrophic velocity anomaly component for one period.

    Args:
        variable (str): Which component to download, 'ugosa' or 'vgosa'.
        period (str): Key into PERIODS dict identifying which date range to download.

    Returns:
        None: Writes {variable}_{period}.nc to OUTPUT_DIR.
    """
    start_date, end_date = PERIODS[period]
    copernicusmarine.subset(
        dataset_id="c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D",
        variables=[variable],
        minimum_longitude=-179.875,
        maximum_longitude=179.875,
        minimum_latitude=-89.875,
        maximum_latitude=89.875,
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T00:00:00",
        output_filename=f"{variable}_{period}.nc",
        output_directory=OUTPUT_DIR,
        username=CMEMS_USERNAME,
        password=CMEMS_PASSWORD,
    )


if __name__ == "__main__":
    for period in PERIODS:
        download_sst(period)
        # download_chl(period)
        # download_altimetry("vgosa", period)
        # download_altimetry("ugosa", period)