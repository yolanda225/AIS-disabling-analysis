"""
Filtering and spatial allocation of AIS gap events, and the loitering activity raster.
"""
import ast

import numpy as np
import pandas as pd

from raster_utils import write_raster


def filter_gaps(infile, min_hours=12, max_hours=336, min_shore_km=92.6):
    """Load AIS gap events and filter to those long enough and far enough offshore
    to be considered for suspected disabling.

    Args:
        infile (str): CSV path to the raw gap events.
        min_hours (float, optional): Minimum gap duration in hours. Defaults to 12.
        max_hours (float, optional): Maximum gap duration in hours.
            kept for the hotspot-map variant of this filter. Defaults to 336.
        min_shore_km (float, optional): Minimum start distance from shore in km.
            Defaults to 92.6 (50 nautical miles).

    Returns:
        pandas.DataFrame: The filtered gap events.
    """
    df = pd.read_csv(infile)
    df['start'] = pd.to_datetime(df['start'], utc=True)
    df['end'] = pd.to_datetime(df['end'], utc=True)

    df['duration_hours'] = (df['end'] - df['start']).dt.total_seconds() / 3600
    df['distances'] = df['distances'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['start_distance_from_shore_km'] = df['distances'].apply(lambda x: x.get('start_distance_from_shore_km'))

    df = df[
        (df['duration_hours'] >= min_hours) &
        #(df['duration_hours'] <= max_hours) & # only for hotspot maps
        (df['start_distance_from_shore_km'] > min_shore_km)
    ]

    print(f"After filtering: {len(df)} events")
    print(df['duration_hours'].describe())

    return df


def add_gear_type(gaps_df, vessel_df):
    """Merge each gap event's gear type in from vessel info.

    Args:
        gaps_df (pandas.DataFrame): Gap events with a 'vessel' column.
        vessel_df (pandas.DataFrame): Vessel records with 'vessel_id' and 'gear_types'.

    Returns:
        pandas.DataFrame: gaps_df with a 'gear_type' column merged in.
    """
    vessel_df = vessel_df.copy()
    vessel_df['gear_type'] = vessel_df['gear_types'].apply(
        lambda g: (g if isinstance(g, list) else ast.literal_eval(g))[0]['name']
        if len(g if isinstance(g, list) else ast.literal_eval(g)) > 0 else None
    )

    gaps_df = gaps_df.copy()
    gaps_df['vessel'] = gaps_df['vessel'].apply(lambda v: v if isinstance(v, dict) else ast.literal_eval(v))
    gaps_df['vessel_id'] = gaps_df['vessel'].apply(lambda v: v['id'])

    gaps_df = gaps_df.merge(
        vessel_df[['vessel_id', 'gear_type']], on='vessel_id', how='left'
    )
    return gaps_df


def gap_hours_raster(df, outfile, res=0.25, n_samples=20):
    """Spatially allocate gap duration along a linear interpolation between each
    gap's start and end position, and grid the resulting hours to a raster.

    Args:
        df (pandas.DataFrame): Gap events with a 'gap' column holding off/on
            positions and duration.
        outfile (str): Path to write the resulting raster to.
        res (float, optional): Grid resolution in degrees. Defaults to 0.25.
        n_samples (int, optional): Points sampled along each gap's start-end line.
            Defaults to 20.

    Returns:
        numpy.ndarray: The gridded gap-hours array (also written to outfile).
    """
    df["gap"] = df["gap"].apply(lambda g: g if isinstance(g, dict) else ast.literal_eval(g))

    df["lon0"] = df["gap"].apply(lambda g: g["off_position"]["lon"])
    df["lat0"] = df["gap"].apply(lambda g: g["off_position"]["lat"])
    df["lon1"] = df["gap"].apply(lambda g: g["on_position"]["lon"])
    df["lat1"] = df["gap"].apply(lambda g: g["on_position"]["lat"])
    df["duration_hours"] = df["gap"].apply(lambda g: g["duration_hours"])

    # quarter-degree grid
    xmin, ymin, xmax, ymax = -180, -90, 180, 90
    nx, ny = int((xmax - xmin) / res), int((ymax - ymin) / res)
    grid = np.zeros((ny, nx), dtype="float32")

    def cell_index(lon, lat):
        """Map lon/lat coordinates to raster row/col indices on the quarter-degree grid.

        Args:
            lon (numpy.ndarray): Longitudes.
            lat (numpy.ndarray): Latitudes.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: (row, col) indices.
        """
        col = np.clip(((lon - xmin) / res).astype(int), 0, nx - 1)
        row = np.clip(((ymax - lat) / res).astype(int), 0, ny - 1)
        return row, col

    for lon0, lat0, lon1, lat1, hours in df[["lon0", "lat0", "lon1", "lat1", "duration_hours"]].itertuples(index=False):
        # handle antimeridian crossing
        if lon1 - lon0 > 180:
            lon1 -= 360
        elif lon1 - lon0 < -180:
            lon1 += 360

        t = np.linspace(0, 1, n_samples)
        lons = (lon0 + t * (lon1 - lon0) + 180) % 360 - 180  # wrap back to [-180, 180]
        lats = lat0 + t * (lat1 - lat0)

        rows, cols = cell_index(lons, lats)
        np.add.at(grid, (rows, cols), hours / n_samples)

    write_raster(grid, outfile, res=res)
    return grid


def build_loitering_grid(infile, res=0.25):
    """Sum loitering hours per grid cell and log-transform the result.

    Args:
        infile (str): CSV path to the raw loitering events.
        res (float, optional): Grid resolution in degrees. Defaults to 0.25.

    Returns:
        pandas.DataFrame: One row per grid cell, with 'lat', 'lon', 'loitering_hours',
            and 'loitering_hours_log' columns.
    """
    df_loitering = pd.read_csv(infile)

    # Extract lat/lon from position column
    df_loitering['position'] = df_loitering['position'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_loitering['lat'] = df_loitering['position'].apply(lambda x: x['lat'])
    df_loitering['lon'] = df_loitering['position'].apply(lambda x: x['lon'])

    # Extract duration in hours from loitering column
    df_loitering['loitering'] = df_loitering['loitering'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_loitering['duration_hours'] = df_loitering['loitering'].apply(lambda x: x.get('total_time_hours', 0))

    # Snap to quarter degree grid
    df_loitering['lat_grid'] = np.floor(df_loitering['lat'] / res) * res
    df_loitering['lon_grid'] = np.floor(df_loitering['lon'] / res) * res

    # Sum loitering hours per grid cell
    loitering_grid = df_loitering.groupby(['lat_grid', 'lon_grid'])['duration_hours'].sum().reset_index()
    loitering_grid.columns = ['lat', 'lon', 'loitering_hours']

    # Log transform to remove left skew (add 1 to avoid log(0))
    loitering_grid['loitering_hours_log'] = np.log1p(loitering_grid['loitering_hours'])

    print(loitering_grid.describe())

    return loitering_grid


def save_grid_as_tif(grid_df, outfile, value_col, res=0.25):
    """Write one column of a lat/lon grid DataFrame out as a GeoTIFF.

    Args:
        grid_df (pandas.DataFrame): Grid with 'lat', 'lon', and value_col columns.
        outfile (str): Path to write the raster to.
        value_col (str): Name of the column in grid_df to write as the raster's values.
        res (float, optional): Grid resolution in degrees. Defaults to 0.25.

    Returns:
        numpy.ndarray: The rasterized array (also written to outfile).
    """
    xmin, ymin, xmax, ymax = -180, -90, 180, 90
    nx, ny = int((xmax - xmin) / res), int((ymax - ymin) / res)
    raster = np.full((ny, nx), np.nan, dtype="float32")

    col = np.clip(((grid_df['lon'] - xmin) / res).astype(int), 0, nx - 1)
    row = np.clip(((ymax - res - grid_df['lat']) / res).astype(int), 0, ny - 1)
    raster[row, col] = grid_df[value_col]

    write_raster(raster, outfile, res=res)
    return raster


if __name__ == "__main__":
    periods = {
        "2017_2019": {
            "gaps_file": "data/raw/gaps_2017_2019.csv",
            "vessel_file": "data/raw/vessel_info_2017_2019.csv",
            "loitering_file": "data/raw/loitering_2017_2019.csv",
        },
        "2023_2025": {
            "gaps_file": "data/raw/gaps_2023_2025.csv",
            "vessel_file": "data/raw/vessel_info_2023_2025.csv",
            "loitering_file": "data/raw/loitering_2023_2025.csv",
        },
    }
    for period, files in periods.items():
        # df = filter_gaps(files["gaps_file"])
        # vessel_df = pd.read_csv(files["vessel_file"])
        # df = add_gear_type(df, vessel_df)
        # df.to_csv(f"data/processed/gaps_{period}_filtered.csv", index=False)

        loitering_grid = build_loitering_grid(files["loitering_file"])
        save_grid_as_tif(loitering_grid, f"data/processed/loitering_hours_{period}.tif", "loitering_hours_log")