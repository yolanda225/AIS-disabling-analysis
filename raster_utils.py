"""
Shared raster helpers.
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin
from sklearn.neighbors import BallTree


def write_raster(array, outfile, res=0.25, xmin=-180, ymin=-90, xmax=180, ymax=90):
    """Write a 2D array to a GeoTIFF on a lon/lat grid.

    Args:
        array (numpy.ndarray): 2D array to write, shaped (rows, cols) north-up.
        outfile (str): Path to write the GeoTIFF to.
        res (float, optional): Grid resolution in degrees. Defaults to 0.25.
        xmin (float, optional): Western edge of the grid, in degrees. Defaults to -180.
        ymin (float, optional): Southern edge of the grid, in degrees. Defaults to -90.
        xmax (float, optional): Eastern edge of the grid, in degrees. Defaults to 180.
        ymax (float, optional): Northern edge of the grid, in degrees. Defaults to 90.

    Returns:
        None: Writes outfile.
    """
    transform = from_origin(xmin, ymax, res, res)
    out_meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
        "compress": "lzw",
    }
    with rasterio.open(outfile, "w", **out_meta) as dst:
        dst.write(array, 1)


def read_raster(path):
    """Read the first band of a raster into a numpy array.

    Args:
        path (str): Path to the raster file.

    Returns:
        numpy.ndarray: The raster's first band.
    """
    with rasterio.open(path) as src:
        return src.read(1)


def distance_to_nearest_point_raster(points_latlon, res=0.25, cap_km=400,
                                      xmin=-180, ymin=-90, xmax=180, ymax=90):
    """
    Distance (km) from the center of every grid cell to the nearest point in
    points_latlon (an [N, 2] array of [lat, lon] in degrees), capped at cap_km.

    Args:
        points_latlon (numpy.ndarray): [N, 2] array of [lat, lon] in degrees.
        res (float, optional): Grid resolution in degrees. Defaults to 0.25.
        cap_km (float, optional): Maximum distance value, anything farther is
            clipped to this. Defaults to 400.
        xmin (float, optional): Western edge of the grid in degrees. Defaults to -180.
        ymin (float, optional): Southern edge of the grid in degrees. Defaults to -90.
        xmax (float, optional): Eastern edge of the grid in degrees. Defaults to 180.
        ymax (float, optional): Northern edge of the grid in degrees. Defaults to 90.

    Returns:
        numpy.ndarray: 2D float32 array of distances in km, one per grid cell.
    """
    points_rad = np.radians(points_latlon)
    tree = BallTree(points_rad, metric="haversine")

    lon = np.arange(xmin + res / 2, xmax, res)
    lat = np.arange(ymax - res / 2, ymin, -res)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    grid_points_rad = np.radians(
        np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
    )

    dist_rad, _ = tree.query(grid_points_rad, k=1)
    dist_km = dist_rad.ravel() * 6371.0088  # Earth radius in km
    dist_km = np.clip(dist_km, None, cap_km)

    return dist_km.reshape(lat_grid.shape).astype("float32")