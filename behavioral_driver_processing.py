"""
Processes MPA and piracy-event shapefiles into quarter-degree distance-to-nearest-feature
rasters, capped at 400km.
"""
import gc
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely import segmentize
import pandas as pd

from raster_utils import distance_to_nearest_point_raster, write_raster

RES = 0.25


def load_mpa_boundary_points(zip_path):
    """Extracts boundary points (densified) of one WDPA shapefile chunk.

    Args:
        zip_path (str): Path to one WDPA shapefile chunk (zip archive).

    Returns:
        numpy.ndarray: [N, 2] array of [lat, lon] boundary points.
    """
    df = gpd.read_file(zip_path, columns=[])  # skip attribute columns, keep only geometry

    df["geometry"] = df.geometry.buffer(0)  # fix invalid geometries

    # densify polygon boundaries
    df["geometry"] = df.geometry.apply(lambda geom: segmentize(geom, max_segment_length=0.25))

    # Extract all boundary points from polygons
    mpa_points = []
    for geom in df.geometry:
        boundary = geom.boundary
        if boundary.geom_type == "LineString":
            coords = np.asarray(boundary.coords, dtype="float32")
        else:
            coords = np.vstack([np.asarray(line.coords) for line in boundary.geoms]).astype("float32")
        mpa_points.append(coords)
    mpa_points = np.vstack(mpa_points)

    del df
    gc.collect()

    return mpa_points[:, [1, 0]]  # reorder (lon, lat) -> (lat, lon)


def load_piracy_points(zip_path, shapefile_path):
    """Piracy event points (ASAM), restricted to <= Dec 2019 to match the paper's
    data

    Args:
        zip_path (str): Path to the ASAM shapefile zip archive.
        shapefile_path (str): zip:// path to the .shp file within zip_path.

    Returns:
        numpy.ndarray: [N, 2] array of [lat, lon] piracy event points.
    """
    with zipfile.ZipFile(zip_path) as z:
        print(z.namelist())
    # print(gpd.read_file(shapefile_path).columns.tolist())
    df = gpd.read_file(shapefile_path, columns=["dateofocc"])  # geometry + incident date

    df["dateofocc"] = pd.to_datetime(df["dateofocc"])
    df = df[df["dateofocc"] <= "2019-12-31"]

    asam_points = np.column_stack([df.geometry.y, df.geometry.x]).astype("float32")  # already [lat, lon]

    del df
    gc.collect()

    return asam_points


def combine_rasters_min(paths, outfile):
    """Combine several distance rasters into one by taking the per-pixel minimum.

    Args:
        paths (list[str]): Paths to the rasters to combine.
        outfile (str): Path to write the combined raster to.

    Returns:
        None: Writes outfile.
    """
    arrays = []
    for p in paths:
        with rasterio.open(p) as src:
            arrays.append(src.read(1))
            profile = src.profile  # keep metadata from last file

    combined = np.minimum.reduce(arrays)

    with rasterio.open(outfile, "w", **profile) as dst:
        dst.write(combined, 1)


def process_mpa_distance(zip_paths, chunk_outfiles, combined_outfile, res=RES):
    """Build a distance-to-nearest-MPA raster for each WDPA shapefile chunk, then
    combine them into one global raster.

    Args:
        zip_paths (list[str]): Paths to the WDPA shapefile chunk zip archives.
        chunk_outfiles (list[str]): Output paths for each chunk's distance raster,
            matching zip_paths by position.
        combined_outfile (str): Output path for the combined distance raster.
        res (float, optional): Grid resolution in degrees. Defaults to RES.

    Returns:
        None: Writes chunk_outfiles and combined_outfile.
    """
    for zip_path, outfile in zip(zip_paths, chunk_outfiles):
        points = load_mpa_boundary_points(zip_path)
        dist_raster = distance_to_nearest_point_raster(points, res=res, cap_km=400)
        del points
        gc.collect()
        write_raster(dist_raster, outfile, res=res)

    combine_rasters_min(chunk_outfiles, combined_outfile)

    with rasterio.open(combined_outfile) as src:
        data = src.read(1)
    plt.imshow(data, cmap="viridis")
    plt.colorbar(label="Distance to MPA (km)")
    plt.show()


def process_piracy_distance(zip_path, shapefile_path, outfile, res=RES):
    """Build a distance-to-nearest-piracy-event raster.

    Args:
        zip_path (str): Path to the ASAM shapefile zip archive.
        shapefile_path (str): zip:// path to the .shp file within zip_path.
        outfile (str): Output path for the distance raster.
        res (float, optional): Grid resolution in degrees. Defaults to RES.

    Returns:
        None: Writes outfile.
    """
    points = load_piracy_points(zip_path, shapefile_path)
    dist_raster = distance_to_nearest_point_raster(points, res=res, cap_km=400)
    write_raster(dist_raster, outfile, res=res)

    with rasterio.open(outfile) as src:
        data = src.read(1)
    plt.imshow(data, cmap="viridis")
    plt.colorbar(label="Distance to MPA (km)")  # label as in the original notebook
    plt.show()


if __name__ == "__main__":
    mpa_dir = "data/raw/driver_data/marine-protected-areas/WDPA_Jul2026_Public_shp"
    mpa_zip_paths = [f"{mpa_dir}/WDPA_Jul2026_Public_shp_{i}.zip" for i in range(3)]
    mpa_chunk_outfiles = [f"data/processed/mpa_distance{i}.tif" for i in range(3)]

    # process_mpa_distance(
    #     mpa_zip_paths,
    #     mpa_chunk_outfiles,
    #     "data/processed/mpa_distance_combined.tif",
    # )

    process_piracy_distance(
        "data/raw/driver_data/piracy/ASAM_shp.zip",
        "zip://data/raw/driver_data/piracy/ASAM_shp.zip!Asam_data_download/All ASAM Events.shp",
        "data/processed/asam_distance.tif",
    )