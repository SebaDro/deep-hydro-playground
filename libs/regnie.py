import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rio
import geopandas as gpd
import glob
import os

REGNIE_X_DELTA = 1 / 60
REGNIE_Y_DELTA = 1 / 120
REGNIE_X_OFFSET = (6 - 10 * REGNIE_X_DELTA) - REGNIE_X_DELTA / 2
REGNIE_Y_OFFSET = (55 + 10 * REGNIE_Y_DELTA) + REGNIE_Y_DELTA / 2
REGNIE_X_COORDS = [x + (i * REGNIE_X_DELTA) for i, x in enumerate([REGNIE_X_OFFSET] * 611)]
REGNIE_Y_COORDS = [y - (i * REGNIE_Y_DELTA) for i, y in enumerate([REGNIE_Y_OFFSET] * 971)]
REGNIE_Y_COORDS.reverse()

AMBETI_X_DELTA = 1000
AMBETI_Y_DELTA = 1000
AMBETI_X_OFFSET = 3280414.
AMBETI_Y_OFFSET = 5237501.
AMBETI_X_COORDS = [x + (i * AMBETI_X_DELTA) for i, x in enumerate([AMBETI_X_OFFSET] * 654)]
AMBETI_Y_COORDS = [y + (i * AMBETI_Y_DELTA) for i, y in enumerate([AMBETI_Y_OFFSET] * 866)]


def regnie_as_xarray(path: str, date: str):
    """
    Creates a xarray.Dataset from an gzipped ASCII-file containing DWD REGNIE precipitation data. The ASCII-file has a
    fixed width (971 rows and 611 4-digit values per row). The resulting xarray.Dataset has longitude and latitude
    dimensions as well as a time dimension. It also gets a coordinate reference system (EPSG:4326) by using the rio
    accessor provided by rioxarray. Precipitation values have the unit 1/10 mm.

    Parameters
    ----------
    path: str
        Path to the REGNIE file.
    date: str
        Date for indexing the xarray.Dataset in the format 'yyyy-mm-dd'.

    Returns
    -------
    xarray.Dataset:
        Dataset that holds spatio-temporal precipitation data [1/10 mm].

    """
    pd_regnie = pd.read_fwf(path, header=None, widths=[4] * 611, nrows=971, na_values=-999, compression='gzip')
    np_regnie = pd_regnie.to_numpy()
    np_regnie = np.expand_dims(np_regnie, axis=0)
    np_regnie = np.flip(np_regnie, axis=1)

    xds = xr.Dataset(
        data_vars=dict(precipitation=(["time", "y", "x"], np_regnie)),
        coords=dict(time=[np.datetime64(date)], y=REGNIE_Y_COORDS, x=REGNIE_X_COORDS)
    )

    xds.rio.write_crs(4326, inplace=True)
    xds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    xds.rio.write_coordinate_system(inplace=True)
    return xds


def ambeti_as_xarray(path: str, date: str):
    """
    Creates a xarray.Dataset from an ASCII-file, containing soil temperature values at the depth of 5 cm, calculated
    from the AMBETI model. The ASCII-file has a fixed width (866 rows and 654 6-digit values  per row). The resulting
    xarray.Dataset has x- and y-dimensions as well as a time dimension. It also gets a coordinate reference system
    (EPSG:31467) by using the rio accessor provided by rioxarray. Soil temperature values have the unit 1/10 °C.

    Parameters
    ----------
    path: str
        Path to the ASCII-file.
    date: str
        Date for indexing the xarray.Dataset in the format 'yyyy-mm-dd'.

    Returns
    -------
    xarray.Dataset:
        Dataset that holds spatio-temporal soil temperature data [1/10 °C].

    """
    pd_ambeti = pd.read_fwf(path, header=None, widths=[6] * 654, nrows=866, skiprows=6, na_values=-9999)
    np_ambeti = pd_ambeti.to_numpy()
    np_ambeti = np.expand_dims(np_ambeti, axis=0)
    np_ambeti = np.flip(np_ambeti, axis=1)

    xds = xr.Dataset(
        data_vars=dict(soil_temperature=(["time", "y", "x"], np_ambeti)),
        coords=dict(time=[np.datetime64(date)], y=AMBETI_Y_COORDS, x=AMBETI_X_COORDS)
    )

    xds.rio.write_crs(31467, inplace=True)
    xds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    xds.rio.write_coordinate_system(inplace=True)
    return xds


def load_and_store_regnie_files(start_year: int, end_year: int, base_path: str, out_path: str, single_storage: bool = True):
    """
    Loads multiple REGNIE files for the specified years from a base directory as xarray.Dataset and stores it as NetCDF
    files.

    Parameters
    ----------
    start_year: int
        Start year
    end_year: int
        End year
    base_path: str
        Path to the base directory which contains the REGNIE files
    out_path: str
        Storage path for the resulting NetCDF files
    single_storage: bool
        Indicates whether to store NetCDF files for each year or one single NetCDF files, that comprises
        REGNIE data for all years.

    """
    years = list(range(start_year, end_year + 1))
    dir_paths = [f"{base_path}/ra{year}m" for year in years]
    xds_list = []
    for year, dir_path in zip(years, dir_paths):
        file_paths = glob.glob(f"{dir_path}/ra**.gz", recursive=True)
        nr_files = len(file_paths)
        if len(file_paths) == 0:
            raise FileNotFoundError(f"Can't find files within directory {dir_path}.")
        dates = [f"{year}-{fp[4:6]}-{fp[6:8]}" for fp in [os.path.basename(fp) for fp in file_paths]]
        print(f"Read REGNIE files for year {year}.")
        for i, (fp, date) in enumerate(zip(file_paths, dates)):
            print(f"Reading file {i + 1} of {nr_files} from directory {dir_path}", end="\r")
            try:
                xds_list.append(regnie_as_xarray(fp, date))
            except ValueError as e:
                print(f"Error reading file {fp}: {e}")
        if single_storage:
            xds = xr.concat(xds_list, dim="time")
            out_file_path = os.path.join(out_path, f"regnie_{year}.nc")
            xds.to_netcdf(out_file_path)
            print(f"Stored file {out_file_path}.")
            xds_list.clear()
    if not single_storage:
        xds = xr.concat(xds_list, dim="time")
        out_file_path = os.path.join(out_path, f"regnie_full.nc")
        xds.to_netcdf(out_file_path)
        print(f"Stored file {out_file_path}.")


def merge_and_clip_regnie(netcdf_path: str, geom_path: str, out_path: str, start_year: int, end_year: int):
    """
    Merges multiple REGNIE NetCDF files into a single one and clips it by using the bounding box of a dedicated geometry.
    All files that contain REGNIE data between the specified start and end date will be considered.

    Note: Be sure that REGNIE files and the geometry file have the same CRS, which, by default, should be WGS 84
    (EPSG:4326).

    Parameters
    ----------
    netcdf_path: str
        Path to the directory that contains REGNIE NetCDF files, which will be merged.
    geom_path: str
        Path to a file that contains a geometry, which will be used for clipping the merged REGNIE files.
    out_path: str
        Path for storing the resulting NetCDF file.
    start_year: int
        Start year for considering REGNIE NetCDF files
    end_year
        End year (inclusive) for considering REGNIE NetCDF files

    """
    file_paths = [os.path.join(netcdf_path, f"regnie_{year}.nc") for year in range(start_year, end_year + 1)]

    xds = xr.open_mfdataset(file_paths, parallel=False)
    xds.rio.write_crs(4326, inplace=True)

    basin = gpd.read_file(geom_path)

    xmin = basin.geometry.total_bounds[0]
    ymin = basin.geometry.total_bounds[1]
    xmax = basin.geometry.total_bounds[2]
    ymax = basin.geometry.total_bounds[3]

    xds_clipped = xds.rio.clip_box(xmin, ymin, xmax, ymax)
    xds_clipped["precipitation"].attrs.pop("grid_mapping", None)

    xds_clipped.to_netcdf(out_path)
