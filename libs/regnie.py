import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rio
import geopandas as gpd
import glob
import os

X_DELTA = 1 / 60
Y_DELTA = 1 / 120
X_OFFSET = (6 - 10 * X_DELTA) - X_DELTA / 2
Y_OFFSET = (55 + 10 * Y_DELTA) + Y_DELTA / 2
LONGITUDES = [x + (i * X_DELTA) for i, x in enumerate([X_OFFSET] * 611)]
LATITUDES = [y - (i * Y_DELTA) for i, y in enumerate([Y_OFFSET] * 971)]
LATITUDES.reverse()

def regnie_as_xarray(path: str, date: str):
    """
    Creates a xarray.Dataset from a REGNIE file. The xarray.Dataset has longitude and latitude dimensions as well as
    a time dimension. It also gets a coordinate reference system by using the rio accessor provided by rioxarray.
    Since precipitation has the unit 1/10 mm in the raw REGNIE dataset it will be converted to mm.

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

    xds = xr.Dataset(
        data_vars=dict(precipitation=(["time", "y", "x"], np.flip(np_regnie, axis=1))),
        coords=dict(time=[np.datetime64(date)], y=LATITUDES, x=LONGITUDES)
    )

    xds.rio.write_crs(4326, inplace=True)
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
