import os
import requests as req
import tarfile

DAILY_GRID_BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily"
SOIL_PRODUCT = "soil_temperature_5cm"
FILE_PREFIX = "grids_germany_daily_soil_temperature_5cm_"
ASCII_FILE_POSTFIX = ".tgz"

HYRAS_AIR_TEMPERATURE_MAX = "hyras_air_temperature_max"
HYRAS_AIR_TEMPERATURE_MIN = "hyras_air_temperature_min"
HYRAS_HUMIDITY = "hyras_humidity"
HYRAS_PRECIPITATION = "hyras_precipitation"
HYRAS_RADIATION = "hyras_radiation_global"


def download_cdc_product(out_dir: str, years: list, product: str, extract: bool = False):
    if product == SOIL_PRODUCT:
        download_soil_temperature(out_dir, years, True)
    elif product == HYRAS_AIR_TEMPERATURE_MAX:
        download_hyras_product(out_dir, years, "air_temperature_max", "tmax", "v4-0", 5)
    elif product == HYRAS_AIR_TEMPERATURE_MIN:
        download_hyras_product(out_dir, years, "air_temperature_min", "tmin", "v4-0", 5)
    elif product == HYRAS_HUMIDITY:
        download_hyras_product(out_dir, years, "humidity", "hurs", "v4-0", 5)
    elif product == HYRAS_PRECIPITATION:
        download_hyras_product(out_dir, years, "precipitation", "pr", "v3-0", 1)
    elif product == HYRAS_RADIATION:
        download_hyras_product(out_dir, years, "radiation_global", "rsds", "v2-0", 5)
    else:
        raise ValueError(f"Unsupported product type: {product}")


def download_soil_temperature(out_dir: str, years: list, extract: bool):
    for year in years:
        print(f"Start downloading soil temperature files for year {year}.")
        out_path = os.path.join(out_dir, str(year))
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        files = [f"{FILE_PREFIX}{year}{m:02d}{ASCII_FILE_POSTFIX}" for m in range(1, 13)]

        for file in files:
            url = f"{DAILY_GRID_BASE_URL}/{SOIL_PRODUCT}/{file}"
            download_and_store(out_path, file, url, extract)


def download_hyras_product(out_dir: str, years: list, product: str, file_pref: str, version: str, resolution: int):
    for year in years:
        print(f"Start downloading HYRAS {product} files for year {year}.")
        out_path = os.path.join(out_dir, str(year))
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        file = f"{file_pref}_hyras_{resolution}_{year}_{version}_de.nc"

        url = f"{DAILY_GRID_BASE_URL}/hyras_de/{product}/{file}"
        download_and_store(out_path, file, url, False)


def download_and_store(out_path: str, file: str, url: str, extract: bool):
    file_out_path = os.path.join(out_path, file)
    print(f"Download file: {url}.")
    try:
        download_file(url, file_out_path)
        print(f"Stored file: {file_out_path}.")

        if extract:
            print(f"Extract file: {file_out_path}.")
            tar = tarfile.open(file_out_path, "r:gz")
            tar.extractall(out_path)
            tar.close()
    except req.exceptions.HTTPError as e:
        print(f"Error downloading file {url}: {e}")


def download_file(url, path):
    r = req.get(url, timeout=60)
    if r.status_code != req.codes.ok:
        r.raise_for_status()
    f = open(path, "wb")
    f.write(r.content)
    f.close()
