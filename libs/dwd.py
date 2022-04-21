import os
import requests as req
import tarfile

DAILY_GRID_BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily"
SOIL_PRODUCT = "soil_temperature_5cm"
FILE_PREFIX = "grids_germany_daily_soil_temperature_5cm_"
FILE_POSTFIX = ".tgz"


def download_cdc_product(out_dir: str, years: list, product: str, extract: bool = False):
    if product == SOIL_PRODUCT:
        download_soil_temperature(out_dir, years, True)
    else:
        raise ValueError(f"Unsupported product type: {product}")


def download_soil_temperature(out_dir: str, years: list, extract: bool):
    for year in years:
        print(f"Start downloading soil temperature files for year {year}.")
        out_path = os.path.join(out_dir, str(year))
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        files = [f"{FILE_PREFIX}{year}{m:02d}{FILE_POSTFIX}" for m in range(1, 13)]

        for file in files:
            url = f"{DAILY_GRID_BASE_URL}/{SOIL_PRODUCT}/{file}"
            file_out_path = os.path.join(out_path, file)
            print(f"Download soil temperature file: {url}.")
            try:
                download_file(url, file_out_path)
                print(f"Stored soil temperature file: {file_out_path}.")

                if extract:
                    print(f"Extract soil temperature file: {file_out_path}.")
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
