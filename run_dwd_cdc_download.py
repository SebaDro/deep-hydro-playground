from libs.dwd import cdc
import argparse

PRODUCTS = ["soil_temperature_5cm", "hyras_air_temperature_min", "hyras_air_temperature_max", "hyras_humidity",
            "hyras_precipitation", "hyras_radiation_global"]

def main():
    parser = argparse.ArgumentParser(description="Download climate data from DWD CDC portal.")
    parser.add_argument("-o", "--outdir", type=str, help="Directory that will be used for storing downloaded files.")
    parser.add_argument("-s", "--startyear", type=int, help="Start year for file download.")
    parser.add_argument("-e", "--endyear", type=int, help="End year for file download.")
    parser.add_argument("-p", "--product", type=str, choices=PRODUCTS, help="Data product to download.")
    args = parser.parse_args()

    cdc.download_cdc_product(args.outdir, list(range(args.startyear, args.endyear + 1)), args.product)


if __name__ == "__main__":
    main()


HYRAS_AIR_TEMPERATURE_MAX = "air_temperature_max"
HYRAS_AIR_TEMPERATURE_MIN = "air_temperature_min"
HYRAS_HUMIDITY = "humidity"
HYRAS_PRECIPITATION = "humidity"
HYRAS_RADIATION = "radiation_global"