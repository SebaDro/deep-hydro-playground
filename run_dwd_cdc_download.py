from libs import dwd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Download climate data from DWD CDC portal.")
    parser.add_argument("-o", "--outdir", type=str, help="Directory that will be used for storing downloaded files.")
    parser.add_argument("-s", "--startyear", type=int, help="Start year for file download.")
    parser.add_argument("-e", "--endyear", type=int, help="End year for file download.")
    parser.add_argument("-p", "--product", type=str, choices=["soil_temperature_5cm"], help="Data product to download.")
    args = parser.parse_args()

    dwd.download_cdc_product(args.outdir, list(range(args.startyear, args.endyear + 1)), args.product)


if __name__ == "__main__":
    main()
