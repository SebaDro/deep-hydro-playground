from libs.dwd import grid as dwdgrid
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process some Daymet files.")
    parser.add_argument("-i", "--inputdir", type=str, help="Directory that contains folders with yearly REGNIE files.")
    parser.add_argument("-o", "--outdir", type=str, help="Directory that will be used for stroing results.")
    parser.add_argument("-s", "--startdate", type=int, help="Start date for loading REGNIE data.")
    parser.add_argument("-e", "--enddate", type=int, help="End date for loading REGNIE data.")
    parser.add_argument("-p", "--product", type=str, choices=["regnie", "soil_temperature_5cm"], help="DWD data product"
                                                                                                      "to download.")
    parser.add_argument("-S", "--singlestorage", action="store_true", help="If set, one single NetCDF file for all will"
                                                                           " be stored all years, rather than storing "
                                                                           " a separate file for each year.")
    args = parser.parse_args()

    if args.product == "regnie":
        dwdgrid.load_and_store_regnie_files(args.startdate, args.enddate, args.inputdir, args.outdir,
                                             args.singlestorage)
    elif args.product == "soil_temperature_5cm":
        dwdgrid.load_and_store_ambeti_files(args.startdate, args.enddate, args.inputdir, args.outdir,
                                             args.singlestorage)
    else:
        print(f"Unsupported product type: '{args.product}'")


if __name__ == "__main__":
    main()
