from libs import regnie
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process some Daymet files.")
    parser.add_argument("-i", "--inputdir", type=str, help="Directory that contains folders with yearly REGNIE files.")
    parser.add_argument("-o", "--outdir", type=str, help="Directory that will be used for stroing results.")
    parser.add_argument("-s", "--startdate", type=int, help="Start date for loading REGNIE data.")
    parser.add_argument("-e", "--enddate", type=int, help="End date for loading REGNIE data.")
    parser.add_argument("-S", "--singlestorage", action="store_true", help="If set, one single NetCDF file for all will"
                                                                           " be stored all years, rather than storing "
                                                                           " a separate file for each year.")
    args = parser.parse_args()
    regnie.load_and_store_regnie_files(args.startdate, args.enddate, args.inputdir, args.outdir, args.singlestorage)


if __name__ == "__main__":
    main()
