"""
download_broanband.py
John Ragland, June 2023

download_broadband.py takes a csv file containing a list of
sensors and time segments that you would like to download,
and downloads them to your local machine. You can specify the file format
that you want them to be saved. Supported file formats at this time
include ['.mat', '.pkl', '.nc', '.wav'].

example csv file:
-----------------
node,start_time,end_time,file_format
LJ03A,2019-08-03T08:00:00,2019-08-03T08:01:00,pkl
LJ03A,2019-08-03T12:01:00,2019-08-03T12:02:00,pkl

- create a csv file with the above contents and save it in your working path

script usage:
-------------
python download_broadband.py --csv path/to/csv --output_path path/to/output
"""

import argparse
import sys

import pandas as pd
from tqdm import tqdm

import ooipy

hyd_type = {
    "LJ01D": "BB",
    "LJ01A": "BB",
    "PC01A": "BB",
    "PC03A": "BB",
    "LJ01C": "BB",
    "LJ03A": "BB",
    "AXBA1": "LF",
    "AXCC1": "LF",
    "AXEC2": "LF",
    "HYS14": "LF",
    "HYSB1": "LF",
}

# Create the argument parser
parser = argparse.ArgumentParser()

# Add command-line options
parser.add_argument("--csv", help="file path to csv file")
parser.add_argument("--output_path", help="file path to save files in")

# Parse the command-line arguments
args = parser.parse_args()

# Check if the --path_to_csv option is present
if args.csv is None:
    raise Exception("You must provide a path to the csv file, --csv <absolute file path>")
if args.output_path is None:
    raise Exception(
        "You must provide a path to the output directory, --output_path <absolute file path>"
    )

# Access the values of the command-line options
df = pd.read_csv(args.csv)

# estimate total download size and ask to proceed
total_time = 0
for k, item in df.iterrows():
    total_time += (pd.Timestamp(item.end_time) - pd.Timestamp(item.start_time)).value / 1e9

total_storage = total_time * 64e3 * 8  # 8 Bytes per sample


def format_bytes(size):
    power = 2**10  # Power of 2^10
    n = 0
    units = ["B", "KB", "MB", "GB", "TB"]

    while size >= power and n < len(units) - 1:
        size /= power
        n += 1

    formatted_size = "{:.2f} {}".format(size, units[n])
    return formatted_size


print(f"total uncompressed download size: ~{format_bytes(total_storage)}")
proceed = input("Do you want to proceed? (y/n): ")

if proceed.lower() != "y":
    print("Exiting the script.")
    sys.exit(0)

# download the data
for k, item in tqdm(df.iterrows()):
    if item.node not in hyd_type.keys():
        print(f"node {item.node} invalid, skipping")
        continue

    start_time_d = pd.Timestamp(item.start_time).to_pydatetime()
    end_time_d = pd.Timestamp(item.end_time).to_pydatetime()

    if hyd_type[item.node] == "LF":
        hdata = ooipy.get_acoustic_data_LF(start_time_d, end_time_d, item.node)
    else:
        hdata = ooipy.get_acoustic_data(start_time_d, end_time_d, item.node)

    if hdata is None:
        print(f"no data found for {item.node} between {start_time_d} and {end_time_d}")
        continue
    # downsample
    downsample_factor = item.downsample_factor
    if item.downsample_factor <= 16:
        hdata_ds = hdata.decimate(item.downsample_factor)
    else:
        hdata_ds = hdata
        while downsample_factor > 16:
            hdata_ds = hdata_ds.decimate(16)
            downsample_factor //= 16
        hdata_ds = hdata_ds.decimate(downsample_factor)

    # save
    op_path = args.output_path
    hdat_loc = hdata_ds.stats.location
    hdat_start_time = hdata_ds.stats.starttime.strftime("%Y%m%dT%H%M%S")
    hdat_end_time = hdata_ds.stats.endtime.strftime("%Y%m%dT%H%M%S")
    filename = f'{op_path}/{hdat_loc}_{hdat_start_time}_{hdat_end_time}'
    
    print(filename)
    hdata_ds.save(filename=filename, file_format=item.file_format, wav_kwargs={"norm": True})
