"""
download_broanband.py
John Ragland, June 2023

download_broadband.py takes a csv file containing a list of
sensors and time segments that you would like to download,
and downloads them to your local machine. You can specify the file format
that you want them to be saved. Supported file formats at this time 
include ['.mat', '.pkl'].

example csv file:
-----------------
node, start_time, end_time, file_format
LJ01C, 2018-01-01T00:00:00, 2018-01-01T00:01:00, .pkl

- create a csv file with the above contents and save it in your working path

script usage:
-------------
python download_broadband.py --csv 
"""

import argparse
import pandas as pd
import sys

# Create the argument parser
parser = argparse.ArgumentParser()

# Add command-line options
parser.add_argument("--csv", help="file path to csv file")

# Parse the command-line arguments
args = parser.parse_args()

# Check if the --path_to_csv option is present
if args.csv is None:
    raise Exception("You must provide a path to the csv file, --csv <absolute file path>")

# Access the values of the command-line options
df = pd.read_csv(args.csv)

# estimate total download size and ask to proceed
total_time = 0
for k, item in df.iterrows():
    total_time += (pd.Timestamp(item.end_time) - pd.Timestamp(item.start_time)).value/1e9

total_storage = total_time*64e3*8 # 8 Bytes per sample

def format_bytes(size):
    power = 2**10  # Power of 2^10
    n = 0
    units = ["B", "KB", "MB", "GB", "TB"]

    while size >= power and n < len(units) - 1:
        size /= power
        n += 1

    formatted_size = "{:.2f} {}".format(size, units[n])
    return formatted_size

print(f'total uncompressed download size: ~{format_bytes(total_storage)}')
proceed = input("Do you want to proceed? (y/n): ")

if proceed.lower() != 'y':
    print("Exiting the script.")
    sys.exit(0)

