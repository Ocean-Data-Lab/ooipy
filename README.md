# OOI Data Processing

## Python Package for OOI Data Processing

ooi_data_processing_library.py contains functions for processing and visualizing the hydrophone data from the OOI. In the current version, the library, while being far from exhaustive, contains basic functions for loading acoustic data and computing and plotting spectrograms and power spectral density (PSD) estimates. It also provides functions for computing spectrograms and PSD estimates using Pythons multiprocessing library to handle large amounts of acoustic data.

The demo.ipynb provides examples on how to use the functions in the library.

## Dependancies

I've seen repos in the past that have a conda enviroment in the repo that let's you create a new enviroment with all of the dependancies. I don't know how to do this, but it would be cool to do some day.


- numpy
- json
- os
-  matplotlib
- obspy
- math
- requests
- lmxl
- scipy
- datetime
- urllib
- time
- pandas
- sys
- thredds_crawler
- multiprocessing
- pickle
