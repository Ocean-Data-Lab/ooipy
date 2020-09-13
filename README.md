<img src='imgs/OOIPY_Logo.png' align="right" width=150>

# OOIPY 
![PEP8](https://github.com/ooipy/ooipy/workflows/Flake8/badge.svg?branch=master)
A python toolbox for acquiring and analyzing Ocean Obvservatories Initiative (OOI) Data

## Python Package for OOI Data Processing

ooi_data_processing_library.py contains functions for processing and visualizing the hydrophone data from the OOI. In the current version, the library, while being far from exhaustive, contains basic functions for loading acoustic data and computing and plotting spectrograms and power spectral density (PSD) estimates. It also provides functions for computing spectrograms and PSD estimates using Pythons multiprocessing library to handle large amounts of acoustic data.

The demo.ipynb provides examples on how to use the functions in the library.

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/FelixSchwock/OOI-data-processing-Python
cd OOI-data-processing-Python
```

- Install Conda Packages
  - we have provided a .yml to create a new conda enviroment. Navigate to ./conda_envs and create a new environemnt given your operating system
  ```bash
  cd conda_envs
  conda env create -f ooi_linux.yml
  ```

### Demo

  - demo.ipynb provides examples of some of the most used functions in the library
