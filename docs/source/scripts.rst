.. image:: ../../imgs/ooipy_banner2.png
  :width: 700
  :alt: OOIPY Logo
  :align: left

Scripts
=======
There are a collection of useful scripts that can be used to
download and process data. The scripts are included in the package if downloaded with pypi,
but can also be downloaded and run individually from the `github repo <https://github.com/Ocean-Data-Lab/ooipy/tree/master/ooipy/scripts/>`_.

download_hydrophone_data
************************
location: ooipy/scripts/download_hydrophone_data.py

This script downloads hydrophone data that is specified in a csv.

Here is an example csv file:

| node,start_time,end_time,file_format,downsample_factor
| LJ03A,2019-08-03T08:00:00,2019-08-03T08:01:00,mat,64
| AXBA1,2019-08-03T12:01:00,2019-08-03T12:02:00,mat,1

The script can be run with the following command:

.. code-block :: bash
  python download_hydrophone_data.py --csv <path_to_csv> --output_path <output_directory>
