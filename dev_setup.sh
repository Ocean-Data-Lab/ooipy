conda install numpy
conda install pandas
conda install matplotlib
conda install -c conda-forge obspy
conda install -c conda-forge fsspec
conda install black
conda install -c conda-forge check-manifest
conda install flake8
conda install -c conda-forge isort
conda install numpydoc
conda install -c conda-forge pre_commit
conda install -c conda-forge pylint
conda develop ooipy
python ooipy/setup.py
python ooipy/verify_setup.py
