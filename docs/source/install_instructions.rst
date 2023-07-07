.. image:: ../../imgs/ooipy_banner2.png
  :width: 700
  :alt: OOIPY Logo
  :align: left

How to Install OOIPY
====================

To fork our project, you can visit our Github Repository.

OOIPY is available on PyPI.

Install Instruction
-------------------
OOIPY is designed to run on Python 3.8.5. To install OOIPY, run the following command.

.. code-block :: bash

  pip install ooipy

Please see our Getting Started Guide for a tutorial on basic OOIPY usage.


How to setup OOIPY development environment
==========================================

1. Navigate into the folder where you wish to setup ooipy and clone the repository using this command:

.. code-block :: bash

  git clone https://github.com/Ocean-Data-Lab/ooipy.git

2. Setup a new Conda environment using this command:

.. code-block :: bash

  conda create --name env_name

3. Activate the environment using this command:

.. code-block :: bash

  source activate env_name

4. Run the script to setup the development environment and install dependencies using this command:

.. code-block :: bash

  bash ooipy/dev_setup.sh

Conda will ask for permissions while installing each library, answer 'y' to each.

5. When this runs successfully, run the python file to verify if installation is proper, using this command:

.. code-block :: bash

  python ooipy/verify_setup.py

If the path printed out matches your local installation path for the ooipy github repository, your development environment has been properly setup.
