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
OOIPY is designed to run on Python version 3.9 and above. To install OOIPY, run the following command:

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

  conda create --name env_name python=3.10 pip

3. Activate the environment using this command:

.. code-block :: bash

  source activate env_name

4. Install ooipy using this command in the same directory where the repository has been cloned:

.. code-block :: bash

  pip install -e ooipy

5. Install the development environment package requirements :

.. code-block :: bash

  cd ooipy
  pip install -r dev-requirements.txt

5. When this runs successfully, open a python prompt to verify if installation is proper, using this command:

.. code-block :: bash

  python
  import ooipy
  print(ooipy.__file__)

If the path printed out matches the __init__.py from your local installation path for the ooipy github repository, your development environment has been properly setup.
