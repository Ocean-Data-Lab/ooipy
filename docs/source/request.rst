.. image:: ../../imgs/ooipy_banner2.png
  :width: 700
  :alt: OOIPY Logo
  :align: left

Request Module
==============
This module provides the tools to download OOI data from the OOI Raw Data
Server.

Hydrophone Nodes
^^^^^^^^^^^^^^^^
* `Oregon Shelf Base Seafloor (Fs = 64 kHz) <https://ooinet.oceanobservatories.org/data_access/?search=CE02SHBP-LJ01D-11-HYDBBA106>`_
    * 'LJ01D'
* `Oregon Slope Base Seafloor (Fs = 64 kHz) <https://ooinet.oceanobservatories.org/data_access/?search=RS01SLBS-LJ01A-09-HYDBBA102>`_
    * 'LJ01A'
* `Slope Base Shallow (Fs = 64 kHz) <https://ooinet.oceanobservatories.org/data_access/?search=RS01SBPS-PC01A-08-HYDBBA103>`_
    * 'PC01A'
* `Axial Base Shallow Profiler (Fs = 64 kHz) <https://ooinet.oceanobservatories.org/data_access/?search=RS03AXPS-PC03A-08-HYDBBA303>`_
    * 'PC03A'
* `Offshore Base Seafloor (Fs = 64 kHz) <https://ooinet.oceanobservatories.org/data_access/?search=CE04OSBP-LJ01C-11-HYDBBA105>`_
    * 'LJ01C'
* `Axial Base Seafloor (Fs = 64 kHz) <https://ooinet.oceanobservatories.org/data_access/?search=RS03AXBS-LJ03A-09-HYDBBA302>`_
    * 'LJ03A'
* `Axial Base Seaflor (Fs = 200 Hz) <https://ooinet.oceanobservatories.org/data_access/?search=RS03AXBS-MJ03A-05-HYDLFA301>`_
    * 'Axial_Base'
    * 'AXABA1'
* `Central Caldera (Fs = 200 Hz) <https://ooinet.oceanobservatories.org/data_access/?search=RS03CCAL-MJ03F-06-HYDLFA305>`_
    * 'Central_Caldera'
    * 'AXCC1'
* `Eastern Caldera (Fs = 200 Hz) <https://ooinet.oceanobservatories.org/data_access/?search=RS03ECAL-MJ03E-09-HYDLFA304>`_
    * 'Eastern_Caldera'
    * 'AXEC2'
* `Southern Hydrate (Fs = 200 Hz) <https://ooinet.oceanobservatories.org/data_access/?search=RS01SUM1-LJ01B-05-HYDLFA104>`_
    * 'Southern_Hydrate'
    * 'HYS14'
* `Oregon Slope Base Seafloor (Fs = 200 Hz) <https://ooinet.oceanobservatories.org/data_access/?search=RS01SLBS-MJ01A-05-HYDLFA101>`_
    * 'Slope_Base'
    * 'HYSB1'

Hydrophone
^^^^^^^^^^
.. automodule:: ooipy.request.hydrophone_request
    :members:

CTD
^^^
.. automodule:: ooipy.request.ctd_request
    :members:
