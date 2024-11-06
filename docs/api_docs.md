# API Documentation
This module provides the tools to download OOI data from- the OOI Raw Data
Server.

## Request Module
Tools for downloading OOI data

### Hydrophone Request
- [Oregon Shelf Base Seafloor (Fs = 64 kHz)](https://ooinet.oceanobservatories.org/data_access/?search=CE02SHBP-LJ01D-11-HYDBBA106)
    - 'LJ01D'
- [Oregon Slope Base Seafloor (Fs = 64 kHz)](https://ooinet.oceanobservatories.org/data_access/?search=RS01SLBS-LJ01A-09-HYDBBA102)
    - 'LJ01A'
- [Slope Base Shallow (Fs = 64 kHz)](https://ooinet.oceanobservatories.org/data_access/?search=RS01SBPS-PC01A-08-HYDBBA103)
    - 'PC01A'
-  [Axial Base Shallow Profiler (Fs = 64 kHz)](https://ooinet.oceanobservatories.org/data_access/?search=RS03AXPS-PC03A-08-HYDBBA303)
    - 'PC03A'
- [Offshore Base Seafloor (Fs = 64 kHz)](https://ooinet.oceanobservatories.org/data_access/?search=CE04OSBP-LJ01C-11-HYDBBA105)
    - 'LJ01C'
- [Axial Base Seafloor (Fs = 64 kHz)](https://ooinet.oceanobservatories.org/data_access/?search=RS03AXBS-LJ03A-09-HYDBBA302)
    - 'LJ03A'
- [Axial Base Seafloor (Fs = 200 Hz)](https://ooinet.oceanobservatories.org/data_access/?search=RS03AXBS-MJ03A-05-HYDLFA301)
    - 'Axial_Base'
    - 'AXABA1'
- [Central Caldera (Fs = 200 Hz)](https://ooinet.oceanobservatories.org/data_access/?search=RS03CCAL-MJ03F-06-HYDLFA305)
    - 'Central_Caldera'
    - 'AXCC1'
- [Eastern Caldera (Fs = 200 Hz)](https://ooinet.oceanobservatories.org/data_access/?search=RS03ECAL-MJ03E-09-HYDLFA304)
    - 'Eastern_Caldera'
    - 'AXEC2'
- [Southern Hydrate (Fs = 200 Hz)](https://ooinet.oceanobservatories.org/data_access/?search=RS01SUM1-LJ01B-05-HYDLFA104)
    - 'Southern_Hydrate'
    - 'HYS14'
- [Oregon Slope Base Seafloor (Fs = 200 Hz)](https://ooinet.oceanobservatories.org/data_access/?search=RS01SLBS-MJ01A-05-HYDLFA101)
    - 'Slope_Base'
    - 'HYSB1'

```{eval-rst}
.. automodule:: ooipy.request.hydrophone_request
    :members:
```

### CTD Request
```{eval-rst}
.. automodule:: ooipy.request.ctd_request
    :members:
```

## Hydrophone data object
```{eval-rst}
.. automodule:: ooipy.hydrophone.basic
    :members:
```

## CTD data object
```{eval-rst}
.. automodule:: ooipy.ctd.basic
    :members:
```