'''
Unit tests for request module
'''

import ooipy.request.hydrophone_request as hyd_request
from ooipy.hydrophone.basic import HydrophoneData
import datetime
import numpy as np


def test_get_acoustic_data():
    # 1. case: 100% data coverage
    start_time = datetime.datetime(2017,3,10,0,0,0)
    end_time = datetime.datetime(2017,3,10,0,5,0)
    node = 'PC01A'

    data = hyd_request.get_acoustic_data(start_time, end_time, node)

    assert type(data) == HydrophoneData
    assert type(data.data) == np.ndarray

    diff_start = abs((start_time - data.stats.starttime.datetime).microseconds)
    diff_end = abs((end_time - data.stats.endtime.datetime).microseconds)

    assert diff_start <= 100
    assert diff_end <= 100

    # 2. case: 0% data coverage
    start_time = datetime.datetime(2017,10,10,15,30,0)
    end_time = datetime.datetime(2017,10,10,15,35,0)
    node = 'LJ01C'

    data = hyd_request.get_acoustic_data(start_time, end_time, node, append=False)

    assert data is None

    # 3. case: partial data coverage (data available until 15:17:50)
    start_time = datetime.datetime(2017,10,10,15,15,0)
    end_time = datetime.datetime(2017,10,10,15,20,0)
    node = 'LJ01C'

    data = hyd_request.get_acoustic_data(start_time, end_time, node, append=False)

    assert type(data) == HydrophoneData
    assert type(data.data) == np.ndarray

    diff_start = abs((start_time - data.stats.starttime.datetime).microseconds)
    diff_end = abs((end_time - data.stats.endtime.datetime).microseconds)
    assert diff_start <= 100
    assert diff_end > 100

    # 4. case: 0% data coverage for entire day (directory does not exists)
    start_time = datetime.datetime(2019,11,1,0,0,0)
    end_time = datetime.datetime(2019,11,1,0,5,0)
    node = 'LJ01D'

    data = hyd_request.get_acoustic_data(start_time, end_time, node, append=False)

    assert data is None

def test_hydrophone_node_names():
    node_arr = ['Oregon_Shelf_Base_Seafloor', 'Oregon_Slope_Base_Seafloor',
                'Oregon_Slope_Base_Shallow', 'Axial_Base_Shallow',
                'Oregon_Offshore_Base_Seafloor', 'Axial_Base_Seafloor']
    node_id_arr = ['LJ01D', 'LJ01A', 'PC01A', 'PC03A', 'LJ01C', 'LJ03A']

    starttime = datetime.datetime(2017,3,20,0,0,0) # time of first sample
    endtime = datetime.datetime(2017,3,20,0,0,1) # time of last sample

    for item in node_arr:
        hyd_data = hyd_request.get_acoustic_data(starttime, endtime, node=item)
        assert hyd_data.stats.location in node_id_arr
            
    node_arr = ['Slope_Base', 'Southern_Hydrate', 'Axial_Base',
                'Central_Caldera', 'Eastern_Caldera']
    node_id_arr = ['HYSB1', 'HYS14', 'AXBA1', 'AXCC1', 'AXEC2']

    for item in node_arr:
        hyd_data = hyd_request.get_acoustic_data_LF(starttime, endtime, node=item)
        assert hyd_data.stats.location in node_id_arr