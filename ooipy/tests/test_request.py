'''
Unit tests for request module
'''

import ooipy.request.hydrophone_request as hyd_request
from ooipy.hydrophone.basic import HydrophoneData
import datetime
import numpy as np

def test_get_acoustic_data_archive():
    # 1. case: 100% data coverage
    start_time = datetime.datetime(2017,3,10,0,0,0)
    end_time = datetime.datetime(2017,3,10,0,5,0)
    node = '/PC01A'

    data = hyd_request.get_acoustic_data_archive(start_time, end_time, node)

    assert type(data) == HydrophoneData
    assert type(data.data) == np.ndarray

    diff_start = abs((start_time - data.stats.starttime.datetime).microseconds)
    diff_end = abs((end_time - data.stats.endtime.datetime).microseconds)
    assert diff_start <= 100
    assert diff_end <= 100

    # 2. case: 0% data coverage
    start_time = datetime.datetime(2017,10,10,15,30,0)
    end_time = datetime.datetime(2017,10,10,15,35,0)
    node = '/LJ01C'

    data = hyd_request.get_acoustic_data_archive(start_time, end_time, node)

    assert data is None

    # 3. case: partial data coverage (data available until 15:17:50)
    start_time = datetime.datetime(2017,10,10,15,15,0)
    end_time = datetime.datetime(2017,10,10,15,20,0)
    node = '/LJ01C'

    data = hyd_request.get_acoustic_data_archive(start_time, end_time, node)

    assert type(data) == HydrophoneData
    assert type(data.data) == np.ndarray

    diff_start = abs((start_time - data.stats.starttime.datetime).microseconds)
    diff_end = abs((end_time - data.stats.endtime.datetime).microseconds)
    assert diff_start <= 100
    assert diff_end > 100


def test_get_acoustic_data():
    # 1. case: 100% data coverage
    start_time = datetime.datetime(2017,3,10,0,0,0)
    end_time = datetime.datetime(2017,3,10,0,5,0)
    node = '/PC01A'

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
    node = '/LJ01C'

    data = hyd_request.get_acoustic_data(start_time, end_time, node, append=False)

    assert data is None

    # 3. case: partial data coverage (data available until 15:17:50)
    start_time = datetime.datetime(2017,10,10,15,15,0)
    end_time = datetime.datetime(2017,10,10,15,20,0)
    node = '/LJ01C'

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
    node = '/LJ01D'

    data = hyd_request.get_acoustic_data(start_time, end_time, node, append=False)

    assert data is None