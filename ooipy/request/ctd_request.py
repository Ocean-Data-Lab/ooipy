import requests
import multiprocessing as mp
import concurrent.futures
import datetime
import numpy as np
import ooipy.request.authentification
from ooipy.ctd.basic import CtdData

def get_ctd_data(beginDT, endDT, location, limit=10000):

    USERNAME, TOKEN =  ooipy.request.authentification.get_authentification()
    #Sensor Inventory
    DATA_API_BASE_URL = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/'

    dataraw = []
        
    # Oregon Shelf
    if location == 'shelf':
        url_list = ['CE02SHBP/LJ01D/06-CTDBPN106/streamed/ctdbp_no_sample?',
                    'CE02SHSP/SP001/08-CTDPFJ000/telemetered/ctdpf_j_cspp_instrument?',
                    'CE02SHSP/SP001/08-CTDPFJ000/recovered_cspp/ctdpf_j_cspp_instrument_recovered?',
                    'CE02SHSM/RID27/03-CTDBPC000/telemetered/ctdbp_cdef_dcl_instrument?']

    elif location == 'offshore':
        url_list = ['CE04OSPS/SF01B/2A-CTDPFA107/streamed/ctdpf_sbe43_sample?',
                    'CE04OSPS/PC01B/4A-CTDPFA109/streamed/ctdpf_optode_sample?',
                    'CE04OSSM/RID27/03-CTDBPC000/telemetered/ctdbp_cdef_dcl_instrument?',
                    'CE04OSPD/DP01B/01-CTDPFL105/recovered_inst/dpc_ctd_instrument_recovered?',
                    'CE04OSPD/DP01B/01-CTDPFL105/recovered_wfp/dpc_ctd_instrument_recovered?',
                    'CE04OSBP/LJ01C/06-CTDBPO108/streamed/ctdbp_no_sample?']

    for url in url_list:
        data_request_url = DATA_API_BASE_URL + url + \
            'beginDT=' + beginDT + '&endDT=' + endDT + '&limit=' + str(limit)
        
        r = requests.get(data_request_url, auth=(USERNAME, TOKEN))
        dataraw2 = r.json()
        if 'message' not in dataraw2:
            dataraw.extend(dataraw2)

    return CtdData(raw_data=dataraw)

def get_ctd_data_daily(datetime_day, location, limit=10000):
    # get CTD data for one hour
    year = datetime_day.year
    month = datetime_day.month
    day = datetime_day.day
        
    start_end_list = []
        
    for hour in range(24):
        start = datetime.datetime(year, month, day, hour, 0, 0).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'
        end = datetime.datetime(year, month, day, hour, 59, 59, 999).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'
        start_end_list.append((start, end))
        
    raw_data_arr = __map_concurrency(get_ctd_data_concurrent, start_end_list,
                                     {'location': location, 'limit': limit})
    
    raw_data_falttened = []
    for item in raw_data_arr:
        if item is None:
            continue
        else:
            raw_data_falttened.extend(item)

    return CtdData(raw_data=raw_data_falttened)

def __map_concurrency(func, iterator, args=(), max_workers=-1):
    # automatically set max_workers to 2x(available cores)
    if max_workers == -1:
        max_workers = 2 * mp.cpu_count()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) \
            as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(func, i, **args): i for i in iterator}
        for future in concurrent.futures.as_completed(future_to_url):
            data = future.result()
            results.append(data)
    return results

def get_ctd_data_concurrent(start_end_tuple, location, limit):
    start = start_end_tuple[0]
    end = start_end_tuple[1]
    
    rawdata = get_ctd_data(start, end, location=location, limit=limit)
    return rawdata