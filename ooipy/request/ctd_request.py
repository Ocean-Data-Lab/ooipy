import requests
import multiprocessing as mp
import concurrent.futures
import datetime
import numpy as np

def get_ctd_data(beginDT, endDT, location, username, token, limit=10000):
    USERNAME = username
    TOKEN =  token
    #Sensor Inventory
    DATA_API_BASE_URL = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/'
        
    # Oregon Shelf
    if location == 'shelf':
        # htdrophone(bottom) CTD
        data_request_url = DATA_API_BASE_URL+\
                    'CE02SHBP/'+\
                    'LJ01D/'+\
                    '06-CTDBPN106/'+\
                    'streamed/'+\
                    'ctdbp_no_sample'+'?'+\
                    'beginDT=' + beginDT + '&'+\
                    'endDT=' + endDT + '&'+\
                    'limit=' + str(limit)
        
        r = requests.get(data_request_url, auth=(USERNAME, TOKEN))
        dataraw = r.json()
        if 'message' in dataraw:
            dataraw = []
            
        # profiler - telemetered
        data_request_url = DATA_API_BASE_URL+\
                        'CE02SHSP/'+\
                        'SP001/'+\
                        '08-CTDPFJ000/'+\
                        'telemetered/'+\
                        'ctdpf_j_cspp_instrument'+'?'+\
                        'beginDT=' + beginDT + '&'+\
                        'endDT=' + endDT + '&'+\
                        'limit=' + str(limit)
        
        r = requests.get(data_request_url, auth=(USERNAME, TOKEN))
        dataraw2 = r.json()
        if 'message' not in dataraw2:
            dataraw.extend(dataraw2)
            
        # profiler - instrument recovered
        data_request_url = DATA_API_BASE_URL+\
                        'CE02SHSP/'+\
                        'SP001/'+\
                        '08-CTDPFJ000/'+\
                        'recovered_cspp/'+\
                        'ctdpf_j_cspp_instrument_recovered'+'?'+\
                        'beginDT=' + beginDT + '&'+\
                        'endDT=' + endDT + '&'+\
                        'limit=' + str(limit)
        
        r = requests.get(data_request_url, auth=(USERNAME, TOKEN))
        dataraw2 = r.json()
        if 'message' not in dataraw2:
            dataraw.extend(dataraw2)
            
        # surface CTD
        data_request_url = DATA_API_BASE_URL+\
                        'CE02SHSM/'+\
                        'RID27/'+\
                        '03-CTDBPC000/'+\
                        'telemetered/'+\
                        'ctdbp_cdef_dcl_instrument'+'?'+\
                        'beginDT=' + beginDT + '&'+\
                        'endDT=' + endDT + '&'+\
                        'limit=' + str(limit)
        
        r = requests.get(data_request_url, auth=(USERNAME, TOKEN))
        dataraw2 = r.json()
        if 'message' not in dataraw2:
            dataraw.extend(dataraw2)
     
    # Oregon Offshore
    if location == 'offshore':
        # Shallow profiler (0 - 200m)
        data_request_url = DATA_API_BASE_URL+\
                    'CE04OSPS/'+\
                    'SF01B/'+\
                    '2A-CTDPFA107/'+\
                    'streamed/'+\
                    'ctdpf_sbe43_sample'+'?'+\
                    'beginDT=' + beginDT + '&'+\
                    'endDT=' + endDT + '&'+\
                    'limit=' + str(limit)

        r = requests.get(data_request_url, auth=(USERNAME, TOKEN))
        dataraw = r.json()
        if 'message' in dataraw:
            dataraw = []
            
        # shallow profiler 200m CTD
        data_request_url = DATA_API_BASE_URL+\
                        'CE04OSPS/'+\
                        'PC01B/'+\
                        '4A-CTDPFA109/'+\
                        'streamed/'+\
                        'ctdpf_optode_sample'+'?'+\
                        'beginDT=' + beginDT + '&'+\
                        'endDT=' + endDT + '&'+\
                        'limit=' + str(limit)
        
        r = requests.get(data_request_url, auth=(USERNAME, TOKEN))
        dataraw2 = r.json()
        if 'message' not in dataraw2:
            dataraw.extend(dataraw2)
        
        # surface CTD
        data_request_url = DATA_API_BASE_URL+\
                        'CE04OSSM/'+\
                        'RID27/'+\
                        '03-CTDBPC000/'+\
                        'telemetered/'+\
                        'ctdbp_cdef_dcl_instrument'+'?'+\
                        'beginDT=' + beginDT + '&'+\
                        'endDT=' + endDT + '&'+\
                        'limit=' + str(limit)
        
        r = requests.get(data_request_url, auth=(USERNAME, TOKEN))
        dataraw2 = r.json()
        if 'message' not in dataraw2:
            dataraw.extend(dataraw2)


        # deep profiler (200 - 500m)
        # recovered instrument data
        data_request_url = DATA_API_BASE_URL+\
                    'CE04OSPD/'+\
                    'DP01B/'+\
                    '01-CTDPFL105/'+\
                    'recovered_inst/'+\
                    'dpc_ctd_instrument_recovered'+'?'+\
                    'beginDT=' + beginDT + '&'+\
                    'endDT=' + endDT + '&'+\
                    'limit=' + str(limit)

        r = requests.get(data_request_url, auth=(USERNAME, TOKEN))
        dataraw2 = r.json()
        if 'message' not in dataraw2:
            dataraw.extend(dataraw2)

        # recovered wtp data
        data_request_url = DATA_API_BASE_URL+\
                    'CE04OSPD/'+\
                    'DP01B/'+\
                    '01-CTDPFL105/'+\
                    'recovered_wfp/'+\
                    'dpc_ctd_instrument_recovered'+'?'+\
                    'beginDT=' + beginDT + '&'+\
                    'endDT=' + endDT + '&'+\
                    'limit=' + str(limit)

        r = requests.get(data_request_url, auth=(USERNAME, TOKEN))
        dataraw2 = r.json()
        if 'message' not in dataraw2:
            dataraw.extend(dataraw2)

        # hydrophone (bottom) CTD
        data_request_url = DATA_API_BASE_URL+\
                    'CE04OSBP/'+\
                    'LJ01C/'+\
                    '06-CTDBPO108/'+\
                    'streamed/'+\
                    'ctdbp_no_sample'+'?'+\
                    'beginDT=' + beginDT + '&'+\
                    'endDT=' + endDT + '&'+\
                    'limit=' + str(limit)

        r = requests.get(data_request_url, auth=(USERNAME, TOKEN))
        dataraw2 = r.json()
        if 'message' not in dataraw2:
            dataraw.extend(dataraw2)

    return dataraw

    
def ntp_seconds_to_datetime(ntp_seconds):
    ntp_epoch = datetime.datetime(1900, 1, 1)
    unix_epoch = datetime.datetime(1970, 1, 1)
    ntp_delta = (unix_epoch - ntp_epoch).total_seconds()
    return datetime.datetime.utcfromtimestamp(ntp_seconds - ntp_delta).replace(microsecond=0)

def get_parameter_from_rawdata(rawdata, param):
    param_arr = []
    for item in rawdata:
        if param == 'temperature':
            if 'seawater_temperature' in item:
                param_arr.append(item['seawater_temperature'])
            elif 'temperature' in item:
                param_arr.append(item['temperature'])
            else:
                param_arr.append(item['temp'])
        if param == 'pressure':
            if 'ctdbp_no_seawater_pressure' in item:
                param_arr.append(item['ctdbp_no_seawater_pressure'])
            elif 'seawater_pressure' in item:
                param_arr.append(item['seawater_pressure'])
            else:
                param_arr.append(item['pressure'])
                
        if param == 'salinity':
            if 'practical_salinity' in item:
                param_arr.append(item['practical_salinity'])
            else:
                param_arr.append(item['salinity'])
            

    return param_arr

def get_time_from_rawdata(rawdata):
    time_arr = []
    for item in rawdata:
        time_arr.append(ntp_seconds_to_datetime(item['pk']['time']))

    return time_arr

def compute_ssp_9(depth, temp, sal):
    c = 1448.96 + 4.591 * temp - 5.304e-2 * temp**2 + 2.374e-4 * temp**3 \
        + 1.340 * (sal - 35) + 1.63e-2 * depth + 1.675e-7 * depth**2 \
        - 1.025e-2 * temp * (sal - 35) - 7.139e-13 * temp * depth**3
    return c

# code from John
def compute_ssp_long(press, temp, sal):
    
    press_MPa = 0.01 * press

    C00 = 1402.388
    A02 = 7.166E-5
    C01 = 5.03830
    A03 = 2.008E-6
    C02 = -5.81090E-2
    A04 = -3.21E-8
    C03 = 3.3432E-4
    A10 = 9.4742E-5
    C04 = -1.47797E-6
    A11 = -1.2583E-5
    C05 = 3.1419E-9
    A12 = -6.4928E-8
    C10 = 0.153563
    A13 = 1.0515E-8
    C11 = 6.8999E-4
    A14 = -2.0142E-10
    C12 = -8.1829E-6
    A20 = -3.9064E-7
    C13 = 1.3632E-7
    A21 = 9.1061E-9
    C14 = -6.1260E-10
    A22 = -1.6009E-10
    C20 = 3.1260E-5
    A23 = 7.994E-12

    C21 = -1.7111E-6
    A30 = 1.100E-10
    C22 = 2.5986E-8
    A31 = 6.651E-12
    C23 = -2.5353E-10
    A32 = -3.391E-13
    C24 = 1.0415E-12
    B00 = -1.922E-2
    C30 = -9.7729E-9
    B01 = -4.42E-5
    C31 = 3.8513E-10
    B10 = 7.3637E-5
    C32 = -2.3654E-12
    B11 = 1.7950E-7
    A00 = 1.389
    D00 = 1.727E-3
    A01 = -1.262E-2
    D10 = -7.9836E-6 

    T = 3
    S = 1
    P = 700
    T = temp
    S = sal
    P = press_MPa*10

    D = D00 + D10*P 
    B = B00 + B01*T + (B10 + B11*T)*P 
    A = (A00 + A01*T + A02*T**2 + A03*T**3 + A04*T**4) + (A10 + A11*T + A12*T**2 + A13*T**3 + A14*T**4)*P + (A20 + A21*T + A22*T**2 + A23*T**3)*P**2 + (A30 + A31*T + A32*T**2)*P**3
    Cw = (C00 + C01*T + C02*T**2 + C03*T**3 + C04*T**4 + C05*T**5) + (C10 + C11*T + C12*T**2 + C13*T**3 + C14*T**4)*P + (C20 +C21*T +C22*T**2 + C23*T**3 + C24*T**4)*P**2 + (C30 + C31*T + C32*T**2)*P**3

    # Calculate Speed of Sound
    c = Cw + A*S + B*S**(3/2) + D*S**2

    # Calculate Depth from pressure
    lat = 44.52757 #deg

    # Calculate gravity constant for given latitude
    g_phi = 9.780319*(1 + 5.2788E-3*(np.sin(np.deg2rad(lat))**2) + 2.36E-5*(np.sin(np.deg2rad(lat))**4))

    # Calculate Depth for Pressure array
    depth_m = (9.72659e2*press_MPa - 2.512e-1*press_MPa**2 + 2.279e-4*press_MPa**3 - 1.82e-7*press_MPa**4)/(g_phi + 1.092e-4*press_MPa)

    return depth_m, c


def compute_ssp_24h_mp(datetime_day, location, depth_range, limit=10000):
    # get CTD data for one hour
    year = datetime_day.year
    month = datetime_day.month
    day = datetime_day.day
    
    ssp_dct = {}
    for k in range(depth_range):
        ssp_dct[str(k)] = {'ssp': [], 'd': [], 'temp': [], 'sal': [], 'pres': []}
        
    start_end_list = []
        
    for hour in range(24):
        start = datetime.datetime(year, month, day, hour, 0, 0).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'
        end = datetime.datetime(year, month, day, hour, 59, 59, 999).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'
        start_end_list.append((start, end))
        
    raw_data_arr = __map_concurrency(get_ctd_data_concurrent, start_end_list, {'location': location,
                                                                                           'limit': limit})
    
    for item in raw_data_arr:
        
        if item is None:
            continue

        temp = get_parameter_from_rawdata(item, 'temperature')
        press = get_parameter_from_rawdata(item, 'pressure')
        sal = get_parameter_from_rawdata(item, 'salinity')
        #depth = get_parameter_from_rawdata(rawdata, 'depth')
        #time = get_time_from_rawdata(rawdata)

        depth, ssp = compute_ssp_long(np.array(press), np.array(temp), np.array(sal))

        for d, ss, t, s, p in zip(depth, ssp, temp, sal, press):
            if str(int(d)) in ssp_dct:
                ssp_dct[str(int(d))]['d'].append(d)
                ssp_dct[str(int(d))]['ssp'].append(ss)
                ssp_dct[str(int(d))]['temp'].append(t)
                ssp_dct[str(int(d))]['sal'].append(s)
                ssp_dct[str(int(d))]['pres'].append(p)
    
    ssp_mean = []
    depth_mean = []
    ssp_var = []
    depth_var = []
    
    temp_mean = []
    sal_mean = []
    temp_var = []
    sal_var = []
    pres_mean = []
    pres_var = []
    
    n_samp = []
    
    for key in ssp_dct:
        ssp_mean.append(np.mean(ssp_dct[key]['ssp']))
        depth_mean.append(np.mean(ssp_dct[key]['d']))
        temp_mean.append(np.mean(ssp_dct[key]['temp']))
        sal_mean.append(np.mean(ssp_dct[key]['sal']))
        pres_mean.append(np.mean(ssp_dct[key]['pres']))
        
        ssp_var.append(np.var(ssp_dct[key]['ssp']))
        depth_var.append(np.var(ssp_dct[key]['d']))
        temp_var.append(np.var(ssp_dct[key]['temp']))
        sal_var.append(np.var(ssp_dct[key]['sal']))
        pres_var.append(np.var(ssp_dct[key]['pres']))
        
        n_samp.append(len(ssp_dct[key]['d']))
        
    idx = np.argsort(depth_mean)
    depth_mean = np.array(depth_mean)[idx]
    ssp_mean = np.array(ssp_mean)[idx]
    temp_mean = np.array(temp_mean)[idx]
    sal_mean = np.array(sal_mean)[idx]
    pres_mean = np.array(pres_mean)[idx]
    
    depth_var = np.array(depth_var)[idx]
    ssp_var = np.array(ssp_var)[idx]
    temp_var = np.array(temp_var)[idx]
    sal_var = np.array(sal_var)[idx]
    pres_var = np.array(pres_var)[idx]
    n_samp = np.array(n_samp)[idx]
    
    parameter_dct = {'depth': {'mean': depth_mean, 'var': depth_var},
                     'sound_speed': {'mean': ssp_mean, 'var': ssp_var},
                     'temperature': {'mean': temp_mean, 'var': temp_var},
                     'salinity': {'mean': sal_mean, 'var': sal_var},
                     'pressure': {'mean': pres_mean, 'var': pres_var},
                     'number_samples': n_samp}
        
    return parameter_dct, ssp_dct

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