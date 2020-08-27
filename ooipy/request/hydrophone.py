# Import all dependancies
import numpy as np
import json
import os
import sys
sys.path.append("..") #TODO: remove this before publishing
from ooipy.hydrophone.basic import HydrophoneData
from matplotlib import pyplot as plt
from obspy import read,Stream, Trace
from obspy.core import UTCDateTime
import math
from matplotlib import mlab
from matplotlib.colors import Normalize
import requests
from lxml import html
from scipy import signal
from scipy import interpolate
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib
import datetime
import urllib
import time
import pandas as pd
from thredds_crawler.crawl import Crawl
import multiprocessing as mp
import pickle
import obspy
import scipy
import progressbar
from datetime import timedelta
import concurrent.futures




def _web_crawler_acoustic_data(day_str, node):
    '''
    get URLs for a specific day from OOI raw data server

    day_str (str): date for which URLs are requested; format: yyyy/mm/dd, e.g. 2016/07/15

    return ([str]): list of URLs, each URL refers to one data file. If no data is available for
        specified date, None is returned.
    '''

    if node == '/LJ01D': #LJ01D'  Oregon Shelf Base Seafloor
        array = '/CE02SHBP'
        instrument = '/11-HYDBBA106'
    if node == '/LJ01A': #LJ01A Oregon Slope Base Seafloore
        array = '/RS01SLBS'
        instrument = '/09-HYDBBA102'
    if node == '/PC01A': #Oregan Slope Base Shallow
        array = '/RS01SBPS'
        instrument = '/08-HYDBBA103'
    if node == '/PC03A': #Axial Base Shallow Profiler
        array = '/RS03AXPS'
        instrument = '/08-HYDBBA303'
    if node == '/LJ01C': #Oregon Offshore Base Seafloor
        array = '/CE04OSBP'
        instrument = '/11-HYDBBA105'
        
    mainurl = 'https://rawdata.oceanobservatories.org/files'+array+node+instrument+day_str
    try:
        mainurlpage =requests.get(mainurl, timeout=60)
    except:
        print('Timeout URL request')
        return None
    webpage = html.fromstring(mainurlpage.content)
    suburl = webpage.xpath('//a/@href') #specify that only request .mseed files

    FileNum = len(suburl)
    data_url_list = []
    for filename in suburl[6:FileNum]:
        data_url_list.append(str(mainurl + filename[2:]))
        
    return data_url_list


def get_acoustic_data(starttime, endtime, node, fmin=None, fmax=None, append=True, verbose=False, limit_seed_files=True, data_gap_mode=0):
    '''
    Get acoustic data for specific time frame and node:

    start_time (datetime.datetime): time of the first noise sample
    end_time (datetime.datetime): time of the last noise sample
    node (str): hydrophone
    fmin (float): lower cutoff frequency of hydrophone's bandpass filter. Default is None which results in no filtering.
    fmax (float): higher cutoff frequency of hydrophones bandpass filter. Default is None which results in no filtering.
    print_exceptions (bool): whether or not exeptions are printed in the terminal line
    verbose (bool) : Determines if information is printed to command line

    return (obspy.core.stream.Stream): obspy Stream object containing one Trace and date
        between start_time and end_time. Returns None if no data are available for specified time frame

    '''
    
    #data_gap = False
    
    # Save last mseed of previous day to data_url_list
    prev_day = starttime - timedelta(days=1)
    
    if append:
        data_url_list_prev_day = _web_crawler_acoustic_data(prev_day.strftime("/%Y/%m/%d/"), node)
        # keep only .mseed files
        del_list = []
        for i in range(len(data_url_list_prev_day)):
            url = data_url_list_prev_day[i].split('.')
            if url[len(url) - 1] != 'mseed':
                del_list.append(i)
        data_url_prev_day = np.delete(data_url_list_prev_day, del_list)
        data_url_prev_day = data_url_prev_day[-1]
    
    # get URL for first day
    day_start = UTCDateTime(starttime.year, starttime.month, starttime.day, 0, 0, 0)
    data_url_list = _web_crawler_acoustic_data(starttime.strftime("/%Y/%m/%d/"), node)
    if data_url_list == None:
        if verbose:
            print('No data available for specified day and node. Please change the day or use a differnt node')
        return None
    
    #increment day start by 1 day
    day_start = day_start + 24*3600
    
    #get all urls for each day untill endtime is reached
    while day_start < endtime:
        data_url_list.extend(_web_crawler_acoustic_data(starttime.strftime("/%Y/%m/%d/"), node))
        day_start = day_start + 24*3600
    
    if limit_seed_files:
        # if too many files for one day -> skip day (otherwise program takes too long to terminate)
        if len(data_url_list) > 1000:
            if verbose:
                print('Too many files for specified day. Cannot request data as web crawler cannot terminate.')
            return None
    
    # keep only .mseed files
    del_list = []
    for i in range(len(data_url_list)):
        url = data_url_list[i].split('.')
        if url[len(url) - 1] != 'mseed':
            del_list.append(i)
    data_url_list = np.delete(data_url_list, del_list)
    
    if append: data_url_list = np.insert(data_url_list,0,data_url_prev_day)
            
    st_all = None
    first_file=True
    # only acquire data for desired time

    for i in range(len(data_url_list)):
        # get UTC time of current and next item in URL list
        # extract start time from ith file
        utc_time_url_start = UTCDateTime(data_url_list[i].split('YDH')[1][1:].split('.mseed')[0])
        
        # this line assumes no gaps between current and next file
        if i != len(data_url_list) - 1:
            utc_time_url_stop = UTCDateTime(data_url_list[i+1].split('YDH')[1][1:].split('.mseed')[0])
        else: 
            utc_time_url_stop = UTCDateTime(data_url_list[i].split('YDH')[1][1:].split('.mseed')[0])
            utc_time_url_stop.hour = 23
            utc_time_url_stop.minute = 59
            utc_time_url_stop.second = 59
            utc_time_url_stop.microsecond = 999999
            
        # if current segment contains desired data, store data segment
        if (utc_time_url_start >= starttime and utc_time_url_start < endtime) or \
            (utc_time_url_stop >= starttime and utc_time_url_stop < endtime) or  \
            (utc_time_url_start <= starttime and utc_time_url_stop >= endtime):

            if (first_file) and (i != 0):
                first_file = False
                try:
                    if append:
                        # add one extra file on front end
                        st = read(data_url_list[i-1], apply_calib=True)
                        st += read(data_url_list[i], apply_calib=True)
                    else:
                        st = read(data_url_list[i], apply_calib=True)
                except:
                    if verbose:
                        print(f"Data Segment, {data_url_list[i-1]} or {data_url_list[i] } Broken")
                    #self.data = None
                    #self.data_available = False
                    #return None
            #normal operation (not first or last file)
            else:
                try:
                    st = read(data_url_list[i], apply_calib=True)
                except:
                    if verbose:
                        print(f"Data Segment, {data_url_list[i]} Broken")
                    #self.data = None
                    #self.data_available = False
                    #return None

            # Add st to acculation of all data st_all                
            if st_all == None: st_all = st
            else: 
                st_all += st
        # adds one more mseed file to st_ll             
        else:
            #Checks if last file has been downloaded within time period
            if first_file == False:
                first_file = True
                try:
                    if append: st = read(data_url_list[i], apply_calib=True)
                except:
                    if verbose:
                        print(f"Data Segment, {data_url_list[i]} Broken")
                    #self.data = None
                    #self.data_available = False
                    #return None                   

                if append: st_all += st
        
    # Merge all traces together
    if data_gap_mode == 0:
        st_all.merge(fill_value ='interpolate', method=1)
    # Returns Masked Array if there are data gaps
    elif data_gap_mode == 1:
        st_all.merge(method=1)
    else:
        if verbose: print('Invalid Data Gap Mode')
        return None
    # Slice data to desired window                
    st_all = st_all.slice(UTCDateTime(starttime), UTCDateTime(endtime))

    if isinstance(st_all[0].data, np.ma.core.MaskedArray):
        #data_gap = True
        if verbose: print('Data has Gaps') #Note this will only trip if masked array is returned
                                                            #interpolated is treated as if there is no gap
    if st_all != None:
        if len(st_all) == 0:
            if verbose:
                print('No data available for selected time frame.')
            return None
    
    #Filter Data
    try:
        if (fmin != None and fmax != None):
            st_all = st_all.filter("bandpass", freqmin=fmin, freqmax=fmax)
            if verbose: print('Signal Filtered')
        data = st_all[0]
        #data_available = True
        return HydrophoneData(data.data, data.stats, node)
    except:
        if st_all == None:
            if verbose:
                print('No data available for selected time frame.')
        else: 
            if verbose:
                print('Other exception')
        return None

#TODO revison necessary; function might be superfluous
'''
def get_acoustic_data_mp(starttime, endtime, node, n_process=None, fmin=None, fmax=None,
    append=True, verbose=False, limit_seed_files=True, data_gap_mode=0):


    Same as function get acoustic data but using multiprocessing.
  


    # entire time frame is divided into n_process parts of equal length 
    if n_process == None:
        N  = mp.cpu_count()
    else:
        N = n_process

    seconds_per_process = (endtime - starttime).total_seconds() / N

    get_data_list = [(starttime + datetime.timedelta(seconds=i * seconds_per_process),
        starttime + datetime.timedelta(seconds=(i + 1) * seconds_per_process),
        node, fmin, fmax) for i in range(N)]
    
    # create pool of processes require one part of the data in each process
    with mp.get_context("spawn").Pool(N) as p:
        try:
            data_list = p.starmap(self.get_acoustic_data, get_data_list)
        except:
            if verbose:
                print('Data cannot be requested.')
            #data_available = False
            return None
    
    #if all data is None, return None and set flags
    if (all(x==None for x in data_list)):
        if verbose:
            print('No data available for specified time and node')
        self.data = None
        self.data_available = False
        st_all = None
    
    
    #if only some of data is none, remove None entries
    if (None in data_list):
        if self.print_exceptions:
            print('Some mseed files missing or corrupted for time range')
        data_list = list(filter(None.__ne__, data_list))


    # merge data segments together
    st_all = data_list[0]
    if len(data_list) > 1:
        for d in data_list[1:]:
            st_all = st_all + d
    self._data_segmented = data_list
    st_all.merge(method=1)

    if isinstance(st_all[0].data, np.ma.core.MaskedArray):
        self.data_gap = True
        if self.print_exceptions: print('Data has gaps')

    # apply bandpass filter to st_all if desired
    if (self.fmin != None and self.fmax != None):
        st_all = st_all.filter("bandpass", freqmin=fmin, freqmax=fmax)
        print('data filtered')
    self.data = st_all[0]
    self.data_available = True

    self.starttime = starttime
    self.endtime = endtime
    return st_all
    '''

def __get_mseed_urls(day_str, node):
    import fsspec

    '''
    get URLs for a specific day from OOI raw data server

    day_str (str): date for which URLs are requested; format: yyyy/mm/dd, e.g. 2016/07/15

    return ([str]): list of URLs, each URL refers to one data file. If no data is available for
        specified date, None is returned.
    '''

    if node == '/LJ01D': #LJ01D'  Oregon Shelf Base Seafloor
        array = '/CE02SHBP'
        instrument = '/11-HYDBBA106'
    if node == '/LJ01A': #LJ01A Oregon Slope Base Seafloore
        array = '/RS01SLBS'
        instrument = '/09-HYDBBA102'
    if node == '/PC01A': #Oregan Slope Base Shallow
        array = '/RS01SBPS'
        instrument = '/08-HYDBBA103'
    if node == '/PC03A': #Axial Base Shallow Profiler
        array = '/RS03AXPS'
        instrument = '/08-HYDBBA303'
    if node == '/LJ01C': #Oregon Offshore Base Seafloor
        array = '/CE04OSBP'
        instrument = '/11-HYDBBA105'
        
    mainurl = 'https://rawdata.oceanobservatories.org/files'+array+node+instrument+day_str

    FS = fsspec.filesystem('http')
    data_url_list = sorted([f['name'] for f in FS.ls(mainurl) if f['type'] == 'file' and f['name'].endswith('.mseed')])
    
    return data_url_list

def get_acoustic_data_conc(starttime, endtime, node, fmin=None, fmax=None, max_workers=-1, append=True, verbose=False,
    data_gap_mode=0):
    '''
    Get acoustic data for specific time frame and node:

    start_time (datetime.datetime): time of the first noise sample
    end_time (datetime.datetime): time of the last noise sample
    node (str): hydrophone
    fmin (float): lower cutoff frequency of hydrophone's bandpass filter. Default is None which results in no filtering.
    fmax (float): higher cutoff frequency of hydrophones bandpass filter. Default is None which results in no filtering.
    print_exceptions (bool): whether or not exeptions are printed in the terminal line
    max_workers (int) : number of maximum workers for concurrent processing
    append (bool) : specifies if extra mseed files should be appended at beginning and end in case of boundary gaps in data
    verbose (bool) : specifies whether print statements should occur or not
    
    
    return (obspy.core.stream.Stream): obspy Stream object containing one Trace and date
        between start_time and end_time. Returns None if no data are available for specified time frame

    '''
    
    #data_gap = False

    if verbose: print('Fetching URLs...')

    # Save last mseed of previous day to data_url_list
    prev_day = starttime - timedelta(days=1)
    data_url_list_prev_day = __get_mseed_urls(prev_day.strftime("/%Y/%m/%d/"), node)
    data_url_prev_day = data_url_list_prev_day[-1]
    
    # get URL for first day
    day_start = UTCDateTime(starttime.year, starttime.month, starttime.day, 0, 0, 0)
    data_url_list = __get_mseed_urls(starttime.strftime("/%Y/%m/%d/"), node)
    
    if data_url_list == None:
        if verbose:
            print('No data available for specified day and node. Please change the day or use a differnt node')
        return None
    
    #increment day start by 1 day
    day_start = day_start + 24*3600
    
    #get all urls for each day until endtime is reached
    while day_start < endtime:
        data_url_list.extend(__get_mseed_urls(starttime.strftime("/%Y/%m/%d/"), node))
        day_start = day_start + 24*3600

    #get 1 more day of urls
    data_url_last_day_list= __get_mseed_urls(starttime.strftime("/%Y/%m/%d/"), node)
    data_url_last_day = data_url_last_day_list[0]

    #add 1 extra mseed file at beginning and end to handle gaps if append is true
    if append: data_url_list = np.insert(data_url_list,0,data_url_prev_day)
    if append: data_url_list = np.insert(data_url_list,-1,data_url_last_day)


    if verbose: print('Sorting valid URLs for Time Window...')
    #Create list of urls for specified time range
    
    valid_data_url_list = []
    first_file=True

    
    # Create List of mseed urls for valid time range
    for i in range(len(data_url_list)):

        # get UTC time of current and next item in URL list
        # extract start time from ith file
        utc_time_url_start = UTCDateTime(data_url_list[i].split('YDH')[1][1:].split('.mseed')[0])
        
        # this line assumes no gaps between current and next file
        if i != len(data_url_list) - 1:
            utc_time_url_stop = UTCDateTime(data_url_list[i+1].split('YDH')[1][1:].split('.mseed')[0])
        else: 
            utc_time_url_stop = UTCDateTime(data_url_list[i].split('YDH')[1][1:].split('.mseed')[0])
            utc_time_url_stop.hour = 23
            utc_time_url_stop.minute = 59
            utc_time_url_stop.second = 59
            utc_time_url_stop.microsecond = 999999
            
        # if current segment contains desired data, store data segment
        if (utc_time_url_start >= starttime and utc_time_url_start < endtime) or \
            (utc_time_url_stop >= starttime and utc_time_url_stop < endtime) or  \
            (utc_time_url_start <= starttime and utc_time_url_stop >= endtime):
            
            if append:
                if i == 0:
                    first_file = False
                    valid_data_url_list.append(data_url_list[i])

                elif (first_file):
                    first_file = False
                    valid_data_url_list = [data_url_list[i-1], data_url_list[i]]
                else:
                    valid_data_url_list.append(data_url_list[i])
            else:
                if i == 0:
                    first_file = False
                valid_data_url_list.append(data_url_list[i])

        # adds one more mseed file to st_ll             
        else:
            #Checks if last file has been downloaded within time period
            if first_file == False:
                first_file = True
                if append: valid_data_url_list.append(data_url_list[i])
                break
        

    if verbose: print('Downloading mseed files...')
    
    # Code Below from Landung Setiawan
    st_list = __map_concurrency(__read_mseed, valid_data_url_list)        #removed max workers argument
    st_all = None
    for st in st_list:
        if st:
            if not isinstance(st_all, Stream):
                st_all = st
            else:
                st_all += st
    
    ##Merge all traces together
    
    #Interpolation
    if data_gap_mode == 0:
        st_all.merge(fill_value ='interpolate', method=1)
    #Masked Array
    elif data_gap_mode == 1:
        st_all.merge(method=1)
    #Masked Array, Zero-Mean, Zero Fill
    elif data_gap_mode == 2:
        st_all.merge(method=1)
        st_all[0].data = st_all[0].data - np.mean(st_all[0].data)

        st_all[0].data.fill_value = 0
        st_all[0].data = np.ma.filled(st_all[0].data)
    
    
    else:
        if verbose: print('Invalid Data Gap Mode')
        return None
    # Slice data to desired window                
    st_all = st_all.slice(UTCDateTime(starttime), UTCDateTime(endtime))

    if isinstance(st_all[0].data, np.ma.core.MaskedArray):
        #data_gap = True
        if verbose: print('Data has Gaps') #Note this will only trip if masked array is returned
                                                            #interpolated is treated as if there is no gap
    if st_all != None:
        if len(st_all) == 0:
            if verbose:
                print('No data available for selected time frame.')
            return None
    
    #Filter Data
    try:
        if (fmin != None and fmax != None):
            st_all = st_all.filter("bandpass", freqmin=fmin, freqmax=fmax)
            if verbose: print('Signal Filtered')
        #return st_all[0]
        return HydrophoneData(st_all[0].data, st_all[0].stats, node)
    except:
        if st_all == None:
            if verbose:
                print('No data available for selected time frame.')
        else: 
            if verbose:
                print('Other exception')
        return None


def __map_concurrency(func, iterator, args=(), max_workers=-1):
    
    #automatically set max_workers to 2x(available cores)
    if max_workers == -1:
        max_workers = 2*mp.cpu_count()
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(func, i, *args): i for i in iterator}
        for future in concurrent.futures.as_completed(future_to_url):
            data = future.result()
            results.append(data)
    return results

def __read_mseed(url):
    fname = os.path.basename(url)
    #print(f"=== Reading: {fname} ===")
    try:
        st = read(url, apply_calib=True)
    except:
        print(f'Data Segment {url} Broken')

        return None
    if isinstance(st, Stream):

        return st
    else:
        print(f"Problem Reading {url}")

        return None