# Import all dependancies
import numpy as np
import json
import os
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
import sys
from thredds_crawler.crawl import Crawl
import multiprocessing as mp
import pickle
import obspy
import scipy
import progressbar
from datetime import timedelta


class OOIHydrophoneData:

    def __init__(self, starttime=None, endtime=None, node=None, fmin=None,
        fmax=None, print_exceptions=None, limit_seed_files=True):
        
        ''' 
        Initialize Class OOIHydrophoneData

        Attributes
        ----------
        starttime : datetime.datetime
            indicates start time for acquiring data
        endtime : datetime.datetime
            indicates end time for acquiring data
        node : str
            indicates hydrophone location as listed below
            ____________________________________________
            |Node Name |        Hydrophone Name        |
            |__________|_______________________________|
            |'/LJ01D'  | Oregon Shelf Base Seafloor    |
            |__________|_______________________________|
            |'/LJ01A   | Oregon Slope Base Seafloore   |
            |__________|_______________________________|
            |'/PC01A'  | Oregan Slope Base Shallow     |
            |__________|_______________________________|
            |'/PC03A'  | Axial Base Shallow Profiler   |
            |__________|_______________________________|
            |'/LJ01C'  | Oregon Offshore Base Seafloor |
            |__________|_______________________________|
       
        fmin : int or float
            indicates minimum frequency in bandpass filter. Default value is None, which results in unfiltered signal
        fmax : int or float
            indicates maximum frequency in bandpass filter. Default value is None, which results in unfiltered signal
        print_exceptions : bool
            indicates whether or not to print errors
        data_available : bool
            indicates if data is available for specified start-/end time and node
        limit_seed_files : bool
            indicates if number of seed files per data retrieval should be limited
        data : obspy.core.trace.Trace
            object containg the acoustic data between start and end time for the specified node
        spectrogram : Spectrogram
            spectrogram of data.data. Spectral level, time, and frequency bins can be acccessed by
            spectrogram.values, spectrogram.time, and spectrogram.freq
        psd : Psd
            power spectral density estimate of data.data. Spectral level and frequency bins can be
            accessed by psd.values and psd.freq
        psd_list : list of Psd
            the data object is divided into N segments and for each segment a separate power spectral
            desity estimate is computed and stored in psd_list. The number of segments N is determined
            by the split parameter in compute_psd_welch_mp
        data_gap : bool
            specifies if data retrieved has gaps in it
        data_gap_mode : int
            specifies how to handle gapped data
            mode 0 (default) - linearly interpolates between gap
            mode 1 - returns numpy masked array

        Private Attributes
        ------------------
        _data_segmented : list of obspy.core.stream.Stream
            data segmented in multiple sub-streams. Used to reduce computation time in
            compute_spectrogram_mp if split = None.

        Methods
        -------
        get_acoustic_data(starttime, endtime, node, fmin=20.0, fmax=30000.0)
            Fetches hydrophone data form OOI server
        get_acoustic_data_mp(starttime, endtime, node, n_process=None, fmin=20.0, fmax=30000.0)
            Same as get_acoustic_data but using multiprocessing to parallelize data fetching
        compute_spectrogram(win='hann', L=4096, avg_time=None, overlap=0.5)
            Computes spectrogram for data attribute.
        compute_spectrogram_mp(split=None, n_process=None, win='hann', L=4096, avg_time=None, overlap=0.5)
            Same as compute_spectrogram but using multiprocessing to parallelize computations.
        compute_psd_welch(win='hann', L=4096, overlap=0.5, avg_method='median', interpolate=None, scale='log')
            Compute power spectral density estimate for data attribute.
        compute_psd_welch_mp(split, n_process=None, win='hann', L=4096, overlap=0.5, avg_method='median', interpolate=None, scale='log')
            Same as compute_psd_welch but using multiprocessing to parallelize computations.

        Private Methods
        ---------------
        __web_crawler_noise(day_str)
            Fetches URLs from OOI raw data server for specific day.
        _freq_dependent_sensitivity_correct(N)
            TODO: applies frequency dependent sensitivity correction to hydrohone data 
        '''
        self.starttime = starttime
        self.endtime = endtime
        self.node = node
        self.fmin = fmin
        self.fmax = fmax
        self.print_exceptions = print_exceptions
        self.data_available = None
        self.limit_seed_files = limit_seed_files
        self.data_gap = False
        self.data_gap_mode = 0

        if self.starttime == None or self.endtime == None or self.node == None:
            self.data = None
        else:
            self.get_acoustic_data(self.starttime, self.endtime, self.node, fmin=self.fmin, fmax=self.fmax)

        self.spectrogram = None
        self.psd = None
        self.psd_list = None


    def __web_crawler_noise(self, day_str):
        '''
        get URLs for a specific day from OOI raw data server

        day_str (str): date for which URLs are requested; format: yyyy/mm/dd, e.g. 2016/07/15

        return ([str]): list of URLs, each URL refers to one data file. If no data is available for
            specified date, None is returned.
        '''

        if self.node == '/LJ01D': #LJ01D'  Oregon Shelf Base Seafloor
            array = '/CE02SHBP'
            instrument = '/11-HYDBBA106'
        if self.node == '/LJ01A': #LJ01A Oregon Slope Base Seafloore
            array = '/RS01SLBS'
            instrument = '/09-HYDBBA102'
        if self.node == '/PC01A': #Oregan Slope Base Shallow
            array = '/RS01SBPS'
            instrument = '/08-HYDBBA103'
        if self.node == '/PC03A': #Axial Base Shallow Profiler
            array = '/RS03AXPS'
            instrument = '/08-HYDBBA303'
        if self.node == '/LJ01C': #Oregon Offshore Base Seafloor
            array = '/CE04OSBP'
            instrument = '/11-HYDBBA105'
            
        mainurl = 'https://rawdata.oceanobservatories.org/files'+array+self.node+instrument+day_str
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

    def _freq_dependent_sensitivity_correct(self, N):
        #TODO
        '''
        Apply a frequency dependent sensitivity correction to the acoustic data (in frequency domain).

        N (int): length of the data segment

        return (np.array): array with correction coefficient for every frequency 
        '''
        f_calib = [0, 13500, 27100, 40600, 54100]
        sens_calib = [169, 169.4, 168.1, 169.7, 171.5]
        sens_interpolated = interpolate.InterpolatedUnivariateSpline(f_calib, sens_calib)
        f = np.linspace(0, 32000, N)
        
        return sens_interpolated(f)


    def get_acoustic_data(self, starttime, endtime, node, fmin=None, fmax=None):
        '''
        Get acoustic data for specific time frame and node:

        start_time (datetime.datetime): time of the first noise sample
        end_time (datetime.datetime): time of the last noise sample
        node (str): hydrophone
        fmin (float): lower cutoff frequency of hydrophone's bandpass filter. Default is None which results in no filtering.
        fmax (float): higher cutoff frequency of hydrophones bandpass filter. Default is None which results in no filtering.
        print_exceptions (bool): whether or not exeptions are printed in the terminal line

        return (obspy.core.stream.Stream): obspy Stream object containing one Trace and date
            between start_time and end_time. Returns None if no data are available for specified time frame

        '''
   
        self.starttime = starttime
        self.endtime = endtime
        self.node = node
        self.fmin = fmin
        self.fmax = fmax
        
        self.data_gap = False
        
        # Save last mseed of previous day to data_url_list
        prev_day = self.starttime - timedelta(days=1)
        data_url_list_prev_day = self.__web_crawler_noise(prev_day.strftime("/%Y/%m/%d/"))
        # keep only .mseed files
        del_list = []
        for i in range(len(data_url_list_prev_day)):
            url = data_url_list_prev_day[i].split('.')
            if url[len(url) - 1] != 'mseed':
                del_list.append(i)
        data_url_prev_day = np.delete(data_url_list_prev_day, del_list)
        data_url_prev_day = data_url_prev_day[-1]
       
        # get URL for first day
        day_start = UTCDateTime(self.starttime.year, self.starttime.month, self.starttime.day, 0, 0, 0)
        data_url_list = self.__web_crawler_noise(self.starttime.strftime("/%Y/%m/%d/"))
        if data_url_list == None:
            if self.print_exceptions:
                print('No data available for specified day and node. Please change the day or use a differnt node')
            self.data = None
            self.data_available = False
            return None
        
        #increment day start by 1 day
        day_start = day_start + 24*3600
        
        #get all urls for each day untill endtime is reached
        while day_start < self.endtime:
            data_url_list.extend(self.__web_crawler_noise(self.starttime.strftime("/%Y/%m/%d/")))
            day_start = day_start + 24*3600
        
        if self.limit_seed_files:
            # if too many files for one day -> skip day (otherwise program takes too long to terminate)
            if len(data_url_list) > 1000:
                if self.print_exceptions:
                    print('Too many files for specified day. Cannot request data as web crawler cannot terminate.')
                self.data = None
                self.data_available = False
                return None
        
        # keep only .mseed files
        del_list = []
        for i in range(len(data_url_list)):
            url = data_url_list[i].split('.')
            if url[len(url) - 1] != 'mseed':
                del_list.append(i)
        data_url_list = np.delete(data_url_list, del_list)
        data_url_list = np.insert(data_url_list,0,data_url_prev_day)

        self.data_url_list = data_url_list
                
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
            if (utc_time_url_start >= self.starttime and utc_time_url_start < self.endtime) or \
                (utc_time_url_stop >= self.starttime and utc_time_url_stop < self.endtime) or  \
                (utc_time_url_start <= self.starttime and utc_time_url_stop >= self.endtime):

                if (first_file) and (i != 0):
                    first_file = False
                    try:
                        # add one extra file on front end
                        st = read(data_url_list[i-1], apply_calib=True)
                        st += read(data_url_list[i], apply_calib=True)
                    except:
                        if self.print_exceptions:
                            print(f"Data Segment, {data_url_list[i-1]} or {data_url_list[i] } Broken")
                        #self.data = None
                        #self.data_available = False
                        #return None
                #normal operation (not first or last file)
                else:
                    try:
                        st = read(data_url_list[i], apply_calib=True)
                    except:
                        if self.print_exceptions:
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
                        st = read(data_url_list[i], apply_calib=True)
                    except:
                        if self.print_exceptions:
                            print(f"Data Segment, {data_url_list[i]} Broken")
                        #self.data = None
                        #self.data_available = False
                        #return None                   
    
                    st_all += st
            
        # Merge all traces together
        if self.data_gap_mode == 0:
            st_all.merge(fill_value ='interpolate', method=1)
        # Returns Masked Array if there are data gaps
        elif self.data_gap_mode == 1:
            st_all.merge(method=1)
        else:
            if self.print_exceptions: print('Invalid Data Gap Mode')
            return None
        # Slice data to desired window                
        st_all = st_all.slice(UTCDateTime(self.starttime), UTCDateTime(self.endtime))

        if isinstance(st_all[0].data, np.ma.core.MaskedArray):
            self.data_gap = True
            if self.print_exceptions: print('Data has Gaps') #Note this will only trip if masked array is returned
                                                             #interpolated is treated as if there is no gap
        if st_all != None:
            if len(st_all) == 0:
                if self.print_exceptions:
                    print('No data available for selected time frame.')
                self.data = None
                self.data_available = False
                return None
        
        #Filter Data
        try:
            if (self.fmin != None and self.fmax != None):
                st_all = st_all.filter("bandpass", freqmin=fmin, freqmax=fmax)
                if self.print_exceptions: print('Signal Filtered')
            self.data = st_all[0]
            self.data_available = True
            return st_all
        except:
            if st_all == None:
                if self.print_exceptions:
                    print('No data available for selected time frame.')
            else: 
                if self.print_exceptions:
                    print('Other exception')
            self.data = None
            self.data_available = False
            return None

    def get_acoustic_data_mp(self, starttime, endtime, node, n_process=None, fmin=None, fmax=None):
        '''
        Same as function get acoustic data but using multiprocessing.
        '''
        self.node = node
        self.fmin = fmin
        self.fmax = fmax
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
                self.data_list = data_list
            except:
                if self.print_exceptions:
                    print('Data cannot be requested.')
                self.data = None
                self.data_available = False
                self.starttime = starttime
                self.endtime = endtime
                return self.data
        
        #if all data is None, return None and set flags
        if (all(x==None for x in data_list)):
            if self.print_exceptions:
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

    def compute_spectrogram(self, win='hann', L=4096, avg_time=None, overlap=0.5):
        '''
        Compute spectrogram of acoustic signal. For each time step of the spectrogram either a modified periodogram (avg_time=None)
        or a power spectral density estimate using Welch's method is computed.

        win (str): window function used to taper the data. Default is Hann-window. See scipy.signal.get_window for a list of
            possible window functions
        L (int): length of each data block for computing the FFT
        avg_time (float): time in seconds that is covered in one time step of the spectrogram. Default value is None and one
            time step covers L samples. If signal covers a long time period it is recommended to use a higher value for avg_time
            to avoid memory overflows and facilitate visualization.
        overlap (float): percentage of overlap between adjecent blocks if Welch's method is used. Parameter is ignored if
            avg_time is None.

        return ([datetime.datetime], [float], [float]): tuple including time, frequency, and spectral level.
            If no noise date is available, function returns three empty numpy arrays
        '''
        specgram = []
        time = []
            
        if self.data == None:
            if self.print_exceptions:
                print('Data object is empty. Spectrogram cannot be computed')
            self.spectrogram = None
            return self

        # sampling frequency
        fs = self.data.stats.sampling_rate

        # number of time steps
        if avg_time == None:
            nbins = int(len(self.data.data) / L)
        else:
            nbins = int(np.ceil(len(self.data.data) / (avg_time * fs)))

        # compute spectrogram. For avg_time=None (periodogram for each time step), the last data samples are ignored if 
        # len(noise[0].data) != k * L
        if avg_time == None:
            for n in range(nbins - 1):
                f, Pxx = signal.periodogram(x = self.data.data[n*L:(n+1)*L], fs=fs, window=win)
                if len(Pxx) != int(L/2)+1:
                    if self.print_exceptions:
                        print('Error while computing periodogram for segment', n)
                    self.spectrogram = None
                    return self
                else:
                    Pxx = 10*np.log10(Pxx * np.power(10, self._freq_dependent_sensitivity_correct(int(L/2 + 1))/10))-128.9
                    specgram.append(Pxx)
                    time.append(self.starttime + datetime.timedelta(seconds=n*L / fs))

        else:
            for n in range(nbins - 1):
                f, Pxx = signal.welch(x = self.data.data[n*int(fs*avg_time):(n+1)*int(fs*avg_time)],
                    fs=fs, window=win, nperseg=L, noverlap = int(L * overlap), nfft=L, average='median')
                if len(Pxx) != int(L/2)+1:
                    if self.print_exceptions:
                        print('Error while computing Welch estimate for segment', n)
                    self.spectrogram = None
                    return self
                else:
                    Pxx = 10*np.log10(Pxx * np.power(10, self._freq_dependent_sensitivity_correct(int(L/2 + 1))/10))-128.9
                    specgram.append(Pxx)
                    time.append(self.starttime + datetime.timedelta(seconds=n*avg_time))

            # compute PSD for residual segment if segment has more than L samples
            if len(self.data.data[int((nbins - 1) * fs * avg_time):]) >= L:
                f, Pxx = signal.welch(x = self.data.data[int((nbins - 1) * fs * avg_time):],
                    fs=fs, window=win, nperseg=L, noverlap = int(L * overlap), nfft=L, average='median')
                if len(Pxx) != int(L/2)+1:
                    if self.print_exceptions:
                        print('Error while computing Welch estimate residual segment')
                    self.spectrogram = None
                    return self
                else:
                    Pxx = 10*np.log10(Pxx * np.power(10, self._freq_dependent_sensitivity_correct(int(L/2 + 1))/10))-128.9
                    specgram.append(Pxx)
                    time.append(self.starttime + datetime.timedelta(seconds=(nbins-1)*avg_time))
    
        if len(time) == 0:
            if self.print_exceptions:
                print('Spectrogram does not contain any data')
            self.spectrogram = None
            return self
        else:
            self.spectrogram = Spectrogram(np.array(time), np.array(f), np.array(specgram))
            return self

    def compute_spectrogram_mp(self, split=None, n_process=None, win='hann', L=4096, avg_time=None, overlap=0.5):
        '''
        Same as function compute_spectrogram but using multiprocessing. This function is intended to
        be used when analyzing large data sets.

        split (float or [datetime.datetime]): time period between start_time and end_time is split into parts of length
            split seconds (if float). The last segment can be shorter than split seconds. Alternatively split can be set as
            an list with N start-end time tuples where N is the number of segments. 
        n_process (int): number of processes in the pool. Default (None) means that n_process is equal to the number
            of CPU cores.
        win (str): window function used to taper the data. Default is Hann-window. See scipy.signal.get_window for a list of
            possible window functions
        L (int): length of each data block for computing the FFT
        avg_time (float): time in seconds that is covered in one time step of the spectrogram. Default value is None and one
            time step covers L samples. If signal covers a long time period it is recommended to use a higher value for avg_time
            to avoid memory overflows and facilitate visualization.
        overlap (float): percentage of overlap between adjecent blocks if Welch's method is used. Parameter is ignored if
            avg_time is None.

        return ([datetime.datetime], [float], [float]): tuple including time, frequency, and spectral level.
            If no noise date is available, function returns three empty numpy arrays
        '''

        # create array with N start and end time values
        if n_process == None:
            N  = mp.cpu_count()
        else:
            N = n_process

        ooi_hyd_data_list = []
        # processa data using same segmentation as for get_acoustic_data_mp. This can save time compared to
        # doing the segmentation from scratch
        if split == None:
            for i in range(N):
                tmp_obj = OOIHydrophoneData(starttime=self._data_segmented[i][0].stats.starttime.datetime,
                    endtime=self._data_segmented[i][0].stats.endtime.datetime)
                tmp_obj.data = self._data_segmented[i][0]
                ooi_hyd_data_list.append((tmp_obj, win, L, avg_time, overlap))
        # do segmentation from scratch
        else:
            n_seg = int(np.ceil((self.endtime - self.starttime).total_seconds() / split))
            seconds_per_process = (self.endtime - self.starttime).total_seconds() / n_seg
            for k in range(n_seg - 1):
                starttime = self.starttime + datetime.timedelta(seconds=k * seconds_per_process)
                endtime = self.starttime + datetime.timedelta(seconds=(k+1) * seconds_per_process)
                tmp_obj = OOIHydrophoneData(starttime=starttime, endtime=endtime)
                tmp_obj.data = self.data.slice(starttime=starttime, endtime=endtime)
                ooi_hyd_data_list.append((tmp_obj, win, L, avg_time, overlap))


            starttime = self.starttime + datetime.timedelta(seconds=(n_seg - 1) * seconds_per_process)
            tmp_obj = OOIHydrophoneData(starttime=starttime, endtime=self.endtime)
            tmp_obj.data = self.data.slice(starttime=starttime, endtime=self.endtime)
            ooi_hyd_data_list.append((tmp_obj, win, L, avg_time, overlap))

        with mp.get_context("spawn").Pool(n_process) as p:
            try:
                specgram_list = p.starmap(_spectrogram_mp_helper, ooi_hyd_data_list)
                ## concatenate all small spectrograms to obtain final spectrogram
                specgram = []
                time_specgram = []
                for i in range(len(specgram_list)):
                    time_specgram.extend(specgram_list[i].time)
                    specgram.extend(specgram_list[i].values)
                self.spectrogram = Spectrogram(np.array(time_specgram), specgram_list[0].freq, np.array(specgram))
                return self
            except:
                if self.print_exceptions:
                    print('Cannot compute spectrogram')
                self.spectrogram = None
                return self

    def compute_psd_welch(self, win='hann', L=4096, overlap=0.5, avg_method='median', interpolate=None, scale='log'):
        '''
        Compute power spectral density estimates using Welch's method.

        win (str): window function used to taper the data. Default is Hann-window. See scipy.signal.get_window for a list of
            possible window functions
        L (int): length of each data block for computing the FFT
        avg_time (float): time in seconds that is covered in one time step of the spectrogram. Default value is None and one
            time step covers L samples. If signal covers a long time period it is recommended to use a higher value for avg_time
            to avoid memory overflows and facilitate visualization.
        overlap (float): percentage of overlap between adjecent blocks if Welch's method is used. Parameter is ignored if
            avg_time is None.
        avg_method (str): method for averaging when using Welch's method. Either 'mean' or 'median' can be used
        interpolate (float): resolution in frequency domain in Hz. If not specified, the resolution will be sampling frequency fs
            divided by L. If interpolate is samller than fs/L, the PSD will be interpolated using zero-padding
        scale (str): 'log': PSD in logarithmic scale (dB re 1µPa^2/H) is returned. 'lin': PSD in linear scale (1µPa^2/H) is
            returned


        return ([float], [float]): tuple including time, and spectral level. If no noise date is available,
            function returns two empty numpy arrays.
        '''
        # get noise data segment for each entry in rain_event
        # each noise data segemnt contains usually 1 min of data
        if self.data == None:
            if self.print_exceptions:
                print('Data object is empty. PSD cannot be computed')
            self.psd = None
            return self
        fs = self.data.stats.sampling_rate

        # compute nfft if zero padding is desired
        if interpolate != None:
            if fs / L > interpolate:
                nfft = int(fs / interpolate)
            else: nfft = L
        else: nfft = L

        # compute Welch median for entire data segment
        f, Pxx = signal.welch(x = self.data.data, fs = fs, window=win, nperseg=L, noverlap = int(L * overlap),
            nfft=nfft, average=avg_method)

        if len(Pxx) != int(nfft/2) + 1:
            if self.print_exceptions:
                print('PSD cannot be computed.')
            self.psd = None
            return self

        if scale == 'log':
            Pxx = 10*np.log10(Pxx*np.power(10, self._freq_dependent_sensitivity_correct(int(nfft/2 + 1))/10)) - 128.9
        elif scale == 'lin':
            Pxx = Pxx * np.power(10, self._freq_dependent_sensitivity_correct(int(nfft/2 + 1))/10) * np.power(10, -128.9/10)
        else:
            raise Exception('scale has to be either "lin" or "log".')
        
        self.psd = Psd(f, Pxx)
        return self

    def compute_psd_welch_mp(self, split, n_process=None, win='hann', L=4096, overlap=0.5, avg_method='median',
        interpolate=None, scale='log'):
        '''
        Same as compute_psd_welch but using the multiprocessing library.

        split (float or [datetime.datetime]): time period between start_time and end_time is split into parts of length
            split seconds (if float). The last segment can be shorter than split seconds. Alternatively split can be set as
            an list with N start-end time tuples where N is the number of segments. 
        n_process (int): number of processes in the pool. Default (None) means that n_process is equal to the number
            of CPU cores.
        win (str): window function used to taper the data. Default is Hann-window. See scipy.signal.get_window for a list of
            possible window functions
        L (int): length of each data block for computing the FFT
        avg_time (float): time in seconds that is covered in one time step of the spectrogram. Default value is None and one
            time step covers L samples. If signal covers a long time period it is recommended to use a higher value for avg_time
            to avoid memory overflows and facilitate visualization.
        overlap (float): percentage of overlap between adjecent blocks if Welch's method is used. Parameter is ignored if
            avg_time is None.
        avg_method (str): method for averaging when using Welch's method. Either 'mean' or 'median' can be used
        interpolate (float): resolution in frequency domain in Hz. If not specified, the resolution will be sampling frequency fs
            divided by L. If interpolate is samller than fs/L, the PSD will be interpolated using zero-padding
        scale (str): 'log': PSD in logarithmic scale (dB re 1µPa^2/H) is returned. 'lin': PSD in linear scale (1µPa^2/H) is
            returned

        return ([float], [float]): tuple including frequency indices and PSD estimates, where each PSD estimate. The PSD estimate
            of each segment is stored in a separate row. 
        '''

        # create array with N start and end time values
        if n_process == None:
            N  = mp.cpu_count()
        else:
            N = n_process

        ooi_hyd_data_list = []
        # processa data using same segmentation as for get_acoustic_data_mp. This can save time compared to
        # doing the segmentation from scratch
        if type(split) == type(None):
            for i in range(N):
                tmp_obj = OOIHydrophoneData(starttime=self._data_segmented[i][0].stats.starttime.datetime,
                    endtime=self._data_segmented[i][0].stats.endtime.datetime, print_exceptions=self.print_exceptions)
                tmp_obj.data = self._data_segmented[i][0]
                ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))
        # do segmentation from scratch
        elif type(split) == int or type(split) == float:
            n_seg = int(np.ceil((self.endtime - self.starttime).total_seconds() / split))
            seconds_per_process = (self.endtime - self.starttime).total_seconds() / n_seg
            for k in range(n_seg - 1):
                starttime = self.starttime + datetime.timedelta(seconds=k * seconds_per_process)
                endtime = self.starttime + datetime.timedelta(seconds=(k+1) * seconds_per_process)
                tmp_obj = OOIHydrophoneData(starttime=starttime, endtime=endtime)
                tmp_obj.data = self.data.slice(starttime=UTCDateTime(starttime), endtime=UTCDateTime(endtime))
                ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))
            # treat last segment separately as its length may differ from other segments
            starttime = self.starttime + datetime.timedelta(seconds=(n_seg - 1) * seconds_per_process)
            tmp_obj = OOIHydrophoneData(starttime=starttime, endtime=self.endtime)
            tmp_obj.data = self.data.slice(starttime=UTCDateTime(starttime), endtime=UTCDateTime(self.endtime))
            ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))
        # use segmentation specified by split
        else:
            ooi_hyd_data_list = []
            for row in split:
                tmp_obj = OOIHydrophoneData(starttime=row[0], endtime=row[1])
                tmp_obj.data = self.data.slice(starttime=UTCDateTime(row[0]), endtime=UTCDateTime(row[1]))
                ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))

        with mp.get_context("spawn").Pool(n_process) as p:
            try:
                self.psd_list = p.starmap(_psd_mp_helper, ooi_hyd_data_list)
            except:
                if self.print_exceptions:
                    print('Cannot compute PSd list')
                self.psd_list = None

        return self

def _spectrogram_mp_helper(ooi_hyd_data_obj, win, L, avg_time, overlap):
        ooi_hyd_data_obj.compute_spectrogram(win, L, avg_time, overlap)
        return ooi_hyd_data_obj.spectrogram

def _psd_mp_helper(ooi_hyd_data_obj, win, L, overlap, avg_method, interpolate, scale):
    ooi_hyd_data_obj.compute_psd_welch(win, L, overlap, avg_method, interpolate, scale)
    return ooi_hyd_data_obj.psd


class Spectrogram:
    """
    A class used to represent a spectrogram object.

    Attributes
    ----------
    time : 1-D array of float or datetime objects
        Indices of time-bins of spectrogam.
    freq : 1-D array of float
        Indices of frequency-bins of spectrogram.
    values : 2-D array of float
        Values of the spectrogram. For each time-frequency-bin pair there has to be one entry in values.
        That is, if time has  length N and freq length M, values is a NxM array.

    Methods
    -------
    visualize(plot_spec=True, save_spec=False, filename='spectrogram.png', title='spectrogram',
    xlabel='time', xlabel_rot=70, ylabel='frequency', fmin=0, fmax=32, vmin=20, vmax=80, vdelta=1.0,
    vdelta_cbar=5, figsize=(16,9), dpi=96)
        Visualizes spectrogram using matplotlib.
    save(filename='spectrogram.pickle')
        Saves spectrogram in .pickle file.
    """
    def __init__(self, time, freq, values):
        self.time = time
        self.freq = freq
        self.values = values

    # TODO: allow for visualization of ancillary data. Create SpecgramVisu class?
    def visualize(self, plot_spec=True, save_spec=False, filename='spectrogram.png', title='spectrogram',
        xlabel='time', xlabel_rot=70, ylabel='frequency', fmin=0, fmax=32, vmin=20, vmax=80, vdelta=1.0,
        vdelta_cbar=5, figsize=(16,9), dpi=96, res_reduction_time=1, res_reduction_freq=1):
        '''
        Basic visualization of spectrogram based on matplotlib. The function offers two options: Plot spectrogram
        in Python (plot_spec = True) and save specrogram plot in directory (save_spec = True). Spectrograms are
        plotted in dB re 1µ Pa^2/Hz.

        plot_spec (bool): whether or not spectrogram is plotted using Python
        save_spec (bool): whether or not spectrogram plot is saved
        filename (str): directory where spectrogram plot is saved. Use ending ".png" or ".pdf" to save as PNG or PDF
            file. This value will be ignored if save_spec=False
        title (str): title of plot
        ylabel (str): label of vertical axis
        xlabel (str): label of horizontal axis
        xlabel_rot (float): rotation of xlabel. This is useful if xlabel are longer strings for example when using
            datetime.datetime objects.
        fmin (float): minimum frequency (unit same as f) that is displayed
        fmax (float): maximum frequency (unit same as f) that is displayed
        vmin (float): minimum value (dB) of spectrogram that is colored. All values below are diplayed in white.
        vmax (float): maximum value (dB) of spectrogram that is colored. All values above are diplayed in white.
        vdelta (float): color resolution
        vdelta_cbar (int): label ticks in colorbar are in vdelta_cbar steps
        figsize (tuple(int)): size of figure
        dpi (int): dots per inch
        '''

        #set backend for plotting/saving:
        if not plot_spec: matplotlib.use('Agg')

        font = {'size'   : 22}
        matplotlib.rc('font', **font)

        v = self.values[::res_reduction_time,::res_reduction_freq]

        if len(self.time) != len(self.values):
            t = np.linspace(0, len(self.values) - 1, int(len(self.values) / res_reduction_time))
        else:
            t = self.time[::res_reduction_time]

        if len(self.freq) != len(self.values[0]):
            f = np.linspace(0, len(self.values[0]) - 1, int(len(self.values[0]) / res_reduction_freq))
        else:
            f = self.freq[::res_reduction_freq]

        cbarticks = np.arange(vmin,vmax+vdelta,vdelta)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.contourf(t, f, np.transpose(v), cbarticks, norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=plt.cm.jet)  
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.ylim([fmin, fmax])
        plt.xticks(rotation=xlabel_rot)
        plt.title(title)
        plt.colorbar(im, ax=ax, ticks=np.arange(vmin, vmax+vdelta, vdelta_cbar))
        plt.tick_params(axis='y')

        if type(t[0]) == datetime.datetime:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H:%M'))
        
        if save_spec:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')

        if plot_spec: plt.show()
        else: plt.close(fig)

    def save(self, filename='spectrogram.pickle'):
        '''
        Save spectrogram in pickle file.

        filename (str): directory where spectrogram data is saved. Ending has to be ".pickle".
        '''

        dct = {
            't': self.time,
            'f': self.freq,
            'spectrogram': self.values
            }
        with open(filename, 'wb') as outfile:
            pickle.dump(dct, outfile)


class Psd:
    """
    A calss used to represent a PSD object

    Attributes
    ----------
    freq : array of float
        Indices of frequency-bins of PSD.
    values : array of float
        Values of the PSD.

    Methods
    -------
    visualize(plot_psd=True, save_psd=False, filename='psd.png', title='PSD', xlabel='frequency',
    xlabel_rot=0, ylabel='spectral level', fmin=0, fmax=32, vmin=20, vmax=80, figsize=(16,9), dpi=96)
        Visualizes PSD estimate using matplotlib.
    save(filename='psd.json', ancillary_data=[], ancillary_data_label=[])
        Saves PSD estimate and ancillary data in .json file.
    """
    def __init__(self, freq, values):
        self.freq = freq
        self.values = values

    def visualize(self, plot_psd=True, save_psd=False, filename='psd.png', title='PSD', xlabel='frequency',
        xlabel_rot=0, ylabel='spectral level', fmin=0, fmax=32, vmin=20, vmax=80, figsize=(16,9), dpi=96):
        '''
        Basic visualization of PSD estimate based on matplotlib. The function offers two options: Plot PSD
        in Python (plot_psd = True) and save PSD plot in directory (save_psd = True). PSDs are
        plotted in dB re 1µ Pa^2/Hz.

        plot_psd (bool): whether or not PSD is plotted using Python
        save_psd (bool): whether or not PSD plot is saved
        filename (str): directory where PSD plot is saved. Use ending ".png" or ".pdf" to save as PNG or PDF
            file. This value will be ignored if save_psd=False
        title (str): title of plot
        ylabel (str): label of vertical axis
        xlabel (str): label of horizontal axis
        xlabel_rot (float): rotation of xlabel. This is useful if xlabel are longer strings.
        fmin (float): minimum frequency (unit same as f) that is displayed
        fmax (float): maximum frequency (unit same as f) that is displayed
        vmin (float): minimum value (dB) of PSD.
        vmax (float): maximum value (dB) of PSD.
        figsize (tuple(int)): size of figure
        dpi (int): dots per inch
        '''

        #set backend for plotting/saving:
        if not plot_psd: matplotlib.use('Agg')

        font = {'size'   : 22}
        matplotlib.rc('font', **font)

        if len(self.freq) != len(self.values):
            f = np.linspace(0, len(self.values)-1, len(self.values))
        else:
            f = self.freq

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        plt.semilogx(f, self.values)  
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xlim([fmin, fmax])
        plt.ylim([vmin, vmax])
        plt.xticks(rotation=xlabel_rot)
        plt.title(title)
        plt.grid(True)
        
        if save_psd:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')

        if plot_psd: plt.show()
        else: plt.close(fig)

    def save(self, filename='psd.json', ancillary_data=[], ancillary_data_label=[]):
        '''
        Save PSD estimates along with with ancillary data (stored in dictionary) in json file.

        filename (str): directory for saving the data
        ancillary_data ([array like]): list of ancillary data
        ancillary_data_label ([str]): labels for ancillary data used as keys in the output dictionary.
            Array has same length as ancillary_data array.
        '''

        if len(self.freq) != len(self.values):
            f = np.linspace(0, len(self.values)-1, len(self.values))
        else:
            f = self.freq

        if type(self.values) != list:
            values = self.values.tolist()

        if type(f) != list:
            f = f.tolist()

        dct = {
            'psd': values,
            'f': f
            }

        if len(ancillary_data) != 0:
            for i in range(len(ancillary_data)):
                if type(ancillary_data[i]) != list:
                    dct[ancillary_data_label[i]] = ancillary_data[i].tolist()
                else:
                    dct[ancillary_data_label[i]] = ancillary_data[i]

        with open(filename, 'w+') as outfile:
            json.dump(dct, outfile)


class Hydrophone_Xcorr:


    def __init__(self, node1, node2, avg_time, W=30, verbose=True, fmin=None, fmax=None, mp = True, ckpts=True):
        ''' 
        Initialize Class OOIHydrophoneData

        Attributes
        ----------
        starttime : datetime.datetime
            indicates start time for acquiring data

        node1 : str
            indicates location of reference hydrophone (see table 1 for valid inputs)
        node2 : str
            indicates location of compared hydrophone (see table 1 for valid inputs)
        avg_time : int or float
            indicates length of data pulled from server for one averaging period (minutes)
        W : int or float
            indicates cross correlation window (seconds)
        verbose : bool
            indicates whether to print updates or not
        fmin : int or float
            indicates the lower cutoff frequency of applied bandpass filter. If None, then no filter is applied
        fmax : int or float
            indicates the upper cutoff frequency of applied bandpass filter. If None, then no filter is applied
        mp : bool
            indicates if multiprocessing functions should be used
        ckpts : bool
            indicates if checkpoints are saved in working directory ./ckpts
       
        Private Attributes
        ------------------
        None at this time

        Methods
        -------
        distance_between_hydrophones(self, coord1, coord2)
            Calculates the distance in meteres between hydrophones
        get_audio_avg_period(self, start_time)
            Pulls avg_period amount of data from server
        xcorr_over_avg_period(self, h1, h2)
            Computes cross-correlation for window of length W, averaged over avg_period
        avg_over_mult_periods(self, num_periods, start_time, ckpts=True)
            runs xcorr_over_avg_period() for num_periods amount of periods
  
        Private Methods
        ---------------
        None at this time

        TABLE 1:
            ____________________________________________
            |Node Name |        Hydrophone Name        |
            |__________|_______________________________|
            |'/LJ01D'  | Oregon Shelf Base Seafloor    |
            |__________|_______________________________|
            |'/LJ01A   | Oregon Slope Base Seafloore   |
            |__________|_______________________________|
            |'/PC01A'  | Oregan Slope Base Shallow     |
            |__________|_______________________________|
            |'/PC03A'  | Axial Base Shallow Profiler   |
            |__________|_______________________________|
            |'/LJ01C'  | Oregon Offshore Base Seafloor |
            |__________|_______________________________|
        '''
        hydrophone_locations = {'/LJ01D':[44.63714, -124.30598], '/LJ01C':[44.36943, -124.95357], '/PC01A':[44.52897, -125.38967], '/LJ01A':[44.51512, -125.38992], '/LJ03A':[45.81668, -129.75435], '/PC03A':[45.83049, -129.75327]}
        
        self.hydrophone_locations = hydrophone_locations
        self.node1 = node1
        self.node2 = node2
        self.W = W
        self.verbose = verbose
        self.avg_time = avg_time
        self.mp = mp
        self.ckpts = ckpts
        self.Fs = 64000
        self.Ts = 1/self.Fs
        self.fmin = fmin
        self.fmax = fmax
        
        
        self.__distance_between_hydrophones(hydrophone_locations[node1],hydrophone_locations[node2])
        self.__bearing_between_hydrophones(hydrophone_locations[node1],hydrophone_locations[node2])
        
        print('Distance Between Hydrophones: ', self.distance,' meters')
        print('Estimate Time Delay Between Hydrophones: ',self.time_delay,' seconds')
        print('Bearing Between Hydrophone 1 and 2: ', self.theta_bearing_d_1_2,' degrees')
        
   
    # Calculate Distance Between 2 Hydrophones
    # function from https://www.geeksforgeeks.org/program-distance-two-points-earth/
    def __distance_between_hydrophones(self, coord1, coord2): 
        '''
        distance_between_hydrophones(coord1, coord2) - calculates the distance in meters between two global cooridinates
        
        Inputs:
        coord1 - numpy array of shape [2,1] containing latitude and longitude of point 1
        coord2 - numpy array of shape [2,1] containing latitude and longitude of point 2
        
        Outpus:
        self.distance - distance between 2 hydrophones in meters
        self.time_delay - approximate time delay between 2 hydrophones (assuming speed of sound = 1480 m/s)
        
        '''
        from math import radians, cos, sin, asin, sqrt 
        # The math module contains a function named 
        # radians which converts from degrees to radians. 
        lon1 = radians(coord1[1]) 
        lon2 = radians(coord2[1]) 
        lat1 = radians(coord1[0]) 
        lat2 = radians(coord2[0]) 

        # Haversine formula  
        dlon = lon2 - lon1  
        dlat = lat2 - lat1 
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

        c = 2 * asin(sqrt(a))  

        # Radius of earth in kilometers. Use 3956 for miles 
        r = 6371000
        D = c*r

        self.distance = D
        self.time_delay = D/1480

    
    def __bearing_between_hydrophones(self, coord1, coord2):
        '''
        bearing_between_hydrophones(coord1, coord2) - calculates the bearing in degrees (NSEW) between coord1 and coord2

        Inputs:
        coord1 - numpy array
            of shape [2,1] containing latitude and longitude of point1
        coord2 - numpy array
            of shape [2,1] containing latitude and longitude of point2

        Outputs:
        self.bearing_d_1_2 - float
            bearing in degrees between node 1 and node 2
        '''

        psi1 = np.deg2rad(coord1[0])
        lambda1 = np.deg2rad(coord1[1])
        psi2 = np.deg2rad(coord2[0])
        lambda2 = np.deg2rad(coord2[1])
        del_lambda = lambda2-lambda1

        y = np.sin(del_lambda)*np.cos(psi2)
        x = np.cos(psi1)*np.sin(psi2) - np.sin(psi1)*np.cos(psi2)*np.cos(del_lambda);

        theta_bearing_rad = np.arctan2(y,x)
        theta_bearing_d_1_2 = (np.rad2deg(theta_bearing_rad)+360) % 360

        self.theta_bearing_d_1_2 = theta_bearing_d_1_2


    def get_audio_avg_period(self, start_time):

        '''
        Downloads, and Reshapes Data from OOI server for given average period and start time
        
        Inputs:
        start_time - indicates UTC time that data starts with
       
        Outputs:
        h1_reshaped : float
            hydrophone data from node 1 of shape (B,N) where B = avg_time*60/W and N = W*Fs
        h2_reshaped : float
            hydrophone data from node 2 of shape (B,N) where B = avg_time*60/W and N = W*Fs
        flag : bool
            TODO flag stucture to be added later
        '''

        # Frequency Design Functions
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = signal.butter(order, [high], btype='lowpass')
            return b, a
        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = signal.lfilter(b, a, data)
            return y
        
        flag = False
        avg_time = self.avg_time
        verbose = self.verbose
        W = self.W
        fmax = self.fmax
        fmin = self.fmin
        
        avg_time_seconds = avg_time * 60
        
        if avg_time_seconds % W != 0:
            print('Error: Average Time Must Be Interval of Window')
            return None
        # Initialze Two Classes for Two Hydrophones
        ooi1 = OOIHydrophoneData(limit_seed_files=False, print_exceptions=True)
        ooi2 = OOIHydrophoneData(limit_seed_files=False, print_exceptions=True)
        ooi1.data_gap_mode=1
        ooi2.data_gap_mode=1

        # Calculate end_time
        end_time = start_time + timedelta(minutes=avg_time)

        if verbose: print('Getting Audio from Node 1...')
        stopwatch_start = time.time()
        
        #Audio from Node 1
        if self.mp: ooi1.get_acoustic_data_mp(start_time, end_time, node=self.node1)
        else: ooi1.get_acoustic_data(start_time, end_time, node=self.node1)
        
        if verbose: print('Getting Audio from Node 2...')
        #Audio from Node 2
        if self.mp: ooi2.get_acoustic_data_mp(start_time, end_time, node=self.node2)
        else: ooi2.get_acoustic_data(start_time, end_time, node=self.node2)
        
        if (ooi1.data == None) or (ooi2.data == None):
            print('Error with Getting Audio')
            return None, None, None
        #Combine Data into Stream
        data_stream = obspy.Stream(traces=[ooi1.data, ooi2.data])
        
        stopwatch_end = time.time()
        print('Time to Download Data from Server: ',stopwatch_end-stopwatch_start)
        
        if data_stream[0].data.shape != data_stream[1].data.shape:
            print('Data streams are not the same length. Flag to be added later')
            # TODO: Set up flag structure of some kind
            
        # Make Data Zero Mean
        data_stream[0].data = data_stream[0].data - np.mean(data_stream[0].data) 
        data_stream[1].data = data_stream[1].data - np.mean(data_stream[1].data)
                      
        # Cut off extra points if present
        h1_data = data_stream[0].data[:avg_time*60*self.Fs]
        h2_data = data_stream[1].data[:avg_time*60*self.Fs]

        # Set fill value to zero and fill in mask if there are gaps
        if ooi1.data_gap:
            h1_data.fill_value = 0
            h1_data = np.ma.filled(h1_data)
        if ooi2.data_gap:
            h2_data.fill_value = 0
            h2_data = np.ma.filled(h2_data)
            
        if ((h1_data.shape[0] < avg_time*60*self.Fs)):
            print('Length of Audio at node 1 too short, zeros added. Length: ', data_stream[0].data.shape[0])
            h1_data = np.pad(h1_data, (0, avg_time*60*self.Fs-data_stream[0].data.shape[0]))

        if ((h2_data.shape[0] < avg_time*60*self.Fs)):
            print('Length of Audio at node 2 too short, zeros added. Length: ', data_stream[1].data.shape[0])
            h2_data = np.pad(h2_data, (0, avg_time*60*self.Fs-data_stream[1].data.shape[0]))
        
        # Filter Data
        if (fmin != None) and (fmax != None):
            h1_data = butter_bandpass_filter(h1_data, fmin, fmax, data_stream[0].stats.sampling_rate)
            h2_data = butter_bandpass_filter(h2_data, fmin, fmax, data_stream[1].stats.sampling_rate)
        
        h1_reshaped = np.reshape(h1_data,(int(avg_time*60/W), int(W*self.Fs)))
        h2_reshaped = np.reshape(h2_data,(int(avg_time*60/W), int(W*self.Fs)))                    
              
        return h1_reshaped, h2_reshaped, flag
    
    
    def xcorr_over_avg_period(self, h1, h2):
        '''
        finds cross correlation over average period and avereages all correlations
        
        Inputs:
        h1 - audio data from hydrophone 1
        h2 - audio data from hydrophone 2
        
        Output - avg_xcorr of shape (N) where N = W*Fs
        '''
        verbose = self.verbose
        avg_time = self.avg_time
        M = h1.shape[1]
        N = h2.shape[1]

        xcorr = np.zeros((int(avg_time*60/30),int(N+M-1)))
        
        
        stopwatch_start = time.time()
        if verbose:
            bar = progressbar.ProgressBar(maxval=h1.shape[0], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
        for k in range(h1.shape[0]):
            xcorr[k,:] = scipy.signal.correlate(h1[k,:],h2[k,:],'full')
            if verbose: bar.update(k+1)
        
        avg_xcorr = np.average(xcorr,axis=0)
        stopwatch_end = time.time()
        print('Time to Calculate Cross Correlation of 1 period: ',stopwatch_end-stopwatch_start)
        return avg_xcorr
    
 
    def avg_over_mult_periods(self, num_periods, start_time):
        '''
        Computes average over num_periods of averaging periods
        
        Inputs:
        num_periods - number of periods to average over
        start_time - start time for data
        
        Outputs:
        xcorr - average xcorr over num_periods of averaging
        '''
        verbose = self.verbose
        
        first_loop = True
        for k in range(num_periods):
            stopwatch_start = time.time()
            if verbose: print('Time Period: ',k + 1)
            
            h1, h2, flag = self.get_audio_avg_period(start_time)
            

            # Compute Cross Correlation for Each Window and Average
            if first_loop:
                xcorr_avg_period = self.xcorr_over_avg_period(h1, h2)
                xcorr = xcorr_avg_period
                first_loop = False
            else:
                xcorr_avg_period = self.xcorr_over_avg_period(h1, h2)
                xcorr += xcorr_avg_period
                start_time = start_time + timedelta(minutes=self.avg_time)
            
            stopwatch_end = time.time()
            print('Time to Complete 1 period: ',stopwatch_end - stopwatch_start)
            
            #Save Checkpoints for every average period
            filename = './ckpts/ckpt_' + str(k) + '.pkl'
            
            with open(filename,'wb') as f:
                pickle.dump(xcorr, f)
                pickle.dump(xcorr_avg_period, f)
                pickle.dump(k,f)

            # Calculate time variable TODO change to not calculate every loop
            dt = self.Ts
            t = np.arange(-xcorr.shape[0]*dt/2,xcorr.shape[0]*dt/2,dt)
            
        xcorr = xcorr / num_periods
        # Calculate Bearing of Max Peak
        max_idx = np.argmax(xcorr)
        time_of_max = t[max_idx]

        #bearing is with respect to node1 (where node2 is at 0 deg)
        bearing_max_local = [np.rad2deg(np.arccos(1480*time_of_max/self.distance)), -np.rad2deg(np.arccos(1480*time_of_max/self.distance))]
        #convert bearing_max_local to numpy array
        bearing_max_local = np.array(bearing_max_local)
        #convert to global (NSEW) degrees
        bearing_max_global = self.theta_bearing_d_1_2 + bearing_max_local
        #make result between 0 and 360
        bearing_max_global = bearing_max_global % 360
        self.bearing_max_global = bearing_max_global

        return t, xcorr, bearing_max_global
    
    
    def plot_map_bearing(self):
        coord1 = self.hydrophone_locations[self.node1]
        coord2 = self.hydrophone_locations[self.node2]
        thetaB1 = self.bearing_max_global[0]
        thetaB2 = self.bearing_max_global[1]
        
        midpoint, phantom_point1 = self.__find_phantom_point(coord1, coord2, thetaB1)
        midpoint, phantom_point2 = self.__find_phantom_point(coord1, coord2, thetaB2)

        import plotly.graph_objects as go

        hyd_lats = [coord1[0], coord2[0]]
        hyd_lons = [coord1[1], coord2[1]]

        antmidpoint = self.__get_antipode(midpoint)
        fig = go.Figure()

        fig.add_trace(go.Scattergeo(
            lat = [midpoint[0], phantom_point1[0], antmidpoint[0]],
            lon = [midpoint[1], phantom_point1[1], antmidpoint[1]],
            mode = 'lines',
            line = dict(width = 1, color = 'blue')
        ))

        fig.add_trace(go.Scattergeo(
            lat = [midpoint[0], phantom_point2[0], antmidpoint[0]],
            lon = [midpoint[1], phantom_point2[1], antmidpoint[1]],
            mode = 'lines',
            line = dict(width = 1, color = 'green')
        ))

        fig.add_trace(go.Scattergeo(
            lon = hyd_lons,
            lat = hyd_lats,
            hoverinfo = 'text',
            text = ['Oregon Slope Base Hydrophone','Oregon Cabled Benthic Hydrophone'],
            mode = 'markers',
            marker = dict(
                size = 5,
                color = 'rgb(255, 0, 0)',
                line = dict(
                    width = 3,
                    color = 'rgba(68, 68, 68, 0)'
                )
            )))


        fig.update_layout(
            title_text = 'Possible Bearings of Max Correlation Peak',
            showlegend = False,
            geo = dict(
                resolution = 50,
                showland = True,
                showlakes = True,
                landcolor = 'rgb(204, 204, 204)',
                countrycolor = 'rgb(204, 204, 204)',
                lakecolor = 'rgb(255, 255, 255)',
                projection_type = "natural earth",
                coastlinewidth = 1,
                lataxis = dict(
                    #range = [20, 60],
                    showgrid = True,
                    dtick = 10
                ),
                lonaxis = dict(
                    #range = [-100, 20],
                    showgrid = True,
                    dtick = 20
                ),
            )
        )

        fig.show()

    def __find_phantom_point(self, coord1, coord2, thetaB):
        '''
        find_phantom_point

        Inputs:
        coord1 - list
            coordinate of first hydrophone
        coord2 - list
            coordinate of second hydrophone

        Output:
        midpoint, phantom_point
        '''
        midpoint = [coord1[0] - (coord1[0] - coord2[0])/2, coord1[1] - (coord1[1] - coord2[1])/2]

        del_lat = 0.01*np.cos(np.deg2rad(thetaB))
        del_lon = 0.01*np.sin(np.deg2rad(thetaB))

        phantom_point = [midpoint[0] + del_lat, midpoint[1] + del_lon]

        return midpoint, phantom_point
    
    def __get_antipode(self, coord):
        # get antipodes
        antlon=coord[1]+180
        if antlon>360:
            antlon= antlon - 360
        antlat=-coord[0]
        antipode_coord = [antlat, antlon]
        return antipode_coord    