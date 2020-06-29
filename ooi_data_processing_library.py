
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

class OOIHyrophoneData:

    def __init__(self, starttime=None, endtime=None, node=None, fmin=None,
        fmax=None, apply_filter=True, print_exceptions=None):

        self.starttime = starttime
        self.endtime = endtime
        self.node = node
        self.fmin = fmin
        self.fmax = fmax
        self.apply_filter = apply_filter
        self.print_exceptions = print_exceptions
        self.data_available = None

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

        return ([str]): list of URLs, each URL refers to one data file. If no data is avauilable for
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


    def get_acoustic_data(self, starttime, endtime, node, fmin=20.0, fmax=30000.0):
        '''
        Get acoustic data for specific time frame and node:

        start_time (datetime.datetime): time of the first noise sample
        end_time (datetime.datetime): time of the last noise sample
        node (str): hydrophone
        fmin (float): lower cutoff frequency of hydrophone's bandpass filter
        fmax (float): higher cutoff frequency of hydrophones bandpass filter
        print_exceptions (bool): whether or not exeptions are printed in the terminal line

        return (obspy.core.stream.Stream): obspy Stream object containing one Trace and date
            between start_time and end_time. Returns None if no data are available for specified time frame

        '''

        self.starttime = starttime
        self.endtime = endtime
        self.node = node
        self.fmin = fmin
        self.fmax = fmax
        
        # get URLs
        day_start = UTCDateTime(self.starttime.year, self.starttime.month, self.starttime.day, 0, 0, 0)
        data_url_list = self.__web_crawler_noise(self.starttime.strftime("/%Y/%m/%d/"))
        if data_url_list == None:
            if self.print_exceptions:
                print('No data available for specified day and node. Please change the day or use a differnt node')
            self.data = None
            self.data_available = False
            return None
        
        day_start = day_start + 24*3600
        while day_start < self.endtime:
            data_url_list.extend(self.__web_crawler_noise(self.starttime.strftime("/%Y/%m/%d/")))
            day_start = day_start + 24*3600
        
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
                
        st_all = None

        # only acquire data for desired time
        for i in range(len(data_url_list)):
            # get UTC time of current and next item in URL list
            utc_time_url_start = UTCDateTime(data_url_list[i].split('YDH')[1][1:].split('.mseed')[0])
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
                
                try:
                    st = read(data_url_list[i], apply_calib=True)
                except:
                    if self.print_exceptions:
                        print("Data are broken")
                    self.data = None
                    self.data_available = False
                    return None
                
                # slice stream to get desired data
                st = st.slice(UTCDateTime(self.starttime), UTCDateTime(self.endtime))
                
                if st_all == None: st_all = st
                else: 
                    st_all += st
                    st_all.merge(fill_value ='interpolate', method=1)
                    
        if st_all != None:
            if len(st_all) == 0:
                if self.print_exceptions:
                    print('No data available for selected time frame.')
                self.data = None
                self.data_available = False
                return None

        try:
            #st_all = st_all.split()
            if self.apply_filter:
                if self.fmin == None:
                    fmin = 0.0
                if self.fmax == None:
                    fmax = st_all[0].stats.sampling_rate
                st_all = st_all.filter("bandpass", freqmin=fmin, freqmax=fmax)
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

    def get_acoustic_data_mp(self, starttime, endtime, node, n_process=None, fmin=20.0, fmax=30000.0):
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
            node, fmin, fmin) for i in range(N)]
        
        # create pool of processes require one part of the data in each process
        apply_filter_temp = self.apply_filter
        self.apply_filter = False
        with mp.get_context("spawn").Pool(N) as p:
            try:
                data_list = p.starmap(self.get_acoustic_data, get_data_list)
            except:
                if self.print_exceptions:
                    print('Data cannot be requested.')
                self.data = None
                self.data_available = False
                self.starttime = starttime
                self.endtime = endtime
                return self.data

        if None in data_list:
            if self.print_exceptions:
                print('No data available for specified time and node')
            self.data = None
            self.data_available = False
            data = None
        else:
            # merge data segments together
            data = data_list[0]
            for d in data_list[1:]:
                data = data + d
            self._data_segmented = data
            data.merge(fill_value='interpolate', method=1)

            # apply bandpass filter to data if desired
            if apply_filter_temp:
                if self.fmin == None:
                    fmin = 0.0
                if self.fmax == None:
                    fmax = data[0].stats.sampling_rate
                data = data.filter("bandpass", freqmin=fmin, freqmax=fmax)

            self.data = data[0]
            self.data_available = True

        self.starttime = starttime
        self.endtime = endtime
        return data

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
                tmp_obj = OOIHyrophoneData(starttime=self._data_segmented[i][0].stats.starttime.datetime,
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
                tmp_obj = OOIHyrophoneData(starttime=starttime, endtime=endtime)
                tmp_obj.data = self.data.slice(starttime=starttime, endtime=endtime)
                ooi_hyd_data_list.append((tmp_obj, win, L, avg_time, overlap))


            starttime = self.starttime + datetime.timedelta(seconds=(n_seg - 1) * seconds_per_process)
            tmp_obj = OOIHyrophoneData(starttime=starttime, endtime=self.endtime)
            tmp_obj.data = self.data.slice(starttime=starttime, endtime=self.endtime)
            ooi_hyd_data_list.append((tmp_obj, win, L, avg_time, overlap))

        with mp.get_context("spawn").Pool(n_process) as p:
            try:
                specgram_list = p.starmap(self.__spectrogram_mp_helper, ooi_hyd_data_list)
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

    def __spectrogram_mp_helper(self, ooi_hyd_data_obj, win, L, avg_time, overlap):
        ooi_hyd_data_obj.compute_spectrogram(win, L, avg_time, overlap)
        return ooi_hyd_data_obj.spectrogram


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
        if split == None:
            for i in range(N):
                tmp_obj = OOIHyrophoneData(starttime=self._data_segmented[i][0].stats.starttime.datetime,
                    endtime=self._data_segmented[i][0].stats.endtime.datetime)
                tmp_obj.data = self._data_segmented[i][0]
                ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))
        # do segmentation from scratch
        elif type(split) == int or type(split) == float:
            n_seg = int(np.ceil((self.endtime - self.starttime).total_seconds() / split))
            seconds_per_process = (self.endtime - self.starttime).total_seconds() / n_seg
            for k in range(n_seg - 1):
                starttime = self.starttime + datetime.timedelta(seconds=k * seconds_per_process)
                endtime = self.starttime + datetime.timedelta(seconds=(k+1) * seconds_per_process)
                tmp_obj = OOIHyrophoneData(starttime=starttime, endtime=endtime)
                tmp_obj.data = self.data.slice(starttime=starttime, endtime=endtime)
                ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))
            # treat last segment separately as its length may differ from other segments
            starttime = self.starttime + datetime.timedelta(seconds=(n_seg - 1) * seconds_per_process)
            tmp_obj = OOIHyrophoneData(starttime=starttime, endtime=self.endtime)
            tmp_obj.data = self.data.slice(starttime=starttime, endtime=self.endtime)
            ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))
        # use segmentation specified by split
        else:
            ooi_hyd_data_list = []
            for row in split:
                tmp_obj = OOIHyrophoneData(starttime=row[0], endtime=row[1])
                tmp_obj.data = self.data.slice(starttime=row[0], endtime=row[1])
                ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))

        with mp.get_context("spawn").Pool(n_process) as p:
            try:
                self.psd_list = p.starmap(self.__psd_mp_helper, ooi_hyd_data_list)
            except:
                if self.print_exceptions:
                    print('Cannot compute PSd list')
                self.psd_list = None

        return self

    def __psd_mp_helper(self, ooi_hyd_data_obj, win, L, overlap, avg_method, interpolate, scale):
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

    """
    def __init__(self, time, freq, values):
        self.time = time
        self.freq = freq
        self.values = values

    # TODO: allow for visualization of ancillary data. Create SpecgramVisu class?
    def visualize(self, plot_spec=True, save_spec=False, filename='spectrogram.png', title='spectrogram',
        xlabel='time', xlabel_rot=70, ylabel='frequency', fmin=0, fmax=32, vmin=20, vmax=80, vdelta=1.0,
        vdelta_cbar=5, figsize=(16,9), dpi=96):
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

        if len(self.time) != len(self.values):
            t = np.linspace(0, len(self.values) - 1, len(self.values))
        else:
            t = self.time
        if len(self.freq) != len(self.values[0]):
            f = np.linspace(0, len(self.values[0]) - 1, len(self.values[0]))
        else:
            f = self.freq

        cbarticks = np.arange(vmin,vmax+vdelta,vdelta)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.contourf(t, f, np.transpose(self.values), cbarticks, norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=plt.cm.jet)  
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