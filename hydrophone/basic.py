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
import concurrent.futures
import logging
import copy


class HydrophoneData(Trace):

    def __init__(self, data=np.array([]), header=None, node=''):
        
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
            indicates minimum frequency in bandpass 
            . Default value is None, which results in unfiltered signal
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
            mode 2 - makes valid data zero mean, and fills invalid data with zeros

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
        super(HydrophoneData, self).__init__(data, header)

        self.node = node

        #self.starttime = starttime
        #self.endtime = endtime
        #self.node = node
        #self.fmin = fmin
        #self.fmax = fmax
        #self.print_exceptions = print_exceptions
        #self.data_available = None
        #self.limit_seed_files = limit_seed_files
        #self.data_gap = False
        #self.data_gap_mode = data_gap_mode

        self.spectrogram = None
        self.psd = None
        self.psd_list = None


    #TODO: use correct frequency response for hydrophones
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


    

    def compute_spectrogram(self, win='hann', L=4096, avg_time=None, overlap=0.5, verbose=True):
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
            
        if any(self.data) == None:
            if verbose:
                print('Data object is empty. Spectrogram cannot be computed')
            self.spectrogram = None
            return None

        # sampling frequency
        fs = self.stats.sampling_rate

        # number of time steps
        if avg_time == None:
            nbins = int(len(self.data) / L)
        else:
            nbins = int(np.ceil(len(self.data) / (avg_time * fs)))

        # compute spectrogram. For avg_time=None (periodogram for each time step), the last data samples are ignored if 
        # len(noise[0].data) != k * L
        if avg_time == None:
            for n in range(nbins - 1):
                f, Pxx = signal.periodogram(x = self.data[n*L:(n+1)*L], fs=fs, window=win)
                if len(Pxx) != int(L/2)+1:
                    if verbose:
                        print('Error while computing periodogram for segment', n)
                    self.spectrogram = None
                    return None
                else:
                    Pxx = 10*np.log10(Pxx * np.power(10, self._freq_dependent_sensitivity_correct(int(L/2 + 1))/10))-128.9
                    specgram.append(Pxx)
                    time.append(self.stats.starttime + datetime.timedelta(seconds=n*L / fs))

        else:
            for n in range(nbins - 1):
                f, Pxx = signal.welch(x = self.data[n*int(fs*avg_time):(n+1)*int(fs*avg_time)],
                    fs=fs, window=win, nperseg=L, noverlap = int(L * overlap), nfft=L, average='median')
                if len(Pxx) != int(L/2)+1:
                    if verbose:
                        print('Error while computing Welch estimate for segment', n)
                    self.spectrogram = None
                    return None
                else:
                    Pxx = 10*np.log10(Pxx * np.power(10, self._freq_dependent_sensitivity_correct(int(L/2 + 1))/10))-128.9
                    specgram.append(Pxx)
                    time.append(self.stats.starttime + datetime.timedelta(seconds=n*avg_time))

            # compute PSD for residual segment if segment has more than L samples
            if len(self.data[int((nbins - 1) * fs * avg_time):]) >= L:
                f, Pxx = signal.welch(x = self.data[int((nbins - 1) * fs * avg_time):],
                    fs=fs, window=win, nperseg=L, noverlap = int(L * overlap), nfft=L, average='median')
                if len(Pxx) != int(L/2)+1:
                    if verbose:
                        print('Error while computing Welch estimate residual segment')
                    self.spectrogram = None
                    return None
                else:
                    Pxx = 10*np.log10(Pxx * np.power(10, self._freq_dependent_sensitivity_correct(int(L/2 + 1))/10))-128.9
                    specgram.append(Pxx)
                    time.append(self.stats.starttime + datetime.timedelta(seconds=(nbins-1)*avg_time))
    
        if len(time) == 0:
            if verbose:
                print('Spectrogram does not contain any data')
            self.spectrogram = None
            return None
        else:
            self.spectrogram = Spectrogram(np.array(time), np.array(f), np.array(specgram))
            return self.spectrogram

    def compute_spectrogram_mp(self, n_process=None, win='hann', L=4096, avg_time=None, overlap=0.5, verbose=True):
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
        #if split == None:
        #    for i in range(N):
        #        tmp_obj = HydrophoneData(starttime=self._data_segmented[i][0].stats.starttime.datetime,
        #            endtime=self._data_segmented[i][0].stats.endtime.datetime)
        #        tmp_obj.data = self._data_segmented[i][0]
        #        ooi_hyd_data_list.append((tmp_obj, win, L, avg_time, overlap))
        # do segmentation from scratch
        #else:
        seconds_per_process = (self.stats.endtime - self.stats.starttime).total_seconds() / N
        for k in range(N - 1):
            starttime = self.stats.starttime + datetime.timedelta(seconds=k * seconds_per_process)
            endtime = self.stats.starttime + datetime.timedelta(seconds=(k+1) * seconds_per_process)
            stats_tmp = copy.deepcopy(self.stats)
            stats_tmp.starttime = starttime
            stats_tmp.endtime = endtime
            tmp_obj = HydrophoneData(header=stats_tmp)
            tmp_obj.data = self.data.slice(starttime=starttime, endtime=endtime)
            ooi_hyd_data_list.append((tmp_obj, win, L, avg_time, overlap))


        starttime = self.stats.starttime + datetime.timedelta(seconds=(N - 1) * seconds_per_process)
        stats_tmp = copy.deepcopy(self.stats)
        stats_tmp.starttime = starttime
        stats_tmp.endtime = self.stats.endtime
        tmp_obj = HydrophoneData(header=stats_tmp)
        tmp_obj.data = self.data.slice(starttime=starttime, endtime=self.stats.endtime)
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
                return self.spectrogram
            except:
                if verbose:
                    print('Cannot compute spectrogram')
                self.spectrogram = None
                return None

    def compute_psd_welch(self, win='hann', L=4096, overlap=0.5, avg_method='median', interpolate=None, scale='log', verbose=True):
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
        if any(self.data) == None:
            if verbose:
                print('Data object is empty. PSD cannot be computed')
            self.psd = None
            return None
        fs = self.stats.sampling_rate

        # compute nfft if zero padding is desired
        if interpolate != None:
            if fs / L > interpolate:
                nfft = int(fs / interpolate)
            else: nfft = L
        else: nfft = L

        # compute Welch median for entire data segment
        f, Pxx = signal.welch(x = self.data, fs = fs, window=win, nperseg=L, noverlap = int(L * overlap),
            nfft=nfft, average=avg_method)

        if len(Pxx) != int(nfft/2) + 1:
            if verbose:
                print('PSD cannot be computed.')
            self.psd = None
            return None

        if scale == 'log':
            Pxx = 10*np.log10(Pxx*np.power(10, self._freq_dependent_sensitivity_correct(int(nfft/2 + 1))/10)) - 128.9
        elif scale == 'lin':
            Pxx = Pxx * np.power(10, self._freq_dependent_sensitivity_correct(int(nfft/2 + 1))/10) * np.power(10, -128.9/10)
        else:
            raise Exception('scale has to be either "lin" or "log".')
        
        self.psd = Psd(f, Pxx)
        return self.psd

    def compute_psd_welch_mp(self, split, n_process=None, win='hann', L=4096, overlap=0.5, avg_method='median',
        interpolate=None, scale='log', verbose=True):
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
        #if type(split) == type(None):
        #    for i in range(N):
        #        tmp_obj = OOIHydrophoneData(starttime=self._data_segmented[i][0].stats.starttime.datetime,
        #            endtime=self._data_segmented[i][0].stats.endtime.datetime, print_exceptions=self.print_exceptions)
        #        tmp_obj.data = self._data_segmented[i][0]
        #        ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))
        # do segmentation from scratch
        if isinstance(split, int) or isinstance(split, float):
            n_seg = int(np.ceil((self.stats.endtime - self.stats.starttime).total_seconds() / split))
            seconds_per_process = (self.stats.endtime - self.stats.starttime).total_seconds() / n_seg
            for k in range(n_seg - 1):
                starttime = self.stats.starttime + datetime.timedelta(seconds=k * seconds_per_process)
                endtime = self.stats.starttime + datetime.timedelta(seconds=(k+1) * seconds_per_process)
                stats_tmp = copy.deepcopy(self.stats)
                stats_tmp.starttime = starttime
                stats_tmp.endtime = endtime
                tmp_obj = HydrophoneData(header=stats_tmp)
                tmp_obj.data = self.data.slice(starttime=UTCDateTime(starttime), endtime=UTCDateTime(endtime))
                ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))
            # treat last segment separately as its length may differ from other segments
            starttime = self.stats.starttime + datetime.timedelta(seconds=(n_seg - 1) * seconds_per_process)
            stats_tmp = copy.deepcopy(self.stats)
            stats_tmp.starttime = starttime
            stats_tmp.endtime = self.stats.endtime
            tmp_obj = HydrophoneData(header=stats_tmp)
            tmp_obj.data = self.data.slice(starttime=UTCDateTime(starttime), endtime=UTCDateTime(self.stats.endtime))
            ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))
        # use segmentation specified by split
        else:
            ooi_hyd_data_list = []
            for row in split:
                stats_tmp = copy.deepcopy(self.stats)
                stats_tmp.starttime = row[0]
                stats_tmp.endtime = row[1]
                tmp_obj = HydrophoneData(header=stats_tmp)
                tmp_obj.data = self.data.slice(starttime=UTCDateTime(row[0]), endtime=UTCDateTime(row[1]))
                ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))

        with mp.get_context("spawn").Pool(n_process) as p:
            try:
                self.psd_list = p.starmap(_psd_mp_helper, ooi_hyd_data_list)
            except:
                if verbose:
                    print('Cannot compute PSd list')
                self.psd_list = None

        return self.psd_list

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



