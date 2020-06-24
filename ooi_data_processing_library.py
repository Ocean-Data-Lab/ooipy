
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
from scipy import signal
import requests
import urllib
import datetime
import time
import pandas as pd
import sys
from thredds_crawler.crawl import Crawl
import multiprocessing as mp
import pickle



def _web_crawler_noise(day_str, node):
    '''
    get URLs for a specific day from OOI raw data server

    day_str (str): date for which URLs are requested; format: yyyy/mm/dd, e.g. 2016/07/15
    node (str): specifies the hydrophone; see bolow for possible values

    return ([str]): list of URLs, each URL refers to one data file. If no data is avauilable for
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

def _freq_dependent_sensitivity_correct(N):
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


def get_noise_data(start_time, end_time, node='/LJ01D', fmin=20.0, fmax=30000.0, print_exceptions=False):
    '''
    Get noise data for specific time frame and node:

    start_time (datetime.datetime): time of the first noise sample
    end_time (datetime.datetime): time of the last noise sample
    node (str): hydrophone
    fmin (float): lower cutoff frequency of hydrophone's bandpass filter
    fmax (float): higher cutoff frequency of hydrophones bandpass filter
    print_exceptions (bool): whether or not exeptions are printed in the terminal line

    return (obspy.core.stream.Stream): obspy Stream object containing one Trace and date
        between start_time and end_time. Returns None if no data are available for specified time frame

    '''
    
    # get URLs
    day_start = UTCDateTime(start_time.year, start_time.month, start_time.day, 0, 0, 0)
    data_url_list = _web_crawler_noise(day_start.strftime("/%Y/%m/%d/"), node)
    if data_url_list == None: return None
    
    day_start = day_start + 24*3600
    while day_start < end_time:
        data_url_list.extend(_web_crawler_noise(day_start.strftime("/%Y/%m/%d/"), node))
        day_start = day_start + 24*3600
    
    # if too many files for one day -> skip day (otherwise program takes too long to terminate)
    if len(data_url_list) > 1000:
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
        if (utc_time_url_start >= start_time and utc_time_url_start < end_time) or \
            (utc_time_url_stop >= start_time and utc_time_url_stop < end_time) or  \
            (utc_time_url_start <= start_time and utc_time_url_stop >= end_time):
            
            try:
                st = read(data_url_list[i],apply_calib=True)
            except:
                if print_exceptions: print("Data are broken")
                return None
            
            # slice stream to get desired data
            st = st.slice(UTCDateTime(start_time), UTCDateTime(end_time))
            
            if st_all == None: st_all = st
            else: 
                st_all += st
                st_all.merge(fill_value ='interpolate',method=1)
                
    if st_all != None:
        if len(st_all) == 0:
            if print_exceptions: print('No data available for selected time frame.')
            return None

    try:
        st_all = st_all.split()
        st_all = st_all.filter("bandpass", freqmin=fmin, freqmax=fmax)
        return st_all
    except:
        if st_all == None:
            if print_exceptions: print('No data available for selected time frame.')
            return None
        else: 
            if print_exceptions: print('Other exception')
            return None

def compute_spectrogram(start_time, end_time, node='/LJ01D', win='hann', L=4096, avg_time=None, overlap=0.5, fmin=20.0, fmax=30000.0):
    '''
    Compute spectrogram of acoustic signal. For each time step of the spectrogram either a modified periodogram (avg_time=None)
    or a power spectral density estimate using Welch's method is computed.

    start_time (datetime.datetime): time of the first noise sample
    end_time (datetime.datetime): time of the last noise sample
    node (str): hydrophone
    fmin (float): lower cutoff frequency of hydrophone's bandpass filter
    fmax (float): higher cutoff frequency of hydrophones bandpass filter
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
    time_specgram = []
           
    # get noise data for entire time period
    noise = get_noise_data(start_time, end_time, node=node)
        
    if noise == None:
        return np.array([]), np.array([]), np.array([])

    # sampling frequency
    fs = noise[0].stats.sampling_rate

    # number of time steps
    if avg_time == None: K = int(len(noise[0].data) / L)
    else: K = int(np.ceil(len(noise[0].data) / (avg_time * fs)))

    # compute spectrogram. For avg_time=None (periodogram for each time step), the last data samples are ignored if 
    # len(noise[0].data) != k * L
    if avg_time == None:
        for k in range(K - 1):
            f, Pxx = signal.periodogram(x = noise[0].data[k*L:(k+1)*L], fs=fs, window=win)
            if len(Pxx) != int(L/2)+1:
                return np.array([]), np.array([]), np.array([])
            else:
                Pxx = 10*np.log10(Pxx * np.power(10, _freq_dependent_sensitivity_correct(int(L/2 + 1))/10))-128.9
                specgram.append(Pxx)
                time_specgram.append(start_time + datetime.timedelta(seconds=k*L / fs))

    else:
        for k in range(K - 1):
            f, Pxx = signal.welch(x = noise[0].data[k*int(fs*avg_time):(k+1)*int(fs*avg_time)],
                fs=fs, window=win, nperseg=L, noverlap = int(L * overlap), nfft=L, average='median')
            if len(Pxx) != int(L/2)+1:
                return np.array([]), np.array([]), np.array([])
            else:
                Pxx = 10*np.log10(Pxx * np.power(10, _freq_dependent_sensitivity_correct(int(L/2 + 1))/10))-128.9
                specgram.append(Pxx)
                time_specgram.append(start_time + datetime.timedelta(seconds=k*avg_time))

        # compute PSD for residual segment if segment has more than L samples
        if len(noise[0].data[int((K - 1) * fs * avg_time):]) >= L:
            f, Pxx = signal.welch(x = noise[0].data[int((K - 1) * fs * avg_time):],
                fs=fs, window=win, nperseg=L, noverlap = int(L * overlap), nfft=L, average='median')
            if len(Pxx) != int(L/2)+1:
                return np.array([]), np.array([]), np.array([])
            else:
                Pxx = 10*np.log10(Pxx * np.power(10, _freq_dependent_sensitivity_correct(int(L/2 + 1))/10))-128.9
                specgram.append(Pxx)
                time_specgram.append(start_time + datetime.timedelta(seconds=(K-1)*avg_time))
 
    if len(time_specgram) == 0:
        return np.array([]), np.array([]), np.array([])
    else:
        return time_specgram, f, specgram

def compute_spectrogram_mp(start_time, end_time, split, n_process=None, node='/LJ01D', win='hann', L=4096,
    avg_time=None, overlap=0.5, fmin=20.0, fmax=30000.0):
    '''
    Same as function compute_spectrogram but using the multiprocessing library. This function is intended to
    be used when analyzing a large amount of data.

    start_time (datetime.datetime): time of the first noise sample
    end_time (datetime.datetime): time of the last noise sample
    split (float or [datetime.datetime]): time period between start_time and end_time is split into parts of length
        split seconds (if float). The last segment can be shorter than split seconds. Alternatively split can be set as
        an list with N start-end time tuples where N is the number of segments. 
    n_process (int): number of processes in the pool. Default (None) means that n_process is equal to the number
        of CPU cores.
    node (str): hydrophone
    fmin (float): lower cutoff frequency of hydrophone's bandpass filter
    fmax (float): higher cutoff frequency of hydrophones bandpass filter
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
    if type(split) == float or type(split) == int:
        n_seg = int(np.ceil((end_time - start_time).total_seconds() / split))
        start_end_list = []
        for k in range(n_seg - 1):
            start_end_list.append((start_time + k * datetime.timedelta(seconds=split),
                start_time + (k+1) * datetime.timedelta(seconds=split), node, win, L, avg_time, overlap, fmin, fmax))
        start_end_list.append((start_time + (n_seg-1) * datetime.timedelta(seconds=split), end_time,
            node, win, L, avg_time, overlap, fmin, fmax))
    else:
        start_end_list = []
        for row in start_end_list:
            start_end_list.append((row[0], row[1], node, win, L, avg_time, overlap, fmin, fmax))

    with mp.get_context("spawn").Pool(n_process) as p:
        try:
            specgram_list = p.starmap(compute_spectrogram, start_end_list)
            ## concatenate all small spectrograms to obtain final spectrogram
            specgram = []
            time_specgram = []
            f = specgram_list[0][1]
            for i in range(len(specgram_list)):
                time_specgram.extend(np.array(specgram_list[i][0]))
                specgram.extend(specgram_list[i][2])
        except:
            return np.array([]), np.array([]), np.array([])

    if len(time_specgram) == 0:
        return np.array([]), np.array([]), np.array([])
    else:
        return np.array(time_specgram), f, np.array(specgram)

# TODO: allow for visualization of ancillary data. Create SpecgramVisu class?
def visualize_spectrogram(spectrogram, t=[], f=[], plot_spec=True, save_spec=False,
    filename='spectrogram.png', title='spectrogram', xlabel='time', xlabel_rot=70, ylabel='frequency',
    fmin=0, fmax=32, vmin=20, vmax=80, vdelta=1.0, vdelta_cbar=5, figsize=(16,9), dpi=96):
    '''
    Basic visualization of spectrogram based on matplotlib. The function offers two options: Plot spectrogram
    in Python (plot_spec = True) and save specrogram plot in directory (save_spec = True). Spectrograms are
    plotted in dB re 1µ Pa^2/Hz.

    spectrogram (N x M numpy.array(float)): numpy array with N rows (one row for each time step) and M columns
        (one column for each frequency value)
    t (N x 1 array like): indices on horizontal (time) axis.
    f (M x 1 array like): indices of vertical (frequency) axis.
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

    if len(t) != len(spectrogram):
        t = np.linspace(0, len(spectrogram) - 1, len(spectrogram))
    if len(f) != len(spectrogram[0]):
        f = np.linspace(0, len(spectrogram[0]) - 1, len(spectrogram[0]))

    cbarticks = np.arange(vmin,vmax+vdelta,vdelta)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.contourf(t, f, np.transpose(spectrogram), cbarticks, norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=plt.cm.jet)  
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

def save_spectrogram(spectrogram, t=None, f=None, filename='spectrogram.pickle'):
    '''
    Save spectrogram in pickle file.

    spectrogram (N x M numpy.array(float)): numpy array with N rows (one row for each time step) and M columns
        (one column for each frequency value)
    t (N x 1 array like): indices on horizontal (time) axis.
    f (M x 1 array like): indices of vertical (frequency) axis.
    filename (str): directory where spectrogram data is saved. Ending has to be ".pickle".
    '''

    dct = {
        't': t,
        'f': f,
        'spectrogram': spectrogram
        }
    with open(filename, 'wb') as outfile:
        pickle.dump(dct, outfile)

def compute_psd_welch(start_time, end_time, node='/LJ01D', win='hann', L=4096, avg_time=None, overlap=0.5,
    avg_method='median', fmin=20.0, fmax=30000.0):
    '''
    Compute power spectral density estimates using Welch's method.

    start_time (datetime.datetime): time of the first noise sample
    end_time (datetime.datetime): time of the last noise sample
    node (str): hydrophone
    fmin (float): lower cutoff frequency of hydrophone's bandpass filter
    fmax (float): higher cutoff frequency of hydrophones bandpass filter
    win (str): window function used to taper the data. Default is Hann-window. See scipy.signal.get_window for a list of
        possible window functions
    L (int): length of each data block for computing the FFT
    avg_time (float): time in seconds that is covered in one time step of the spectrogram. Default value is None and one
        time step covers L samples. If signal covers a long time period it is recommended to use a higher value for avg_time
        to avoid memory overflows and facilitate visualization.
    overlap (float): percentage of overlap between adjecent blocks if Welch's method is used. Parameter is ignored if
        avg_time is None.
    avg_method (str): method for averaging when using Welch's method. Either 'mean' or 'median' can be used

    return ([float], [float]): tuple including time, and spectral level. If no noise date is available,
        function returns two empty numpy arrays.
    '''
    # get noise data segment for each entry in rain_event
    # each noise data segemnt contains usually 1 min of data
    noise = get_noise_data(start_time, end_time, node)
    if noise == None:
        return np.array([]), np.array([])
    fs = noise[0].stats.sampling_rate

    # compute Welch median for entire data segment
    f, Pxx = signal.welch(x = noise[0].data, fs = fs, window=win, nperseg=L, noverlap = int(L * overlap),
        nfft=L, average=avg_method)

    if len(Pxx) != int(L/2) + 1:
        return np.array([]), np.array([])
    else:
        Pxx = 10*np.log10(Pxx*np.power(10, _freq_dependent_sensitivity_correct(int(L/2 + 1))/10))-128.9
    
    return f, Pxx

def compute_psd_welch_mp(start_time, end_time, split, n_process=None, node='/LJ01D', win='hann', L=4096, avg_time=None, overlap=0.5,
    avg_method='median', fmin=20.0, fmax=30000.0):
    '''
    Same as compute_psd_welch but using the multiprocessing library.

    start_time (datetime.datetime): time of the first noise sample
    end_time (datetime.datetime): time of the last noise sample
    split (float or [datetime.datetime]): time period between start_time and end_time is split into parts of length
        split seconds (if float). The last segment can be shorter than split seconds. Alternatively split can be set as
        an list with N start-end time tuples where N is the number of segments. 
    n_process (int): number of processes in the pool. Default (None) means that n_process is equal to the number
        of CPU cores.
    node (str): hydrophone
    fmin (float): lower cutoff frequency of hydrophone's bandpass filter
    fmax (float): higher cutoff frequency of hydrophones bandpass filter
    win (str): window function used to taper the data. Default is Hann-window. See scipy.signal.get_window for a list of
        possible window functions
    L (int): length of each data block for computing the FFT
    avg_time (float): time in seconds that is covered in one time step of the spectrogram. Default value is None and one
        time step covers L samples. If signal covers a long time period it is recommended to use a higher value for avg_time
        to avoid memory overflows and facilitate visualization.
    overlap (float): percentage of overlap between adjecent blocks if Welch's method is used. Parameter is ignored if
        avg_time is None.
    avg_method (str): method for averaging when using Welch's method. Either 'mean' or 'median' can be used

    return ([float], [float]): tuple including frequency indices and PSD estimates, where each PSD estimate. The PSD estimate
        of each segment is stored in a separate row. 
    '''

    # create array with N start and end time values
    if type(split) == float or type(split) == int:
        n_seg = int(np.ceil((end_time - start_time).total_seconds() / split))
        start_end_list = []
        for k in range(n_seg - 1):
            start_end_list.append((start_time + k * datetime.timedelta(seconds=split),
                start_time + (k+1) * datetime.timedelta(seconds=split), node, win, L, avg_time, overlap, avg_method, fmin, fmax))
        start_end_list.append((start_time + (n_seg-1) * datetime.timedelta(seconds=split), end_time,
            node, win, L, avg_time, overlap, avg_method, fmin, fmax))
    else:
        start_end_list = []
        for row in split:
            start_end_list.append((row[0], row[1], node, win, L, avg_time, overlap, avg_method, fmin, fmax))

    with mp.get_context("spawn").Pool(n_process) as p:
        psd_list = p.starmap(compute_psd_welch, start_end_list)
        f = psd_list[0][0]
        psds = []
        for i in range(len(psd_list)):
            psds.append(np.array(psd_list[i][1]))

    return f, np.array(psds)

def save_psd(psd, f=[], filename='psd.json', ancillary_data=[], ancillary_data_label=[]):
    '''
    Save PSD estimates along with with ancillary data (stored in dictionary) in json file.

    psd ([float]): power spectral density estimate
    f ([float]): frequency indices
    filename (str): directory for saving the data
    ancillary_data ([array like]): list of ancillary data
    ancillary_data_label ([str]): labels for ancillary data used as keys in the output dictionary.
        Array has same length as ancillary_data array.
    '''

    if len(f) != len(psd):
        f = np.linspace(0, len(psd)-1, len(psd))

    if type(psd) != list:
        psd = psd.tolist()

    if type(f) != list:
        f = f.tolist()

    dct = {
        'psd': psd,
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

def visualize_psd(psd, f=[], plot_psd=True, save_psd=False,
    filename='psd.png', title='PSD', xlabel='frequency', xlabel_rot=0, ylabel='spectral level',
    fmin=0, fmax=32, vmin=20, vmax=80, figsize=(16,9), dpi=96):


    '''
    Basic visualization of PSD estimate based on matplotlib. The function offers two options: Plot PSD
    in Python (plot_psd = True) and save PSD plot in directory (save_psd = True). PSDs are
    plotted in dB re 1µ Pa^2/Hz.

    psd (numpy.array(float)): PSD values
    f (array like): indices of vertical (frequency) axis.
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

    if len(f) != len(psd):
        f = np.linspace(0, len(psd) - 1, len(psd))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.semilogx(f, psd)  
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