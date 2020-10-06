# Import all dependancies
import numpy as np
import os
import sys
from obspy import read,Stream, Trace
from obspy.core import UTCDateTime
from scipy import signal
from scipy import interpolate
import datetime
import time
from thredds_crawler.crawl import Crawl
import multiprocessing as mp
import obspy
import scipy
from datetime import timedelta
import concurrent.futures
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from gwpy.timeseries import TimeSeries
from multiprocessing.pool import ThreadPool

cwd = os.getcwd()
ooipy_dir = os.path.dirname(os.path.dirname(cwd))
sys.path.append(ooipy_dir)
from ooipy.request import hydrophone_request

def calculate_NCF(NCF_object, loop=False, count=None):
    
    #Start Timing
    stopwatch_start = time.time()

    NCF_object = get_audio(NCF_object)
    
    #See if get_audio returned data:
    if NCF_object == None:
        print('   Error with time period. Period Skipped.\n\n')
        return None
    
    NCF_object = preprocess_audio(NCF_object)
    NCF_object = calc_xcorr(NCF_object, loop, count)
    
    #End Timing
    stopwatch_end = time.time()
    print(f'   Time to Calculate NCF for 1 Average Period: {stopwatch_end-stopwatch_start} \n\n')
    if loop==False:
        return NCF_object
    else:
        return None

def get_audio(NCF_object):
    '''
    Get audio from both hydrophone locations from the OOI Raw Data Server.

    Parameters
    ----------
    NCF_object : NCF
        object specifying all details about NCF calculation

    Returns
    -------
    NCF_object : NCF
        object specifying all details about NCF calculation
    '''
    # unpack values from NCF_object
    avg_time = NCF_object.avg_time
    W = NCF_object.W
    start_time = NCF_object.start_time
    node1 = NCF_object.node1
    node2 = NCF_object.node2
    verbose = NCF_object.verbose
    htype = NCF_object.htype

    flag = False
    
    avg_time_seconds = avg_time * 60
    
    if avg_time_seconds % W != 0:
        raise Exception ('Average Time Must Be Interval of Window')
        return None
    
    # Calculate end_time
    end_time = start_time + timedelta(minutes=avg_time)

    if htype == 'broadband':
        if verbose: print('   Getting Audio from Node 1...')

        #Audio from Node 1
        node1_data = hydrophone_request.get_acoustic_data(start_time, end_time, node=node1, verbose=False, data_gap_mode=2)
        
        if verbose: print('   Getting Audio from Node 2...')

        #Audio from Node 2
        node2_data = hydrophone_request.get_acoustic_data(start_time, end_time, node=node2, verbose=False, data_gap_mode=2)
        if node2_data == None:
            return None
        if (node1_data == None) or (node2_data == None):
            print('Error with Getting Audio')
            return None
    elif htype == 'low_frequency':
        if verbose: print('   Getting Audio from Node 1...')

        #Audio from Node 1
        node1_data = hydrophone_request.get_acoustic_data_LF(start_time, end_time, node=node1, verbose=False, zero_mean=True)

        if verbose: print('   Getting Audio from Node 2...')

        #Audio from Node 2
        node2_data = hydrophone_request.get_acoustic_data_LF(start_time, end_time, node=node2, verbose=False, zero_mean=True)
        
        if (node1_data == None) or (node2_data == None):
            print('   Error with Getting Audio')
            return None    

    else:
        raise Exception ('Invalid htype')
    
    #Combine Data into Stream
    data_stream = obspy.Stream(traces=[node1_data, node2_data])
    
    if data_stream[0].data.shape != data_stream[1].data.shape:
        print('   Data streams are not the same length. Flag to be added later')
        return None
    
    Fs = node1_data.stats.sampling_rate   
    NCF_object.Fs = Fs
    # Cut off extra points if present
    h1_data = data_stream[0].data[:int(avg_time*60*Fs)]
    h2_data = data_stream[1].data[:int(avg_time*60*Fs)]

    try:
        h1_reshaped = np.reshape(h1_data,(int(avg_time*60/W), int(W*Fs)))
        h2_reshaped = np.reshape(h2_data,(int(avg_time*60/W), int(W*Fs))) 
    except:
        NCF_object.length_flag = True
        return NCF_object

    NCF_object.node1_data = h1_reshaped
    NCF_object.node2_data = h2_reshaped

    return NCF_object

def preprocess_audio_single_thread(h1_data, Fs, filter_cutoffs, whiten):
    '''
    Frequency whiten and filter data from single hydrophone.

    Parameters
    ----------
    h1_data : numpy array
        audio data from either node for single window length
    Fs : float
        sampling frequency in Hz
    filter_cuttoffs : list
        corners of bandpass filter
    whiten : bool
        indicates whether to whiten the spectrum

    Returns
    -------
    h1_data_processes : numpy array
        h1_data after preprocessing

    '''     
    ts = TimeSeries(h1_data, sample_rate=Fs)
    if whiten: ts = ts.whiten()
    ts = ts.bandpass(filter_cutoffs[0], filter_cutoffs[1])
    
    
    h1_data_processed = ts.value
            
    return h1_data_processed

def preprocess_audio(NCF_object):
    h1_data = NCF_object.node1_data
    h2_data = NCF_object.node2_data
    W = NCF_object.W
    Fs = NCF_object.Fs
    verbose = NCF_object.verbose
    whiten = NCF_object.whiten
    filter_cutoffs = NCF_object.filter_cutoffs

    preprocess_input_list_node1 = []
    preprocess_input_list_node2 = []
    for k in range(h1_data.shape[0]):
        short_time_input_list_node1 = [h1_data[k,:], Fs, filter_cutoffs, whiten]
        short_time_input_list_node2 = [h2_data[k,:], Fs, filter_cutoffs, whiten]

        preprocess_input_list_node1.append(short_time_input_list_node1)
        preprocess_input_list_node2.append(short_time_input_list_node2)
    
    with ThreadPool(processes=mp.cpu_count()) as pool:

    
    #pool = ThreadPool(processes=mp.cpu_count())
        if verbose: print('   Filtering and Whitening Data for Node 1...')
        processed_data_list_node1 = pool.starmap(preprocess_audio_single_thread, preprocess_input_list_node1)
        if verbose: print('   Filtering and Whitening Data for Node 2...')
        processed_data_list_node2 = pool.starmap(preprocess_audio_single_thread, preprocess_input_list_node2)
    
    node1_processed_data = np.array(processed_data_list_node1)
    node2_procesesd_data = np.array(processed_data_list_node2)

    NCF_object.node1_processed_data = node1_processed_data
    NCF_object.node2_processed_data = node2_procesesd_data

    return NCF_object

def calc_xcorr(NCF_object, loop=False, count=None):
    # Unpack needed values from NCF_object
    h1 = NCF_object.node1_processed_data
    h2 = NCF_object.node2_processed_data
    avg_time = NCF_object.avg_time
    verbose = NCF_object.verbose

    #Build input list for multiprocessing map
    xcorr_input_list = []
    for k in range(h1.shape[0]):
        single_short_time_input = [h1[k,:], h2[k,:]]
        xcorr_input_list.append(single_short_time_input)

    pool = ThreadPool(processes=mp.cpu_count())
    if verbose: print('   Correlating Data...')
    xcorr_list = pool.starmap(calc_xcorr_single_thread, xcorr_input_list)

    xcorr = np.array(xcorr_list)
    
    xcorr_stack = np.sum(xcorr,axis=0)

    if loop:
        #Save Checkpoints for every average period
        filename = './ckpts/ckpt_' + str(count) + '.pkl'
        
        try:
            with open(filename,'wb') as f:
                #pickle.dump(xcorr_short_time, f)    #Short Time XCORR for all of avg_perd
                pickle.dump(xcorr_stack, f)               #Accumulated xcorr
                #pickle.dump(k,f)                    #avg_period number
        except:
            os.makedirs('ckpts')
            with open(filename,'wb') as f:
                #pickle.dump(xcorr_short_time, f)
                pickle.dump(xcorr_stack, f)
                #pickle.dump(k,f)
    
        return None
    NCF_object.NCF = xcorr_stack
    return NCF_object

def calc_xcorr_single_thread(h1, h2):
    '''
    Calculate single short time correlation of h1 and h2. fftconvolve is used for slightly faster performance:

    Parameters
    ----------
    h1 : numpy array
        with shape [M,]. Contains time series of processed acoustic data from node 1
    h2 : numpy array
        with shape [N,]. contains time series of processed acoustic data form node 2

    Returns
    -------
    xcorr : numpy array
        with shape [M+N-1,]. Contains crosscorrelation of h1 and h2
    ''' 

    xcorr = signal.fftconvolve(h1,np.flip(h2,axis=0),'full',axes=0)

    # normalize single short time correlation
    xcorr_norm = xcorr/np.max(xcorr)

    return xcorr_norm



def calculate_NCF_loop(num_periods, node1, node2, avg_time, start_time, W,  filter_cutoffs, verbose=True, whiten=True, htype='broadband', kstart=0):

    #Header File Just Contains NCF object
    if kstart == 0:
        NCF_object = NCF(avg_time, start_time, node1, node2, filter_cutoffs, W, verbose, whiten, htype, num_periods)
        filename = './ckpts/0HEADER.pkl'
        try:
            with open(filename,'wb') as f:
                pickle.dump(NCF_object, f)               
        except:
            os.makedirs('ckpts')
            with open(filename,'wb') as f:
                pickle.dump(NCF_object, f)
                
    for k in range(kstart,num_periods):
        start_time_loop = start_time + timedelta(minutes=(avg_time*k))
        NCF_object = NCF(avg_time, start_time_loop, node1, node2, filter_cutoffs, W, verbose, whiten, htype)
        print(f'Calculting NCF for Period {k}: {start_time_loop} - {start_time_loop+timedelta(minutes=avg_time)}')
        calculate_NCF(NCF_object, loop=True, count=k)

    return

def filter_bandpass(data, Wlow=15, Whigh=25):
    
    #make data zero mean
    data = data - np.mean(data)
    # decimate by 4
    data_ds_4 = scipy.signal.decimate(data,4)

    # decimate that by 8 for total of 32
    data_ds_32 = scipy.signal.decimate(data_ds_4,8)
    # sampling rate = 2000 Hz: Nyquist rate = 1000 Hz

    N = 4

    #HARDCODED TODO: MAKE NOT HARDCODED
    fs = 64000/32
    b,a = signal.butter(N=N, Wn=[Wlow, Whigh], btype='bandpass',fs=fs)

    data_filt_ds= scipy.signal.lfilter(b,a,data_ds_32)

    data_filt = scipy.signal.resample(data_filt_ds ,data.shape[0])

    return(data_filt)

def freq_whiten(x, Fs):
    '''
    Whiten time series data. Python package GWpy utilized for this function

    Parameters
    ----------
    x : numpy array
        array containing time series data to be whitened
    Fs : float
        sampling frequency of the time series array x

    Returns
    -------
    x_new : numpy array
        array containing frequency whitened time series data
    '''

    series = TimeSeries(x, sample_rate=Fs)
    white = series.whiten()
    x_new = white.value
    return x_new


class NCF:
    '''
    Object that stores NCF Data

    Attributes
    ----------
    avg_time : float
        length of single NCF average period in minutes
    start_time : datetime.datetime
        indicates the time that the NCF begins
    node1 : string
        node location for hydrophone 1
    node1 : string
        node location for hydrophone 2
    filter_corner : numpy array
        indicates low and high corner frequencies for implemented butterworth bandpass filter. Should be shape [2,]
    W : float
        indicates short time correlation window in seconds
    node1_data : HydrophoneData
        raw data downloaded from ooi data server for hydrophone 1. Data has shape [avg_time/W, W*Fs] and is a verticle
        stack of short time series' of length W (in seconds)
    node2_data : HydrophoneData
        raw data downloaded from ooi data server for hydrophone 2. Data has shape [avg_time/W, W*Fs] and is a verticle
        stack of short time series' of length W (in seconds)
    node1_processed_data : numpy array
        preprocessed data for hydrophone 1. This includes filtering, normalizing short time correlations and frequency whitening
    node2_processed_data : numpy array
        preprocessed data for hydrophone 2. This includes filtering, normalizing short time correlations and frequency whitening
    NCF : numpy array
        average noise correlation function over avg_time
    verbose : boolean
        specifies whether to print supporting information
    Fs : float
        sampling frequency of data
    whiten : bool
        indicates whether to whiten data or not
    htype : str
        specifices the type of hydrophone that is used. options include, 'broadband' and 'low_frequency'
    num_periods : float
        number of average periods looped through. This attribute exists only for the header file.
    length_flag : bool
        set if length of data does not match between hydrophones.
    '''
    
    def __init__(self, avg_time, start_time, node1, node2, filter_cutoffs, W, verbose=False, whiten=True, htype='broadband', num_periods=None):
        self.avg_time = avg_time
        self.start_time = start_time
        self.node1 = node1
        self.node2 = node2
        self.filter_cutoffs = filter_cutoffs
        self.W = W
        self.verbose = verbose
        self.whiten = whiten
        self.htype = htype
        self.num_periods = num_periods
        self.length_flag = False
        return


# Archive

class Hydrophone_Xcorr:


    def __init__(self, node1, node2, avg_time, W=30, verbose=True, filter_data=True):
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
        filter_data : bool
            indicates whether to filter the data with bandpass with cutofss [10, 1k]
        mp : bool
            indicates if multiprocessing functions should be used
       
        Private Attributes
        ------------------
        None at this time

        Methods
        -------
        distance_between_hydrophones(self, coord1, coord2)
            Calculates the distance in meteres between hydrophones
        get_audio(self, start_time)
            Pulls avg_period amount of data from server
        xcorr_over_avg_period(self, h1, h2)
            Computes cross-correlation for window of length W, averaged over avg_period
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
        self.Fs = 64000
        self.Ts = 1/self.Fs
        self.filter_data = filter_data
        
        
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
        x = np.cos(psi1)*np.sin(psi2) - np.sin(psi1)*np.cos(psi2)*np.cos(del_lambda)

        theta_bearing_rad = np.arctan2(y,x)
        theta_bearing_d_1_2 = (np.rad2deg(theta_bearing_rad)+360) % 360

        self.theta_bearing_d_1_2 = theta_bearing_d_1_2

    def get_audio(self, start_time):

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
        
        flag = False
        avg_time = self.avg_time
        verbose = self.verbose
        W = self.W

        
        avg_time_seconds = avg_time * 60
        
        if avg_time_seconds % W != 0:
            print('Error: Average Time Must Be Interval of Window')
            return None
        
        # Initialze Two Classes for Two Hydrophones
        #self.ooi1 = OOIHydrophoneData(limit_seed_files=False, print_exceptions=True, data_gap_mode=2)
        #self.ooi2 = OOIHydrophoneData(limit_seed_files=False, print_exceptions=True, data_gap_mode=2)

        # Calculate end_time
        end_time = start_time + timedelta(minutes=avg_time)

        if verbose: print('Getting Audio from Node 1...')
        stopwatch_start = time.time()
        
        #Audio from Node 1
        node1_data = request.hydrophone.get_acoustic_data(start_time, end_time, node=self.node1, verbose=self.verbose, data_gap_mode=2)
        
        if verbose: print('Getting Audio from Node 2...')

        #Audio from Node 2
        node2_data = request.hydrophone.get_acoustic_data(start_time, end_time, node=self.node2, verbose=self.verbose, data_gap_mode=2)
        
        if (node1_data == None) or (node2_data == None):
            print('Error with Getting Audio')
            return None, None, None
        
        #Combine Data into Stream
        data_stream = obspy.Stream(traces=[node1_data, node2_data])
        
        stopwatch_end = time.time()
        print('Time to Download Data from Server: ',stopwatch_end-stopwatch_start)
        
        if data_stream[0].data.shape != data_stream[1].data.shape:
            print('Data streams are not the same length. Flag to be added later')
            # TODO: Set up flag structure of some kind
                      
        # Cut off extra points if present
        h1_data = data_stream[0].data[:avg_time*60*self.Fs]
        h2_data = data_stream[1].data[:avg_time*60*self.Fs]
        
        return h1_data, h2_data, flag

    def preprocess_audio(self, h1_data, h2_data):
            
        #Previous Fix for data_gap, Recklessly added zeros
        '''    
        if ((h1_data.shape[0] < avg_time*60*self.Fs)):
            print('Length of Audio at node 1 too short, zeros added. Length: ', data_stream[0].data.shape[0])
            h1_data = np.pad(h1_data, (0, avg_time*60*self.Fs-data_stream[0].data.shape[0]))

        if ((h2_data.shape[0] < avg_time*60*self.Fs)):
            print('Length of Audio at node 2 too short, zeros added. Length: ', data_stream[1].data.shape[0])
            h2_data = np.pad(h2_data, (0, avg_time*60*self.Fs-data_stream[1].data.shape[0]))
        '''

        # Filter Data
        if self.filter_data:
            if self.verbose: print('Filtering Data...')

            h1_data = self.filter_bandpass(h1_data)
            h2_data = self.filter_bandpass(h2_data)
        self.data_node1 = h1_data
        self.data_node2 = h2_data

        plt.plot(h1_data)
        plt.plot(h2_data)

        h1_reshaped = np.reshape(h1_data,(int(self.avg_time*60/self.W), int(self.W*self.Fs)))
        h2_reshaped = np.reshape(h2_data,(int(self.avg_time*60/self.W), int(self.W*self.Fs)))                  
              
        return h1_reshaped, h2_reshaped
    
    def xcorr_over_avg_period(self, h1, h2, loop=True):
        '''
        finds cross correlation over average period and avereages all correlations
        
        Inputs:
        h1 - audio data from hydrophone 1 of shape [avg_time(s)/W(s), W*Fs], 1st axis contains short time NCCF stacked in 0th axis
        h2 - audio data from hydrophone 2 of shape [avg_time(s)/W(s), W*Fs], 1st axis contains short time NCCF stacked in 0th axis
        
        Output :
        avg_xcorr of shape (N) where N = W*Fs
        xcorr - xcorr for every short time window within average period shape [avg_time(s)/W(s), N]
        '''
        verbose = self.verbose
        avg_time = self.avg_time
        M = h1.shape[1]
        N = h2.shape[1]

        xcorr = np.zeros((int(avg_time*60/30),int(N+M-1)))

        stopwatch_start = time.time()
        #if verbose:
        #    bar = progressbar.ProgressBar(maxval=h1.shape[0], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        #    bar.start()
        
        if self.verbose: print('Correlating Data...')
        xcorr = signal.fftconvolve(h1,np.flip(h2,axis=1),'full',axes=1)

        # Normalize Every Short Time Correlation
        xcorr_norm = xcorr / np.max(xcorr,axis=1)[:,np.newaxis]
        
        xcorr_stack = np.sum(xcorr_norm,axis=0)

        if loop:
            #Save Checkpoints for every average period
            filename = './ckpts/ckpt_' + str(self.count) + '.pkl'
            
            try:
                with open(filename,'wb') as f:
                    #pickle.dump(xcorr_short_time, f)    #Short Time XCORR for all of avg_perd
                    pickle.dump(xcorr_norm, f)               #Accumulated xcorr
                    #pickle.dump(k,f)                    #avg_period number
            except:
                os.makedirs('ckpts')
                with open(filename,'wb') as f:
                    #pickle.dump(xcorr_short_time, f)
                    pickle.dump(xcorr_norm, f)
                    #pickle.dump(k,f)

        stopwatch_end = time.time()
        print('Time to Calculate Cross Correlation of 1 period: ',stopwatch_end-stopwatch_start)
        if loop:
            return
        else:
            return xcorr_stack, xcorr_norm   
 
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
        self.count = 0

        for k in range(num_periods):
            stopwatch_start = time.time()
            if verbose: print('\n\nTime Period: ',k + 1)
            
            h1, h2, flag = self.get_audio(start_time)

            if flag == None:
                print(f'{k+1}th average period skipped, no data available')
                continue
            
            h1_processed, h2_processed = self.preprocess_audio(h1,h2)
            
            self.xcorr_over_avg_period(h1_processed, h2_processed)

            self.count=self.count+1
            '''
            # Compute Cross Correlation for Each Window and Average
            if first_loop:
                xcorr_avg_period, xcorr_short_time = self.xcorr_over_avg_period(h1_processed, h2_processed)
                xcorr = xcorr_avg_period
                first_loop = False
            else:
                xcorr_avg_period, xcorr_short_time = self.xcorr_over_avg_period(h1_processed, h2_processed)
                xcorr += xcorr_avg_period
                start_time = start_time + timedelta(minutes=self.avg_time)
            
            stopwatch_end = time.time()
            print('Time to Complete 1 period: ',stopwatch_end - stopwatch_start)
            
            #Save Checkpoints for every average period
            filename = './ckpts/ckpt_' + str(k) + '.pkl'
            
            if self.ckpts:
                try:
                    with open(filename,'wb') as f:
                        #pickle.dump(xcorr_short_time, f)    #Short Time XCORR for all of avg_perd
                        pickle.dump(xcorr_avg_period, f)               #Accumulated xcorr
                        pickle.dump(k,f)                    #avg_period number
                except:
                    os.makedirs('ckpts')
                    with open(filename,'wb') as f:
                        #pickle.dump(xcorr_short_time, f)
                        pickle.dump(xcorr_avg_period, f)
                        pickle.dump(k,f)
            '''
        return None
        '''
            self.count = self.count + 1

            # Calculate time variable TODO change to not calculate every loop
            dt = self.Ts
            self.xcorr = xcorr
            t = np.arange(-np.shape(xcorr)[0]*dt/2,np.shape(xcorr)[0]*dt/2,dt)
            
        #xcorr = xcorr / num_periods

        # Calculate Bearing of Max Peak
        bearing_max_global = self.get_bearing_angle(xcorr, t)

        return t, xcorr, bearing_max_global  
    '''
    def plot_map_bearing(self, bearing_angle):

        coord1 = self.hydrophone_locations[self.node1]
        coord2 = self.hydrophone_locations[self.node2]
        thetaB1 = bearing_angle[0]
        thetaB2 = bearing_angle[1]
        
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
        fig.write_html("21_hr_avg_map.html")

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

    def filter_bandpass(self, data, Wlow=15, Whigh=25):
        
        #make data zero mean
        data = data - np.mean(data)
        # decimate by 4
        data_ds_4 = scipy.signal.decimate(data,4)

        # decimate that by 8 for total of 32
        data_ds_32 = scipy.signal.decimate(data_ds_4,8)
        # sampling rate = 2000 Hz: Nyquist rate = 1000 Hz

        N = 4

        fs = self.Fs/32
        b,a = signal.butter(N=N, Wn=[Wlow, Whigh], btype='bandpass',fs=fs)

        data_filt_ds= scipy.signal.lfilter(b,a,data_ds_32)

        data_filt = scipy.signal.resample(data_filt_ds ,data.shape[0])

        return(data_filt)
    
    def get_bearing_angle(self, t):

        #bearing is with respect to node1 (where node2 is at 0 deg)
        bearing_local = [np.rad2deg(np.arccos(1480*t/self.distance)), -np.rad2deg(np.arccos(1480*t/self.distance))]
        #convert bearing_max_local to numpy array
        bearing_local = np.array(bearing_local)
        #convert to global (NSEW) degrees
        bearing_global = self.theta_bearing_d_1_2 + bearing_local
        #make result between 0 and 360
        bearing_global = bearing_global % 360
        self.bearing_global = bearing_global

        return bearing_global

    def plot_polar_TDOA(self, xcorr, t):
        '''
        plot_polar_TDOA(self, xcorr)

        Inputs:
        xcorr (numpy array) : array of shape [X,] consisting of an averaged cross correlation

        Outputs:
        None
        '''
        
        B = np.arccos(1480*t/self.distance)
        plt.polar(B, xcorr)
        print(type(B))
