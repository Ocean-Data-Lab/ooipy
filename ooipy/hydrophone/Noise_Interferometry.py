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

cwd = os.getcwd()
ooipy_dir = os.path.dirname(os.path.dirname(cwd))
sys.path.append(ooipy_dir)

from ooipy.request import hydrophone


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
        x = np.cos(psi1)*np.sin(psi2) - np.sin(psi1)*np.cos(psi2)*np.cos(del_lambda);

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
        node1_data = hydrophone.get_acoustic_data_conc(start_time, end_time, node=self.node1, verbose=self.verbose, data_gap_mode=2)
        
        if verbose: print('Getting Audio from Node 2...')

        #Audio from Node 2
        node2_data = hydrophone.get_acoustic_data_conc(start_time, end_time, node=self.node2, verbose=self.verbose, data_gap_mode=2)
        
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

        EQ_lon = [-126.3030]
        EQ_lat = [40.454]
        fig.add_trace(go.Scattergeo(
            lon = EQ_lon,
            lat = EQ_lat,
            #hoverinfo = ['Earth Quake Site'],
            mode = 'markers',
            marker = dict(
                size = 5,
                color = 'rgb(148, 0, 211)',
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
    
    def get_bearing_angle(self, xcorr, t):
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

        return bearing_max_global

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

# Implement Hydrophone_Xcorr Class as list of functions


def calculate_NCF(node1, node2, avg_time, start_time, loop, count, W=30, verbose=True, filter_data=True):
    
    stopwatch_start = time.time()
    count = 0
    h1_data, h2_data, Fs, flag = get_audio(start_time, avg_time, node1, node2, verbose=verbose, W=W)
    h1_processed, h2_processed = preprocess_audio(h1_data, h2_data, filter_data=filter_data, verbose=True, Fs=Fs, W=W, avg_time=avg_time)
    calc_xcorr(h1_processed, h2_processed, verbose=True, count=count, avg_time=avg_time, loop=loop)
    stopwatch_end = time.time()
    print(f'   Time to Calculate NCF for 1 Average Period: {stopwatch_end-stopwatch_start} \n\n')

def get_audio(start_time, avg_time, node1, node2, verbose=True, W=30):
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
        
        avg_time_seconds = avg_time * 60
        
        if avg_time_seconds % W != 0:
            print('Error: Average Time Must Be Interval of Window')
            return None
        
        # Calculate end_time
        end_time = start_time + timedelta(minutes=avg_time)

        if verbose: print('   Getting Audio from Node 1...')
        
        #Audio from Node 1
        node1_data = hydrophone.get_acoustic_data_conc(start_time, end_time, node=node1, verbose=False, data_gap_mode=2)
        
        if verbose: print('   Getting Audio from Node 2...')

        #Audio from Node 2
        node2_data = hydrophone.get_acoustic_data_conc(start_time, end_time, node=node2, verbose=False, data_gap_mode=2)
        
        if (node1_data == None) or (node2_data == None):
            print('Error with Getting Audio')
            return None, None, None
        
        #Combine Data into Stream
        data_stream = obspy.Stream(traces=[node1_data, node2_data])
        
        if data_stream[0].data.shape != data_stream[1].data.shape:
            print('Data streams are not the same length. Flag to be added later')
            # TODO: Set up flag structure of some kind
        
        Fs = node1_data.stats.sampling_rate   

        # Cut off extra points if present
        h1_data = data_stream[0].data[:int(avg_time*60*Fs)]
        h2_data = data_stream[1].data[:int(avg_time*60*Fs)]
        

        return h1_data, h2_data, Fs, flag

def preprocess_audio(h1_data, h2_data, filter_data, W, avg_time, verbose, Fs):
        
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
    if filter_data:
        if verbose: print('   Filtering Data...')

        h1_data = filter_bandpass(h1_data)
        h2_data = filter_bandpass(h2_data)

    #plt.plot(h1_data)
    #plt.plot(h2_data)

    h1_reshaped = np.reshape(h1_data,(int(avg_time*60/W), int(W*Fs)))
    h2_reshaped = np.reshape(h2_data,(int(avg_time*60/W), int(W*Fs)))                  
            
    return h1_reshaped, h2_reshaped
   
def calc_xcorr(h1, h2, verbose, count, avg_time, loop):
    '''
    finds cross correlation over average period and avereages all correlations
    
    Inputs:
    h1 - audio data from hydrophone 1 of shape [avg_time(s)/W(s), W*Fs], 1st axis contains short time NCCF stacked in 0th axis
    h2 - audio data from hydrophone 2 of shape [avg_time(s)/W(s), W*Fs], 1st axis contains short time NCCF stacked in 0th axis
    
    Output :
    avg_xcorr of shape (N) where N = W*Fs
    xcorr - xcorr for every short time window within average period shape [avg_time(s)/W(s), N]
    '''

    M = h1.shape[1]
    N = h2.shape[1]

    xcorr = np.zeros((int(avg_time*60/30),int(N+M-1)))
    
    if verbose: print('   Correlating Data...')
    xcorr = signal.fftconvolve(h1,np.flip(h2,axis=1),'full',axes=1)

    # Normalize Every Short Time Correlation
    xcorr_norm = xcorr / np.max(xcorr,axis=1)[:,np.newaxis]
    
    xcorr_stack = np.sum(xcorr_norm,axis=0)

    if loop:
        #Save Checkpoints for every average period
        filename = './ckpts/ckpt_' + str(count) + '.pkl'
        
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
    
    if loop:
        return
    else:
        return xcorr_stack, xcorr_norm   

def calculate_NCF_loop(num_periods, node1, node2, avg_time, start_time, count, W=30, verbose=True):

    for k in range(num_periods):
        print(f'Calculting NCF for Period {k+1}:')
        calculate_NCF(node1, node2, avg_time, start_time, loop=True, count=k, W=30, verbose=True)

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