'''
This modules provides a toolset for calculating Noise Correlation
Functions (NCFs). To calculate NCF follow the below sequence (this is
also accomplished in the calculate_NCF() function)::
    avg_time = 60  #minutes
    start_time = datetime.datetime(2017,3,10,0,0,0) # time of first sample
    node1 = 'LJ01C'
    node2 = 'PC01A'
    filter_cutoffs = [12, 30]
    W = 90
    htype = 'broadband'
    # Initialize NCF_object
    NCF_object = NCF(avg_time, start_time, node1, node2, filter_cutoffs, W,
                     htype=htype)

    # Calculate NCF
    NCF_object = get_audio(NCF_object)
    NCF_object = preprocess_audio(NCF_object)
    NCF_object = calc_xcorr(NCF_object)

    # Access NCF
    NCF_object.NCF
'''
# Import all dependencies
from ooipy.request import hydrophone_request
import numpy as np
import os
import sys
from scipy import signal
import time
import multiprocessing as mp
from datetime import timedelta
import pickle
from multiprocessing.pool import ThreadPool
import scipy
from matplotlib import pyplot as plt

cwd = os.getcwd()
ooipy_dir = os.path.dirname(os.path.dirname(cwd))
sys.path.append(ooipy_dir)


def calculate_NCF(NCF_object, loop=False, count=None):
    '''
    Do all required signal processing, and calculate average Noise
    Correlation Function (NCF) for given average time.

    Example
    -------
    To calculate and access Noise Correlation Function (NCF) execute
    the following code::
        avg_time = 60  #minutes
        start_time = datetime.datetime(2017,3,10,0,0,0) # time of first sample
        node1 = 'LJ01C'
        node2 = 'PC01A'
        filter_cutoffs = [12, 30]
        W = 90
        htype = 'broadband'
        NCF_object = NCF(avg_time, start_time, node1, node2, filter_cutoffs, W,
        htype=htype)
        calculate_NCF(NCF_object)
        # To access NCF:
        NCF_object.NCF

    Parameters
    ----------
    NCF_object : NCF
        object specifying all details about NCF calculation
    loop : boolean
        specifies whether the NCF is calculated in stand alone or loop
        mode. If you plan to use loop mode, use the function
        calculate_NCF_loop
    count : int
        specifies specific index of loop calculation. Default is None
        for stand alone operation.
    process_method : str
        specifies with method of processing should be used

    Returns
    -------
    NCF_object : NCF
        object specifying all details about NCF calculation
    '''
    sp_method = NCF_object.sp_method
    # Start Timing
    stopwatch_start = time.time()

    NCF_object = get_audio(NCF_object)

    # See if get_audio returned data:
    if NCF_object is None:
        print('   Error with time period. Period Skipped.\n\n')
        return None

    if sp_method == 'sabra':
        # create step by step sp figures onces
        if count == 0:
            NCF_object = sabra_processing(NCF_object, plot=True)
        else:
            NCF_object = sabra_processing(NCF_object, plot=False)

        NCF_object = calc_xcorr(NCF_object, loop, count)
    elif sp_method == 'brown':
        NCF_object = brown_processing(NCF_object)
    elif sp_method == 'bit_normalization':
        if count == 0:
            NCF_object = bit_normalization_method(NCF_object, plot=True)
        else:
            NCF_object = bit_normalization_method(NCF_object, plot=False)

        NCF_object = calc_xcorr(NCF_object, loop, count)

    elif sp_method == 'time_eq':
        if count == 0:
            NCF_object = time_EQ(NCF_object, 51, plot=True)
        else:
            NCF_object = time_EQ(NCF_object, 51, plot=False)

        NCF_object = calc_xcorr(NCF_object, loop, count)
    elif sp_method == 'tdoa':
        NCF_object = TDOA_processing(NCF_object)
        NCF_object = calc_xcorr(NCF_object, loop, count)

    if sp_method == 'sabra_b':
        # create step by step sp figures onces
        if count == 0:
            NCF_object = sabra_processing_b(NCF_object, plot=True)
        else:
            NCF_object = sabra_processing_b(NCF_object, plot=False)

        NCF_object = calc_xcorr(NCF_object, loop, count)

    else:
        raise Exception(f'Invalid Signal Processing Method of: {sp_method}')
    # End Timing
    stopwatch_end = time.time()
    print(f'   Time to Calculate NCF for 1 Average\
        Period: {stopwatch_end - stopwatch_start} \n\n')

    # Save NCF to .pkl file if loop is true
    if loop is True:
        save_avg_period(NCF_object, count=count)
        return None
    if loop is False:
        return NCF_object


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
        object specifying all details about NCF calculation. This function
        adds the NCF_object attributes NCF_object.node1_data and
        NCF_object.node2_data
    '''
    # unpack values from NCF_object
    avg_time = NCF_object.avg_time
    W = NCF_object.W
    start_time = NCF_object.start_time
    node1 = NCF_object.node1
    node2 = NCF_object.node2
    verbose = NCF_object.verbose
    htype = NCF_object.htype

    avg_time_seconds = avg_time * 60

    if avg_time_seconds % W != 0:
        raise Exception('Average Time Must Be Interval of Window')

    # Calculate end_time
    end_time = start_time + timedelta(minutes=avg_time)

    if htype == 'broadband':
        if verbose:
            print('   Getting Audio from Node 1...')

        # Audio from Node 1
        node1_data = hydrophone_request.get_acoustic_data(
            start_time, end_time, node=node1, verbose=False, data_gap_mode=2)

        if verbose:
            print('   Getting Audio from Node 2...')

        # Audio from Node 2
        node2_data = hydrophone_request.get_acoustic_data(
            start_time, end_time, node=node2, verbose=False, data_gap_mode=2)

        if node2_data is None:
            return None
        if (node1_data is None) or (node2_data is None):
            print('Error with Getting Audio')
            return None
    elif htype == 'low_frequency':
        if verbose:
            print('   Getting Audio from Node 1...')

        # Audio from Node 1
        node1_data = hydrophone_request.get_acoustic_data_LF(
            start_time, end_time, node=node1, verbose=False, zero_mean=True)

        if verbose:
            print('   Getting Audio from Node 2...')

        # Audio from Node 2
        node2_data = hydrophone_request.get_acoustic_data_LF(
            start_time, end_time, node=node2, verbose=False, zero_mean=True)

        if (node1_data is None) or (node2_data is None):
            print('   Error with Getting Audio')
            return None

    else:
        raise Exception('Invalid htype')

    if node1_data.data.shape != node2_data.data.shape:
        print('   Data streams are not the same length. Flag to be \
            added later')
        return None

    # Decimate broadband hydrophone to Fs = 2000 for easier processing
    if htype == 'broadband':
        # Decimate by 4
        node1_data_dec = scipy.signal.decimate(
            node1_data.data, 4, zero_phase=True)
        node2_data_dec = scipy.signal.decimate(
            node2_data.data, 4, zero_phase=True)
        # Decimate by 8 (total of 32)
        node1_data_dec = scipy.signal.decimate(
            node1_data_dec, 8, zero_phase=True)
        node2_data_dec = scipy.signal.decimate(
            node2_data_dec, 8, zero_phase=True)
        # save data back into node1_data and node2_data
        node1_data.data = node1_data_dec
        node2_data.data = node2_data_dec
        node1_data.stats.sampling_rate = node1_data.stats.sampling_rate / 32
        node2_data.stats.sampling_rate = node2_data.stats.sampling_rate / 32

    Fs = node1_data.stats.sampling_rate
    NCF_object.Fs = Fs
    # Cut off extra points if present
    h1_data = node1_data.data[:int(avg_time * 60 * Fs)]
    h2_data = node2_data.data[:int(avg_time * 60 * Fs)]

    try:
        h1_reshaped = np.reshape(
            h1_data, (int(avg_time * 60 / W), int(W * Fs)))
        h2_reshaped = np.reshape(
            h2_data, (int(avg_time * 60 / W), int(W * Fs)))
    except Exception:
        NCF_object.length_flag = True
        return NCF_object

    NCF_object.node1_data = h1_reshaped
    NCF_object.node2_data = h2_reshaped

    return NCF_object


def sabra_processing(NCF_object, plot=False):
    '''
    preprocess audio using signal processing method from sabra. This function
    is designed to replace preprocess_audio single thread in the signal
    processing flow.

    Sabra, K. G., Roux, P., and Kuperman, W. A. (2005). “Emergence rate of the
    time-domain Green’s function from the ambient noise cross-correlation
    function,” The Journal of the Acoustical Society of America, 118,
    3524–3531. doi:10.1121/1.2109059

    This includes:
    - filtering data to desired frequency band
    - clipping to 3*std of data of short time noise
    - frequency whitening short time noise
    '''
    node1 = NCF_object.node1_data
    node2 = NCF_object.node2_data

    Fs = NCF_object.Fs
    filter_cutoffs = NCF_object.filter_cutoffs

    # Filter Data to Specific Range (fiter_cutoffs)
    print('   Filtering Data from node 1')
    node1_filt = filter_bandpass(node1, Fs, filter_cutoffs)
    print('   Filtering Data from node 2')
    node2_filt = filter_bandpass(node2, Fs, filter_cutoffs)

    # Create before/after plot of filtering
    if plot:
        plot_sp_step(
            node1[0, :], 'Before Filtering', node1_filt[0, :],
            'After Filtering', Fs, 'sabra')

    # Clip Data to amplitude
    thresh1 = 3 * np.std(np.ndarray.flatten(node1_filt))
    thresh2 = 3 * np.std(np.ndarray.flatten(node2_filt))
    node1_clip = np.clip(node1_filt, -thresh1, thresh1)
    node2_clip = np.clip(node2_filt, -thresh2, thresh2)

    if plot:
        # Create before/after plot of filtering
        plot_sp_step(
            node1_filt[0, :], 'Before Clipping', node1_clip[0, :],
            'After Clipping', Fs, 'sabra')

    # Frequency Whiten Data
    node1_whit = freq_whiten(node1_clip, Fs, filter_cutoffs)
    node2_whit = freq_whiten(node2_clip, Fs, filter_cutoffs)

    if plot:
        # Create before/after plot of filtering
        plot_sp_step(
            node1_clip[0, :], 'Before Whitening', node1_whit[0, :],
            'After Whitening', Fs, 'sabra')

    NCF_object.node1_processed_data = node1_whit
    NCF_object.node2_processed_data = node2_whit

    return NCF_object


def sabra_processing_b(NCF_object, plot=False):
    '''
    preprocess audio using signal processing method from sabra. This function
    is designed to replace preprocess_audio single thread in the signal
    processing flow.

    Sabra, K. G., Roux, P., and Kuperman, W. A. (2005). “Emergence rate of the
    time-domain Green’s function from the ambient noise cross-correlation
    function,” The Journal of the Acoustical Society of America, 118,
    3524–3531. doi:10.1121/1.2109059

    This includes:
    - filtering data to desired frequency band
    - clipping to 3*std of data of short time noise
    - frequency whitening short time noise
    '''
    node1 = NCF_object.node1_data
    node2 = NCF_object.node2_data

    Fs = NCF_object.Fs
    filter_cutoffs = NCF_object.filter_cutoffs

    # Filter Data to Specific Range (fiter_cutoffs)
    print('   Filtering Data from node 1')
    node1_filt = filter_bandpass(node1, Fs, filter_cutoffs)
    print('   Filtering Data from node 2')
    node2_filt = filter_bandpass(node2, Fs, filter_cutoffs)

    # Create before/after plot of filtering
    if plot:
        plot_sp_step(
            node1[0, :], 'Before Filtering', node1_filt[0, :],
            'After Filtering', Fs, 'sabra')

    # Clip Data to amplitude
    thresh1 = 3 * np.std(np.ndarray.flatten(node1_filt))
    thresh2 = 3 * np.std(np.ndarray.flatten(node2_filt))
    node1_clip = np.clip(node1_filt, -thresh1, thresh1)
    node2_clip = np.clip(node2_filt, -thresh2, thresh2)

    if plot:
        # Create before/after plot of filtering
        plot_sp_step(
            node1_filt[0, :], 'Before Clipping', node1_clip[0, :],
            'After Clipping', Fs, 'sabra')

    # Frequency Whiten Data
    node1_whit = freq_whiten_b(node1_clip, Fs, filter_cutoffs)
    node2_whit = freq_whiten_b(node2_clip, Fs, filter_cutoffs)

    if plot:
        # Create before/after plot of filtering
        plot_sp_step(
            node1_clip[0, :], 'Before Whitening', node1_whit[0, :],
            'After Whitening', Fs, 'sabra')

    NCF_object.node1_processed_data = node1_whit
    NCF_object.node2_processed_data = node2_whit

    return NCF_object


def TDOA_processing(NCF_object, plot=False):
    '''
    preprocess audio using signal processing method from sabra (without
    clipping). This function is designed to replace preprocess_audio single
    thread in the signal processing flow.

    Sabra, K. G., Roux, P., and Kuperman, W. A. (2005). “Emergence rate of the
    time-domain Green’s function from the ambient noise cross-correlation
    function,” The Journal of the Acoustical Society of America, 118,
    3524–3531. doi:10.1121/1.2109059

    This includes:
    - filtering data to desired frequency band
    - frequency whitening short time noise
    '''
    node1 = NCF_object.node1_data
    node2 = NCF_object.node2_data

    Fs = NCF_object.Fs
    filter_cutoffs = NCF_object.filter_cutoffs

    # Filter Data to Specific Range (fiter_cutoffs)
    print('   Filtering Data from node 1')
    node1_filt = filter_bandpass(node1, Fs, filter_cutoffs)
    print('   Filtering Data from node 2')
    node2_filt = filter_bandpass(node2, Fs, filter_cutoffs)

    # Create before/after plot of filtering
    if plot:
        plot_sp_step(
            node1[0, :], 'Before Filtering', node1_filt[0, :],
            'After Filtering', Fs, 'sabra')

    # Clip Data to amplitude
    node1_clip = node1_filt
    node2_clip = node2_filt

    if plot:
        # Create before/after plot of filtering
        plot_sp_step(
            node1_filt[0, :], 'Before Clipping', node1_clip[0, :],
            'After Clipping', Fs, 'sabra')

    # Frequency Whiten Data
    node1_whit = freq_whiten(node1_clip, Fs, filter_cutoffs)
    node2_whit = freq_whiten(node2_clip, Fs, filter_cutoffs)

    if plot:
        # Create before/after plot of filtering
        plot_sp_step(
            node1_clip[0, :], 'Before Whitening', node1_whit[0, :],
            'After Whitening', Fs, 'sabra')

    NCF_object.node1_processed_data = node1_whit
    NCF_object.node2_processed_data = node2_whit

    return NCF_object


def brown_processing(NCF_object, plot=False):
    h1 = NCF_object.node1_data
    h2 = NCF_object.node2_data
    Fs = NCF_object.Fs

    filter_cutoffs = NCF_object.filter_cutoffs

    # Filter data and save in processed data attribute of NCF_object
    print('   Filtering Data from Node 1...')
    NCF_object.node1_processed_data = filter_bandpass(h1, Fs, filter_cutoffs)
    print('   Filtering Data from Node 2...')
    NCF_object.node2_processed_data = filter_bandpass(h2, Fs, filter_cutoffs)

    # Create before/after plot of filtering
    if plot:
        plot_sp_step(
            h1[0, :], 'Before Filtering',
            NCF_object.node1_processed_data[0, :], 'After Filtering', Fs,
            'brown')
    NCF_object = calc_xcorr(NCF_object)
    st_NCFs = NCF_object.st_NCFs
    # Whiten short-time NCFs
    print('   Whitening short-time data...')
    NCF_whiten = freq_whiten(st_NCFs, Fs, filter_cutoffs)

    # Create before/after plot of Whitening
    if plot:
        plot_sp_step(
            st_NCFs[0, :], 'Before Whitening',
            NCF_whiten[0, :], 'After Whitening', Fs, 'brown')

    # Normalize short-time NCFs
    max_value = np.max(np.abs(NCF_whiten), axis=1)
    NCF_norm = (NCF_whiten.T / max_value).T

    # Create before/after plot of filtering
    if plot:
        plot_sp_step(
            NCF_whiten[0, :], 'Before Normalization',
            NCF_norm[0, :], 'After Normalization', Fs, 'brown')

    # Add up short-time NCFs
    NCF = np.sum(NCF_norm, axis=0)

    NCF_object.NCF = NCF
    return NCF_object


def bit_normalization_method(NCF_object, plot=False):
    '''
    - filter data to desired frequency band
    - execute 1 bit normalization of zero mean data
    - frequency whiten data
    '''
    node1 = NCF_object.node1_data
    node2 = NCF_object.node2_data

    Fs = NCF_object.Fs
    filter_cutoffs = NCF_object.filter_cutoffs

    # Filter Data to Specific Range (fiter_cutoffs)
    print('   Filtering Data from node 1')
    node1_filt = filter_bandpass(node1, Fs, filter_cutoffs)
    print('   Filtering Data from node 2')
    node2_filt = filter_bandpass(node2, Fs, filter_cutoffs)

    # Create before/after plot of filtering
    if plot:
        plot_sp_step(
            node1[0, :], 'Before Filtering', node1_filt[0, :],
            'After Filtering', Fs, 'bit_norm')

    # Clip Data to amplitude
    node1_clip = np.sign(node1_filt)
    node2_clip = np.sign(node2_filt)

    if plot:
        # Create before/after plot of filtering
        plot_sp_step(
            node1_filt[0, :], 'Before Clipping', node1_clip[0, :],
            'After Clipping', Fs, 'bit_norm')

    # Frequency Whiten Data
    node1_whit = freq_whiten(node1_clip, Fs, filter_cutoffs)
    node2_whit = freq_whiten(node2_clip, Fs, filter_cutoffs)

    if plot:
        # Create before/after plot of filtering
        plot_sp_step(
            node1_clip[0, :], 'Before Whitening', node1_whit[0, :],
            'After Whitening', Fs, 'bit_norm')

    NCF_object.node1_processed_data = node1_whit
    NCF_object.node2_processed_data = node2_whit

    return NCF_object


def time_EQ(NCF_object, N, plot=False):
    '''
    instead of clipping, time_EQ is used
    N specifies moving average size (must be odd)

    preprocess audio using signal processing method from sabra. This function
    is designed to replace preprocess_audio single thread in the signal
    processing flow.

    Sabra, K. G., Roux, P., and Kuperman, W. A. (2005). “Emergence rate of the
    time-domain Green’s function from the ambient noise cross-correlation
    function,” The Journal of the Acoustical Society of America, 118,
    3524–3531. doi:10.1121/1.2109059

    This includes:
    - filtering data to desired frequency band
    - clipping to 3*std of data of short time noise
    - frequency whitening short time noise
    '''
    node1 = NCF_object.node1_data
    node2 = NCF_object.node2_data

    Fs = NCF_object.Fs
    filter_cutoffs = NCF_object.filter_cutoffs

    # Filter Data to Specific Range (fiter_cutoffs)
    print('   Filtering Data from node 1')
    node1_filt = filter_bandpass(node1, Fs, filter_cutoffs)
    print('   Filtering Data from node 2')
    node2_filt = filter_bandpass(node2, Fs, filter_cutoffs)

    # Create before/after plot of filtering
    if plot:
        plot_sp_step(
            node1[0, :], 'Before Filtering', node1_filt[0, :],
            'After Filtering', Fs, 'sabra')

    # Implement Time Equalization
    kernel = 1 / N * np.ones((1, N))
    weight1 = signal.fftconvolve(node1_filt, kernel, 'valid', axes=1)
    weight2 = signal.fftconvolve(node2_filt, kernel, 'valid', axes=1)

    weight1start = (
        np.ones((int(NCF_object.avg_time * 60 / NCF_object.W),
                int((N - 1) / 2))).T * weight1[:, 0]).T
    weight1end = (
        np.ones((int(NCF_object.avg_time * 60 / NCF_object.W),
                int((N - 1) / 2))).T * weight1[:, -1]).T
    weight2start = (
        np.ones((int(NCF_object.avg_time * 60 / NCF_object.W),
                int((N - 1) / 2))).T * weight2[:, 0]).T
    weight2end = (
        np.ones((int(NCF_object.avg_time * 60 / NCF_object.W),
                int((N - 1) / 2))).T * weight2[:, -1]).T

    weight1 = np.hstack((weight1start, weight1, weight1end))
    weight2 = np.hstack((weight2start, weight2, weight2end))

    node1_clip = node1_filt / np.abs(weight1)
    node2_clip = node2_filt / np.abs(weight2)

    if plot:
        # Create before/after plot of filtering
        plot_sp_step(
            node1_filt[0, :], 'Before Clipping', node1_clip[0, :],
            'After Clipping', Fs, 'sabra')

    # Frequency Whiten Data
    node1_whit = freq_whiten(node1_clip, Fs, filter_cutoffs)
    node2_whit = freq_whiten(node2_clip, Fs, filter_cutoffs)

    if plot:
        # Create before/after plot of filtering
        plot_sp_step(
            node1_clip[0, :], 'Before Whitening', node1_whit[0, :],
            'After Whitening', Fs, 'sabra')

    NCF_object.node1_processed_data = node1_whit
    NCF_object.node2_processed_data = node2_whit

    return NCF_object


def plot_sp_step(old, old_title, new, new_title, Fs, method):

    N = len(old)
    t = np.linspace(0, N / Fs, N)
    f = np.linspace(0, Fs, N)
    # new style method 2; use an axes array
    fig1, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(t, old)
    axs[0].set_title('Time Domain ' + old_title, y=1.08)
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[1].plot(t, new)
    axs[1].set_title('Time Domain ' + new_title, y=1.08)
    axs[1].set_xlabel('time (s)')

    fig2, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    axs[0].plot(f, np.abs(scipy.fft.fft(old)))
    axs[0].set_title('Frequency Domain ' + old_title, y=1.08)
    axs[0].set_xlabel('frequency (Hz)')
    axs[0].set_xlim([0, Fs / 2])
    axs[1].plot(f, np.abs(scipy.fft.fft(new)))
    axs[1].set_title('Frequency Domain ' + new_title, y=1.08)
    axs[1].set_xlabel('frequency (Hz)')
    axs[1].set_xlim([0, Fs / 2])

    fig3, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    axs[0].plot(f, np.rad2deg(np.angle(scipy.fft.fft(old))))
    axs[0].set_title('Frequency Domain ' + old_title, y=1.08)
    axs[0].set_xlabel('frequency (Hz)')
    axs[0].set_xlim([0, Fs / 2])
    axs[1].plot(f, np.rad2deg(np.angle(scipy.fft.fft(new))))
    axs[1].set_title('Frequency Domain ' + new_title, y=1.08)
    axs[1].set_xlabel('frequency (Hz)')
    axs[1].set_xlim([0, Fs / 2])

    if not os.path.exists('figures'):
        os.makedirs('figures')

    fig1.savefig(
        'figures/' + method + '_' + old_title + new_title + '_time.png',
        dpi=400)
    fig2.savefig(
        'figures/' + method + '_' + old_title + new_title + '_freq_mag.png',
        dpi=400)
    fig3.savefig(
        'figures/' + method + '_' + old_title + new_title + '_freq_ang.png',
        dpi=400)
    return


def calc_xcorr(NCF_object, loop=False, count=None):
    '''
    Takes pre-processed data and correlates all corresponding short time
    data segments. It then sums all short time correlations together
    to create a single NCF.

    Parameters
    ----------
    NCF_object : NCF
        data object specifying details of NCF calcuation
    loop : bool
        specifies if this function is being using within a loop. If calling
        this function, leave as default (False)
    count : int
        specifies number of loop iteration. If calling this function,
        leave as default (None)

    Returns
    -------
    NCF_object : NCF
        data object specifying details of NCF calcuation. This function
        adds NCF attribute NCF_object.NCF

    '''
    # Unpack needed values from NCF_object
    h1 = NCF_object.node1_processed_data
    h2 = NCF_object.node2_processed_data
    verbose = NCF_object.verbose

    # Build input list for multiprocessing map
    xcorr_input_list = []
    for k in range(h1.shape[0]):
        single_short_time_input = [h1[k, :], h2[k, :]]
        xcorr_input_list.append(single_short_time_input)

    pool = ThreadPool(processes=mp.cpu_count())
    if verbose:
        print('   Correlating Data...')
    xcorr_list = pool.starmap(calc_xcorr_single_thread, xcorr_input_list)
    pool.terminate()
    xcorr = np.array(xcorr_list)

    xcorr_avg = np.mean(xcorr, axis=0)

    NCF_object.NCF = xcorr_avg
    NCF_object.st_NCFs = xcorr
    return NCF_object


def save_avg_period(NCF_object, count=None):
    # Save Checkpoints for every average period
    filename = './ckpts/ckpt_' + str(count) + '.pkl'

    try:
        with open(filename, 'wb') as f:
            pickle.dump(NCF_object.NCF, f)               # Accumulated xcorr

    except FileNotFoundError:
        os.makedirs('ckpts')
        with open(filename, 'wb') as f:
            # pickle.dump(xcorr_short_time, f)
            pickle.dump(NCF_object.NCF, f)
            # pickle.dump(k,f)

    return None


def calc_xcorr_single_thread(h1, h2):
    '''
    Calculate single short time correlation of h1 and h2. fftconvolve
    is used for slightly faster performance. In normal operation, this
    function should not be called (it is called within calc_xcorr).
    Documentation is maintained for debugging purposes.

    Parameters
    ----------
    h1 : numpy array
        with shape [M,]. Contains time series of processed acoustic data from
        node 1
    h2 : numpy array
        with shape [N,]. contains time series of processed acoustic data from
        node 2

    Returns
    -------
    xcorr : numpy array
        with shape [M+N-1,]. Contains crosscorrelation of h1 and h2
    '''

    xcorr = signal.fftconvolve(h1, np.flip(h2, axis=0), 'full', axes=0)

    # normalize single short time correlation
    # xcorr_norm = xcorr / np.max(xcorr)

    return xcorr


def calculate_NCF_loop(
    num_periods, node1, node2, avg_time, start_time, W, filter_cutoffs,
        verbose=True, whiten=True, htype='broadband', kstart=0,
        sp_method='sabra', other_notes=None):
    '''
    This function loops through multiple average periods and calculates
    the NCF. The resulting NCF is saved to disk in the file directory
    ./ckpts/\n\n
    This allows for calculating the NCF for longer time segments than the
    RAM of the machine can allow.

    Example
    -------
    To loop through multiple average periods and calculate NCF, execute
    following code::
        # Loop through 5 hours to calculate NCF
        num_periods = 5
        avg_time = 60  #minutes
        start_time = datetime.datetime(2017,3,10,0,0,0) # time of first sample
        node1 = 'LJ01C'
        node2 = 'PC01A'
        filter_cutoffs = [12, 30]
        W = 90
        htype = 'broadband'
        whiten= True
        kstart= 0

        calculate_NCF_loop(num_periods, node1, node2, avg_time, start_time, W,
            filter_cutoffs, verbose=True, whiten=whiten, htype=htype, kstart=0)

    Parameters
    ----------
    num_periods : int
        number of average periods to loop through for NCF calculation
    node1 : str
        specifies the node location specifier for node 1
    node2 : str
        specifies the node location specifier for node 2
    avg_time : float
        length in minutes of single NCF average time for single loop
        instance
    start_time : datetime.datetime
        specifies the start time to calculate NCF from
    W : float
        short-time window given in seconds. avg_time must be divisible by W
    filter_cutoffs : list
        list of low and high frequency cutoffs for filtering
    verbose : bool
        specifies whether updates should be printed or not
    whiten: bool
        specifies whether or not to whiten the data
    htype : str
        specifies if low frequency or broadband hydrophone is being used.
        Acceptable inputs are:
        - 'low_frequency'
        - 'broadband'
    kstart : int
        specifies number to start looping from. This is a useful parameter
        to change if there is an error partially through a loop.

    Returns
    -------
    This function returns None, results are saved in ./ckpts/ directory

    '''
    # Header File Just Contains NCF object
    if kstart == 0:
        NCF_object = NCF(
            avg_time, start_time, node1, node2, filter_cutoffs, W, verbose,
            whiten, htype, num_periods, sp_method, other_notes)
        filename = './ckpts/0HEADER.pkl'
        try:
            with open(filename, 'wb') as f:
                pickle.dump(NCF_object, f)
        except Exception:
            os.makedirs('ckpts')
            with open(filename, 'wb') as f:
                pickle.dump(NCF_object, f)

    for k in range(kstart, num_periods):
        start_time_loop = start_time + timedelta(minutes=(avg_time * k))
        NCF_object = NCF(
            avg_time, start_time_loop, node1, node2, filter_cutoffs, W,
            verbose, whiten, htype, sp_method=sp_method)
        print(f'Calculting NCF for Period {k}: {start_time_loop} - \
            {start_time_loop+timedelta(minutes=avg_time)}')
        calculate_NCF(NCF_object, loop=True, count=k)

    return


def freq_whiten(data, Fs, filter_cutoffs):
    '''
    Whiten time series data. Python package `GWPy <https://gwpy.github.io/>`_
    utilized for this function

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

    # Old Method
    '''
    series = TimeSeries(x, sample_rate=Fs)
    white = series.whiten()
    x_new = white.value
    return x_new
    '''
    shape = np.shape(data)
    # data = np.ndarray.flatten(data)

    # assumes data from node1 and node2 have same length
    N = shape[1]

    f = np.arange(0, N) / N * Fs
    win = scipy.signal.windows.hann(N)
    win = win[:, np.newaxis]

    data_win = (data.T * win).T

    dataf = scipy.fft.fft(data_win, axis=1)

    data_mag = np.abs(dataf)

    data_phase = np.angle(dataf)

    idx1 = np.argmin(np.abs(f - filter_cutoffs[0]))
    idx2 = np.argmin(np.abs(f - filter_cutoffs[1]))

    freq_clip_level = \
        np.mean(data_mag) + (np.max(data_mag) - np.mean(data_mag) / 2)
    data_mag[:, idx1:idx2] = freq_clip_level
    data_mag[:, N - idx2:N - idx1] = freq_clip_level

    data_whiten_f = data_mag * np.exp(data_phase * 1j)

    data_whiten = np.real(scipy.fft.ifft(data_whiten_f, axis=1))

    return data_whiten


def freq_whiten_b(data, Fs, filter_cutoffs):
    '''
    Whiten time series data. Python package `GWPy <https://gwpy.github.io/>`_
    utilized for this function

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

    shape = np.shape(data)
    # data = np.ndarray.flatten(data)

    # assumes data from node1 and node2 have same length
    N = shape[1]

    win = scipy.signal.windows.hann(N)
    win = win[:, np.newaxis]

    data_win = (data.T * win).T

    # create unit pulses
    pulse = signal.unit_impulse(N, idx='mid')
    for k in range(shape[0]):
        if k == 0:
            pulses = pulse
        else:
            pulses = np.vstack((pulses, pulse))
    butt_imp_res = filter_bandpass(pulses, Fs, filter_cutoffs)
    butt_imp_mag = np.abs(scipy.fft.fft(butt_imp_res))

    dataf = scipy.fft.fft(data_win, axis=1)

    data_phase = np.angle(dataf)

    data_whiten_f = butt_imp_mag * np.exp(data_phase * 1j)

    data_whiten = np.real(scipy.fft.ifft(data_whiten_f, axis=1))

    return data_whiten


def filter_bandpass(data, Fs, filter_cutoffs):
    '''
    designed to go after get_audio
    '''

    b, a = signal.butter(4, filter_cutoffs / (Fs / 2), btype='bandpass')
    filtered_data = signal.filtfilt(b, a, data, axis=1)

    return filtered_data


class NCF:
    '''
    Object that stores NCF Data. All steps of data calculation are saved in
    this data object.

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
        indicates low and high corner frequencies for implemented butterworth
        bandpass filter. Should be shape [2,]
    W : float
        indicates short time correlation window in seconds
    node1_data : HydrophoneData
        raw data downloaded from ooi data server for hydrophone 1. Data has
        shape [avg_time/W, W*Fs] and is a verticle
        stack of short time series' of length W (in seconds)
    node2_data : HydrophoneData
        raw data downloaded from ooi data server for hydrophone 2. Data has
        shape [avg_time/W, W*Fs] and is a verticle
        stack of short time series' of length W (in seconds)
    node1_processed_data : numpy array
        preprocessed data for hydrophone 1. This includes filtering,
        normalizing short time correlations and frequency whitening
    node2_processed_data : numpy array
        preprocessed data for hydrophone 2. This includes filtering,
        normalizing short time correlations and frequency whitening
    NCF : numpy array
        average noise correlation function over avg_time
    verbose : boolean
        specifies whether to print supporting information
    Fs : float
        sampling frequency of data
    whiten : bool
        indicates whether to whiten data or not
    htype : str
        specifices the type of hydrophone that is used. options include,
        'broadband' and 'low_frequency'
    num_periods : float
        number of average periods looped through. This attribute exists only
        for the header file.
    length_flag : bool
        set if length of data does not match between hydrophones.
    st_NCFs : numpy array
        all short time correlations for average period
    sp_method : str
        signal processing method (sabra or brown)
    other_notes : str
        other notes about experiment
    '''

    def __init__(
        self, avg_time, start_time, node1, node2, filter_cutoffs, W,
            verbose=False, whiten=True, htype='broadband', num_periods=None,
            sp_method='sabra', other_notes=None):

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
        self.sp_method = sp_method
        self.other_notes = other_notes
        return


# Archive
'''
class Hydrophone_Xcorr:

    def __init__(self, node1, node2, avg_time, W=30,
                 verbose=True, filter_data=True):
        """
        Initialize Class OOIHydrophoneData

        Attributes
        ----------
        starttime : datetime.datetime
            indicates start time for acquiring data

        node1 : str
            indicates location of reference hydrophone
                (see table 1 for valid inputs)
        node2 : str
            indicates location of compared hydrophone
                (see table 1 for valid inputs)
        avg_time : int or float
            indicates length of data pulled from server for one
                averaging period (minutes)
        W : int or float
            indicates cross correlation window (seconds)
        verbose : bool
            indicates whether to print updates or not
        filter_data : bool
            indicates whether to filter the data with bandpass with
                cutoffs [10, 1k]
        mp : bool
            indicates if multiprocessing functions should be used

        Private Attributes
        ------------------
        None at this time

        Methods
        -------
        distance_between_hydrophones(self, coord1, coord2)
            Calculates the distance in meters between hydrophones
        get_audio(self, start_time)
            Pulls avg_period amount of data from server
        xcorr_over_avg_period(self, h1, h2)
            Computes cross-correlation for window of length W,
            averaged over avg_period runs xcorr_over_avg_period()
            for num_periods amount of periods

        Private Methods
        ---------------
        None at this time

        TABLE 1:
            ____________________________________________
            |Node Name |        Hydrophone Name        |
            |__________|_______________________________|
            |'LJ01D'  | Oregon Shelf Base Seafloor    |
            |__________|_______________________________|
            |'LJ01A   | Oregon Slope Base Seafloor   |
            |__________|_______________________________|
            |'PC01A'  | Oregon Slope Base Shallow     |
            |__________|_______________________________|
            |'PC03A'  | Axial Base Shallow Profiler   |
            |__________|_______________________________|
            |'LJ01C'  | Oregon Offshore Base Seafloor |
            |__________|_______________________________|
        """
        hydrophone_locations = {'LJ01D': [44.63714, -124.30598],
                                'LJ01C': [44.36943, -124.95357],
                                'PC01A': [44.52897, -125.38967],
                                'LJ01A': [44.51512, -125.38992],
                                'LJ03A': [45.81668, -129.75435],
                                'PC03A': [45.83049, -129.75327]}

        self.hydrophone_locations = hydrophone_locations
        self.node1 = node1
        self.node2 = node2
        self.W = W
        self.verbose = verbose
        self.avg_time = avg_time
        self.mp = mp
        self.Fs = 64000
        self.Ts = 1 / self.Fs
        self.filter_data = filter_data

        self.__distance_between_hydrophones(hydrophone_locations[node1],
                                            hydrophone_locations[node2])

        self.__bearing_between_hydrophones(hydrophone_locations[node1],
                                           hydrophone_locations[node2])

        print('Distance Between Hydrophones: ', self.distance, ' meters')

        print('Estimate Time Delay Between Hydrophones: ',
              self.time_delay, ' seconds')

        print('Bearing Between Hydrophone 1 and 2: ',
              self.theta_bearing_d_1_2, ' degrees')

    # Calculate Distance Between 2 Hydrophones
    # function from:
    # https://www.geeksforgeeks.org/program-distance-two-points-earth/
    def __distance_between_hydrophones(self, coord1, coord2):
        """
        distance_between_hydrophones(coord1, coord2) - calculates the distance
        in meters between two global coordinates

        Inputs:
        coord1 - numpy array of shape [2,1] containing
            latitude and longitude of point 1
        coord2 - numpy array of shape [2,1] containing
            latitude and longitude of point 2

        Outputs:
        self.distance - distance between 2 hydrophones in meters
        self.time_delay - approximate time delay between 2 hydrophones
            (assuming speed of sound = 1480 m/s)

        """
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
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

        c = 2 * asin(sqrt(a))

        # Radius of earth in kilometers. Use 3956 for miles
        r = 6371000
        D = c * r

        self.distance = D
        self.time_delay = D / 1480

    def __bearing_between_hydrophones(self, coord1, coord2):
        """
        bearing_between_hydrophones(coord1, coord2) - calculates the
            bearing in degrees (NSEW) between coord1 and coord2

        Inputs:
        coord1 - numpy array
            of shape [2,1] containing latitude and longitude of point1
        coord2 - numpy array
            of shape [2,1] containing latitude and longitude of point2

        Outputs:
        self.bearing_d_1_2 - float
            bearing in degrees between node 1 and node 2
        """

        psi1 = np.deg2rad(coord1[0])
        lambda1 = np.deg2rad(coord1[1])
        psi2 = np.deg2rad(coord2[0])
        lambda2 = np.deg2rad(coord2[1])
        del_lambda = lambda2 - lambda1

        y = np.sin(del_lambda)*np.cos(psi2)
        x = np.cos(psi1)*np.sin(psi2) -
            np.sin(psi1)*np.cos(psi2)*np.cos(del_lambda)

        theta_bearing_rad = np.arctan2(y, x)
        theta_bearing_d_1_2 = (np.rad2deg(theta_bearing_rad) + 360) % 360

        self.theta_bearing_d_1_2 = theta_bearing_d_1_2

    def get_audio(self, start_time):

        """
        Downloads, and Reshapes Data from OOI server for given
        average period and start time

        Inputs:
        start_time - indicates UTC time that data starts with

        Outputs:
        h1_reshaped : float
            hydrophone data from node 1 of shape (B,N) where
            B = avg_time*60/W and N = W*Fs
        h2_reshaped : float
            hydrophone data from node 2 of shape (B,N) where
            B = avg_time*60/W and N = W*Fs
        flag : bool
            TODO flag structure to be added later
        """

        flag = False
        avg_time = self.avg_time
        verbose = self.verbose
        W = self.W

        avg_time_seconds = avg_time * 60

        if avg_time_seconds % W != 0:
            print('Error: Average Time Must Be Interval of Window')
            return None

        # Initialize Two Classes for Two Hydrophones
        # self.ooi1 = OOIHydrophoneData(limit_seed_files=False,
        # print_exceptions=True, data_gap_mode=2)
        # self.ooi2 = OOIHydrophoneData(limit_seed_files=False,
        # print_exceptions=True, data_gap_mode=2)

        # Calculate end_time
        end_time = start_time + timedelta(minutes=avg_time)

        if verbose:
            print('Getting Audio from Node 1...')
        stopwatch_start = time.time()

        #Audio from Node 1
        node1_data = request.hydrophone.get_acoustic_data(start_time,
            end_time, node=self.node1, verbose=self.verbose, data_gap_mode=2)

        if verbose: print('Getting Audio from Node 2...')

        #Audio from Node 2
        node2_data = request.hydrophone.get_acoustic_data(start_time, end_time,
            node=self.node2, verbose=self.verbose, data_gap_mode=2)

        if (node1_data is None) or (node2_data is None):
            print('Error with Getting Audio')
            return None, None, None

        # Combine Data into Stream
        data_stream = obspy.Stream(traces=[node1_data, node2_data])

        stopwatch_end = time.time()
        print('Time to Download Data from Server: ',
              stopwatch_end - stopwatch_start)

        if data_stream[0].data.shape != data_stream[1].data.shape:
            print('Data streams are not the same length. '
                  'Flag to be added later')
            # TODO: Set up flag structure of some kind

        # Cut off extra points if present
        h1_data = data_stream[0].data[:avg_time * 60 * self.Fs]
        h2_data = data_stream[1].data[:avg_time * 60 * self.Fs]

        return h1_data, h2_data, flag

    def preprocess_audio(self, h1_data, h2_data):

        # Previous Fix for data_gap, Recklessly added zeros
        """
        if ((h1_data.shape[0] < avg_time*60*self.Fs)):
            print('Length of Audio at node 1 too short, zeros added.
            Length: ', data_stream[0].data.shape[0])
            h1_data = np.pad(h1_data,
            (0, avg_time*60*self.Fs-data_stream[0].data.shape[0]))

        if ((h2_data.shape[0] < avg_time*60*self.Fs)):
            print('Length of Audio at node 2 too short, zeros added.
            Length: ', data_stream[1].data.shape[0])
            h2_data = np.pad(h2_data,
            (0, avg_time*60*self.Fs-data_stream[1].data.shape[0]))
        """

        # Filter Data
        if self.filter_data:
            if self.verbose:
                print('Filtering Data...')

            h1_data = self.filter_bandpass(h1_data)
            h2_data = self.filter_bandpass(h2_data)
        self.data_node1 = h1_data
        self.data_node2 = h2_data

        plt.plot(h1_data)
        plt.plot(h2_data)

        h1_reshaped = np.reshape(h1_data, (int(self.avg_time * 60 / self.W),
                                           int(self.W * self.Fs)))

        h2_reshaped = np.reshape(h2_data, (int(self.avg_time * 60 / self.W),
                                           int(self.W * self.Fs)))

        return h1_reshaped, h2_reshaped

    def xcorr_over_avg_period(self, h1, h2, loop=True):
        """
        finds cross correlation over average period
        and averages all correlations

        Inputs:
        h1 - audio data from hydrophone 1 of shape [avg_time(s)/W(s), W*Fs],
            1st axis contains short time NCCF stacked in 0th axis
        h2 - audio data from hydrophone 2 of shape [avg_time(s)/W(s), W*Fs],
            1st axis contains short time NCCF stacked in 0th axis

        Output :
        avg_xcorr of shape (N) where N = W*Fs
        xcorr - xcorr for every short time window within average period shape
        [avg_time(s)/W(s), N]
        """
        avg_time = self.avg_time
        M = h1.shape[1]
        N = h2.shape[1]

        xcorr = np.zeros((int(avg_time * 60 / 30), int(N + M - 1)))

        stopwatch_start = time.time()

        if self.verbose:
            print('Correlating Data...')
        xcorr = signal.fftconvolve(h1, np.flip(h2, axis=1), 'full', axes=1)

        # Normalize Every Short Time Correlation
        xcorr_norm = xcorr / np.max(xcorr, axis=1)[:, np.newaxis]

        xcorr_stack = np.sum(xcorr_norm, axis=0)

        if loop:
            # Save Checkpoints for every average period
            filename = './ckpts/ckpt_' + str(self.count) + '.pkl'

            try:
                with open(filename, 'wb') as f:
                    # Short Time XCORR for all of avg_perd
                    # pickle.dump(xcorr_short_time, f)

                    # Accumulated xcorr
                    pickle.dump(xcorr_norm, f)

                    # avg_period number
                    # pickle.dump(k,f)
            except Exception:
                os.makedirs('ckpts')
                with open(filename, 'wb') as f:
                    # pickle.dump(xcorr_short_time, f)
                    pickle.dump(xcorr_norm, f)
                    # pickle.dump(k,f)

        stopwatch_end = time.time()
        print('Time to Calculate Cross Correlation of 1 period: ',
              stopwatch_end - stopwatch_start)
        if loop:
            return
        else:
            return xcorr_stack, xcorr_norm

    def avg_over_mult_periods(self, num_periods, start_time):
        """
        Computes average over num_periods of averaging periods

        Inputs:
        num_periods - number of periods to average over
        start_time - start time for data

        Outputs:
        xcorr - average xcorr over num_periods of averaging
        """
        verbose = self.verbose

        self.count = 0

        for k in range(num_periods):
            if verbose:
                print('\n\nTime Period: ', k + 1)

            h1, h2, flag = self.get_audio(start_time)

            if flag is None:
                print(f'{k + 1}th average period skipped, no data available')
                continue

            h1_processed, h2_processed = self.preprocess_audio(h1, h2)

            self.xcorr_over_avg_period(h1_processed, h2_processed)

            self.count = self.count + 1

            # Compute Cross Correlation for Each Window and Average
            if first_loop:
                xcorr_avg_period, xcorr_short_time =
                self.xcorr_over_avg_period(h1_processed, h2_processed)
                xcorr = xcorr_avg_period
                first_loop = False
            else:
                xcorr_avg_period, xcorr_short_time =
                self.xcorr_over_avg_period(h1_processed, h2_processed)
                xcorr += xcorr_avg_period
                start_time = start_time + timedelta(minutes=self.avg_time)

            stopwatch_end = time.time()
            print('Time to Complete 1 period: ',
            stopwatch_end - stopwatch_start)

            #Save Checkpoints for every average period
            filename = './ckpts/ckpt_' + str(k) + '.pkl'

            if self.ckpts:
                try:
                    with open(filename,'wb') as f:
                        # Short Time XCORR for all of avg_perd
                        #pickle.dump(xcorr_short_time, f)

                        # Accumulated xcorr
                        pickle.dump(xcorr_avg_period, f)

                        # avg_period number
                        pickle.dump(k,f)
                except:
                    os.makedirs('ckpts')
                    with open(filename,'wb') as f:
                        #pickle.dump(xcorr_short_time, f)
                        pickle.dump(xcorr_avg_period, f)
                        pickle.dump(k,f)

        return None

            self.count = self.count + 1

            # Calculate time variable TODO change to not calculate every loop
            dt = self.Ts
            self.xcorr = xcorr
            t = np.arange(-np.shape(xcorr)[0]*dt/2,np.shape(xcorr)[0]*dt/2,dt)

        #xcorr = xcorr / num_periods

        # Calculate Bearing of Max Peak
        bearing_max_global = self.get_bearing_angle(xcorr, t)

        return t, xcorr, bearing_max_global


    def plot_map_bearing(self, bearing_angle):

        coord1 = self.hydrophone_locations[self.node1]
        coord2 = self.hydrophone_locations[self.node2]
        thetaB1 = bearing_angle[0]
        thetaB2 = bearing_angle[1]

        midpoint, phantom_point1 = \
            self.__find_phantom_point(coord1, coord2, thetaB1)

        midpoint, phantom_point2 = \
            self.__find_phantom_point(coord1, coord2, thetaB2)

        import plotly.graph_objects as go

        hyd_lats = [coord1[0], coord2[0]]
        hyd_lons = [coord1[1], coord2[1]]

        antmidpoint = self.__get_antipode(midpoint)
        fig = go.Figure()

        fig.add_trace(go.Scattergeo(
            lat=[midpoint[0], phantom_point1[0], antmidpoint[0]],
            lon=[midpoint[1], phantom_point1[1], antmidpoint[1]],
            mode='lines',
            line=dict(width=1, color='blue')
        ))

        fig.add_trace(go.Scattergeo(
            lat=[midpoint[0], phantom_point2[0], antmidpoint[0]],
            lon=[midpoint[1], phantom_point2[1], antmidpoint[1]],
            mode='lines',
            line=dict(width=1, color='green')
        ))

        fig.add_trace(go.Scattergeo(
            lon=hyd_lons,
            lat=hyd_lats,
            hoverinfo='text',
            text=['Oregon Slope Base Hydrophone',
                  'Oregon Cabled Benthic Hydrophone'],
            mode='markers',
            marker=dict(
                size=5,
                color='rgb(255, 0, 0)',
                line=dict(
                    width=3,
                    color='rgba(68, 68, 68, 0)'
                )
            )))

        fig.update_layout(
            title_text='Possible Bearings of Max Correlation Peak',
            showlegend=False,
            geo=dict(
                resolution=50,
                showland=True,
                showlakes=True,
                landcolor='rgb(204, 204, 204)',
                countrycolor='rgb(204, 204, 204)',
                lakecolor='rgb(255, 255, 255)',
                projection_type="natural earth",
                coastlinewidth=1,
                lataxis=dict(
                    # range=[20, 60],
                    showgrid=True,
                    dtick=10
                ),
                lonaxis=dict(
                    # range=[-100, 20],
                    showgrid=True,
                    dtick=20
                ),
            )
        )

        fig.show()
        fig.write_html("21_hr_avg_map.html")

    def __find_phantom_point(self, coord1, coord2, thetaB):
        """
        find_phantom_point

        Inputs:
        coord1 - list
            coordinate of first hydrophone
        coord2 - list
            coordinate of second hydrophone

        Output:
        midpoint, phantom_point
        """
        midpoint = [coord1[0] - (coord1[0] - coord2[0]) / 2,
                    coord1[1] - (coord1[1] - coord2[1]) / 2]

        del_lat = 0.01 * np.cos(np.deg2rad(thetaB))
        del_lon = 0.01 * np.sin(np.deg2rad(thetaB))

        phantom_point = [midpoint[0] + del_lat, midpoint[1] + del_lon]

        return midpoint, phantom_point

    def __get_antipode(self, coord):
        # get antipodes
        antlon = coord[1] + 180
        if antlon > 360:
            antlon = antlon - 360
        antlat = -coord[0]
        antipode_coord = [antlat, antlon]
        return antipode_coord

    def filter_bandpass(self, data, Wlow=15, Whigh=25):

        # make data zero mean
        data = data - np.mean(data)
        # decimate by 4
        data_ds_4 = scipy.signal.decimate(data, 4)

        # decimate that by 8 for total of 32
        data_ds_32 = scipy.signal.decimate(data_ds_4, 8)
        # sampling rate = 2000 Hz: Nyquist rate = 1000 Hz

        N = 4

        fs = self.Fs / 32
        b, a = signal.butter(N=N, Wn=[Wlow, Whigh], btype='bandpass', fs=fs)

        data_filt_ds = scipy.signal.lfilter(b, a, data_ds_32)

        data_filt = scipy.signal.resample(data_filt_ds, data.shape[0])

        return data_filt

        return(data_filt)

    def get_bearing_angle(self, t):

        #bearing is with respect to node1 (where node2 is at 0 deg)
        bearing_local = [np.rad2deg(np.arccos(1480*t/self.distance)),
            -np.rad2deg(np.arccos(1480*t/self.distance))]
        #convert bearing_max_local to numpy array
        bearing_local = np.array(bearing_local)
        #convert to global (NSEW) degrees
        bearing_global = self.theta_bearing_d_1_2 + bearing_local
        #make result between 0 and 360
        bearing_global = bearing_global % 360
        self.bearing_global = bearing_global

        return bearing_global

    def plot_polar_TDOA(self, xcorr, t):
        """
        plot_polar_TDOA(self, xcorr)

        Inputs:
        xcorr (numpy array) : array of shape [X,]
        consisting of an averaged cross correlation

        Outputs:
        None
        """

        B = np.arccos(1480 * t / self.distance)
        plt.polar(B, xcorr)
        print(type(B))


def filter_bandpass(data, Wlow=15, Whigh=25):
    # make data zero mean
    data = data - np.mean(data)
    # decimate by 4
    data_ds_4 = scipy.signal.decimate(data,4)

    # decimate that by 8 for total of 32
    data_ds_32 = scipy.signal.decimate(data_ds_4,8)
    # sampling rate = 2000 Hz: Nyquist rate = 1000 Hz

    N = 4

    # HARDCODED TODO: MAKE NOT HARDCODED
    fs = 64000/32
    b,a = signal.butter(N=N, Wn=[Wlow, Whigh], btype='bandpass',fs=fs)

    data_filt_ds= scipy.signal.lfilter(b,a,data_ds_32)

    data_filt = scipy.signal.resample(data_filt_ds ,data.shape[0])

    return(data_filt)
'''

# Archive of Multiprocessing implementation of signal processing
"""
def preprocess_audio_single_thread(h1_data, Fs, filter_cutoffs, whiten):
    '''
    Frequency whiten and filter data from single hydrophone. This function
    is used by preprocess_audio() for multiprocessing and is not usually
    called in normal operation. Documentation maintained for debug
    purposes.

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
    if whiten:
        ts = ts.whiten()
    ts = ts.bandpass(filter_cutoffs[0], filter_cutoffs[1])

    h1_data_processed = ts.value

    return h1_data_processed


def preprocess_audio(NCF_object):
    '''
    Perform all signal processing to hydrophone data before correlating.
    Processing steps include:
    - seperate acoustic data into short time data segments
    - normalize each short time acoustic data segment
    - frequency whiten across entire spectrum
    (using `GWPy <https://gwpy.github.io/>`_)
    - filter data to specifed frequency
    (using `GWPy <https://gwpy.github.io/>`_)

    Parameters
    ----------
    NCF_object : NCF
        NCF data object specifying NCF calculation details

    Returns
    -------
    NCF_object : NCF
        NCF data object specifying NCF calculation details. This
        function adds the NCF_object attributes
        NCF_object.node1_processed_data and NCF_object.node2_processed_data

    '''
    h1_data = NCF_object.node1_data
    h2_data = NCF_object.node2_data
    Fs = NCF_object.Fs
    verbose = NCF_object.verbose
    whiten = NCF_object.whiten
    filter_cutoffs = NCF_object.filter_cutoffs

    preprocess_input_list_node1 = []
    preprocess_input_list_node2 = []
    for k in range(h1_data.shape[0]):
        short_time_input_list_node1 = [
            h1_data[k, :], Fs, filter_cutoffs, whiten]
        short_time_input_list_node2 = [
            h2_data[k, :], Fs, filter_cutoffs, whiten]

        preprocess_input_list_node1.append(short_time_input_list_node1)
        preprocess_input_list_node2.append(short_time_input_list_node2)

    with ThreadPool(processes=mp.cpu_count()) as pool:
        if verbose:
            print('   Filtering and Whitening Data for Node 1...')
        processed_data_list_node1 = pool.starmap(
            preprocess_audio_single_thread, preprocess_input_list_node1)
        if verbose:
            print('   Filtering and Whitening Data for Node 2...')
        processed_data_list_node2 = pool.starmap(
            preprocess_audio_single_thread, preprocess_input_list_node2)

    node1_processed_data = np.array(processed_data_list_node1)
    node2_procesesd_data = np.array(processed_data_list_node2)

    NCF_object.node1_processed_data = node1_processed_data
    NCF_object.node2_processed_data = node2_procesesd_data

    return NCF_object
    """
