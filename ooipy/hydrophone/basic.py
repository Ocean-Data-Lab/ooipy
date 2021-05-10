"""
Module for hydrophone (acoustic) data objects

The HydrophoneData objects inherits from obspy.Trace. Furthermore,
methods for computing spectrograms and power spectral densities are
added.
"""
import datetime
import json
import multiprocessing as mp
import os
import pickle
import warnings

import matplotlib
import matplotlib.colors as colors
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from obspy import Trace
from obspy.core import UTCDateTime
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile

import ooipy


class HydrophoneData(Trace):
    """
    Object that stores hydrophone data

    Attributes
    ----------
    spectrogram : Spectrogram
        spectrogram of HydrophoneData.data.data.
        Spectral level, time, and frequency bins can be accessed by
        spectrogram.values, spectrogram.time, and spectrogram.freq
    psd : Psd
        power spectral density estimate of HydrophoneData.data.data.
        Spectral level and frequency bins can be accessed by
        psd.values and psd.freq
    psd_list : list of :class:`.Psd`
        the data object is divided into N segments and for each
        segment a separate power spectral density estimate is computed and
        stored in psd_list. psd_list is computed by compute_psd_welch_mp
    type : str
        Either 'broadband' or 'low_frequency' specifies the type of hydrophone
        that the date is from.

    """

    def __init__(self, data=np.array([]), header=None, node=""):

        super().__init__(data, header)
        self.stats.location = node_id(node)

        self.spectrogram = None
        self.psd = None
        self.psd_list = None
        self.type = None

    # TODO: use correct frequency response for all hydrophones
    def frequency_calibration(self, N):
        # TODO
        """
        Apply a frequency dependent sensitivity correction to the
        acoustic data based on the information from the calibration
        sheets.
        TODO Add for all broadband hydrophones
        !!! Currently only implemented for Oregon Offshore Base Seafloor
        and Oregon Shelf Base Seafloor hydrophone. For all other
        hydrophones, an average sensitivity of -169dBV/1uPa is assumed
        !!!

        Parameters
        ----------
        N : int
            length of the data segment

        Returns
        -------
        output_array : np.array
            array with correction coefficient for every frequency
        """
        # Load calibation file and get appropriate calibration info
        filename = os.path.dirname(ooipy.__file__) + "/hydrophone/calibration_by_assetID.csv"
        # Use deployment CSV to determine asset_ID
        assetID = self.get_asset_ID()

        # load calibration data as pandas dataframe
        cal_by_assetID = pd.read_csv(filename, header=[0, 1])

        f_calib = cal_by_assetID[assetID]["Freq (kHz)"].to_numpy() * 1000
        sens_calib_0 = cal_by_assetID[assetID]["0 phase"].to_numpy()
        sens_calib_90 = cal_by_assetID[assetID]["90 phase"].to_numpy()
        sens_calib = 0.5 * (sens_calib_0 + sens_calib_90)
        f = np.linspace(0, round(self.stats.sampling_rate / 2), N)

        # Convert calibration to correct units
        if round(self.stats.sampling_rate) == 200:
            sens_calib = 20 * np.log10(sens_calib * 1e-6)
        elif round(self.stats.sampling_rate) == 64000:
            sens_calib = sens_calib + 128.9
        else:
            raise Exception("Invalid sampling rate")

        sens_interpolated = interp1d(f_calib, sens_calib)

        f_calib = sens_interpolated(f)
        return f_calib

    def compute_spectrogram(self, win="hann", L=4096, avg_time=None, overlap=0.5, verbose=True):
        """
        Compute spectrogram of acoustic signal. For each time step of the
        spectrogram either a modified periodogram (avg_time=None)
        or a power spectral density estimate using Welch's method with median
        averaging is computed.

        Parameters
        ----------
        win : str, optional
            Window function used to taper the data. See scipy.signal.get_window
            for a list of possible window functions (Default is Hann-window.)
        L : int, optional
            Length of each data block for computing the FFT (Default is 4096).
        avg_time : float, optional
            Time in seconds that is covered in one time step of the
            spectrogram. Default value is None and one time step covers L
            samples. If the signal covers a long time period it is recommended
            to use a higher value for avg_time to avoid memory overflows and
            to facilitate visualization.
        overlap : float, optional
            Percentage of overlap between adjacent blocks if Welch's method is
            used. Parameter is ignored if avg_time is None. (Default is 50%)
        verbose : bool, optional
            If true (defult), exception messages and some comments are printed.

        Returns
        -------
        Spectrogram
            A Spectrogram object that contains time and frequency bins as
            well as corresponding values. If no noise date is available,
            None is returned.
        """
        specgram = []
        time = []

        if any(self.data) is None:
            if verbose:
                print("Data object is empty. Spectrogram cannot be computed")
            self.spectrogram = None
            return None

        # sampling frequency
        fs = self.stats.sampling_rate

        # number of time steps
        if avg_time is None:
            nbins = int((len(self.data) - L) / ((1 - overlap) * L)) + 1
        else:
            nbins = int(np.ceil(len(self.data) / (avg_time * fs)))

        # sensitivity correction
        sense_corr = -self.frequency_calibration(int(L / 2 + 1))

        # compute spectrogram. For avg_time=None
        # (periodogram for each time step), the last data samples are ignored
        # if len(noise[0].data) != k * L
        if avg_time is None:
            n_hop = int(L * (1 - overlap))
            for n in range(nbins):
                f, Pxx = signal.periodogram(
                    x=self.data[n * n_hop : n * n_hop + L], fs=fs, window=win  # noqa
                )
                if len(Pxx) != int(L / 2) + 1:
                    if verbose:
                        print("Error while computing periodogram for segment", n)
                    self.spectrogram = None
                    return None
                else:
                    Pxx = 10 * np.log10(Pxx * np.power(10, sense_corr / 10))

                    specgram.append(Pxx)
                    time.append(
                        self.stats.starttime.datetime + datetime.timedelta(seconds=n * L / fs)
                    )

        else:
            for n in range(nbins - 1):
                f, Pxx = signal.welch(
                    x=self.data[n * int(fs * avg_time) : (n + 1) * int(fs * avg_time)],  # noqa
                    fs=fs,
                    window=win,
                    nperseg=L,
                    noverlap=int(L * overlap),
                    nfft=L,
                    average="median",
                )

                if len(Pxx) != int(L / 2) + 1:
                    if verbose:
                        print("Error while computing " "Welch estimate for segment", n)
                    self.spectrogram = None
                    return None
                else:
                    Pxx = 10 * np.log10(Pxx * np.power(10, sense_corr / 10))
                    specgram.append(Pxx)
                    time.append(
                        self.stats.starttime.datetime + datetime.timedelta(seconds=n * avg_time)
                    )

            # compute PSD for residual segment
            # if segment has more than L samples
            if len(self.data[int((nbins - 1) * fs * avg_time) :]) >= L:  # noqa
                f, Pxx = signal.welch(
                    x=self.data[int((nbins - 1) * fs * avg_time) :],  # noqa
                    fs=fs,
                    window=win,
                    nperseg=L,
                    noverlap=int(L * overlap),
                    nfft=L,
                    average="median",
                )
                if len(Pxx) != int(L / 2) + 1:
                    if verbose:
                        print("Error while computing Welch " "estimate residual segment")
                    self.spectrogram = None
                    return None
                else:
                    Pxx = 10 * np.log10(Pxx * np.power(10, sense_corr / 10))
                    specgram.append(Pxx)
                    time.append(
                        self.stats.starttime.datetime
                        + datetime.timedelta(seconds=(nbins - 1) * avg_time)
                    )

        if len(time) == 0:
            if verbose:
                print("Spectrogram does not contain any data")
            self.spectrogram = None
            return None
        else:
            self.spectrogram = Spectrogram(np.array(time), np.array(f), np.array(specgram))
            return self.spectrogram

    def compute_spectrogram_mp(
        self,
        n_process=None,
        win="hann",
        L=4096,
        avg_time=None,
        overlap=0.5,
        verbose=True,
    ):
        """
        Same as function compute_spectrogram but using multiprocessing.
        This function is intended to be used when analyzing large data sets.

        Parameters
        ----------
        n_process : int, optional
            Number of processes in the pool. None (default) means that
            n_process is equal to the number of CPU cores.
        win : str, optional
            Window function used to taper the data.
            See scipy.signal.get_window for a list of possible window functions
            (Default is Hann-window.)
        L : int, optional
            Length of each data block for computing the FFT (Default is 4096).
        avg_time : float, optional
            Time in seconds that is covered in one time step of the
            spectrogram. Default value is None and one time step covers L
            samples. If the signal covers a long time period it is recommended
            to use a higher value for avg_time to avoid memory overflows and
            to facilitate visualization.
        overlap : float, optional
            Percentage of overlap between adjacent blocks if Welch's method
            is used. Parameter is ignored if avg_time is None. (Default is 50%)
        verbose : bool, optional
            If true (defult), exception messages and some comments are printed.

        Returns
        -------
        Spectrogram
            A Spectrogram object that contains time and frequency bins as well
            as corresponding values. If no noise date is available,
            None is returned.
        """

        # create array with N start and end time values
        if n_process is None:
            N = mp.cpu_count()
        else:
            N = n_process

        ooi_hyd_data_list = []
        seconds_per_process = (self.stats.endtime - self.stats.starttime) / N
        for k in range(N - 1):
            starttime = self.stats.starttime + datetime.timedelta(seconds=k * seconds_per_process)
            endtime = self.stats.starttime + datetime.timedelta(
                seconds=(k + 1) * seconds_per_process
            )
            temp_slice = self.slice(starttime=UTCDateTime(starttime), endtime=UTCDateTime(endtime))
            tmp_obj = HydrophoneData(
                data=temp_slice.data, header=temp_slice.stats, node=self.stats.location
            )
            ooi_hyd_data_list.append((tmp_obj, win, L, avg_time, overlap))

        starttime = self.stats.starttime + datetime.timedelta(seconds=(N - 1) * seconds_per_process)
        temp_slice = self.slice(
            starttime=UTCDateTime(starttime), endtime=UTCDateTime(self.stats.endtime)
        )
        tmp_obj = HydrophoneData(
            data=temp_slice.data, header=temp_slice.stats, node=self.stats.location
        )
        ooi_hyd_data_list.append((tmp_obj, win, L, avg_time, overlap))

        with mp.get_context("spawn").Pool(n_process) as p:
            try:
                specgram_list = p.starmap(_spectrogram_mp_helper, ooi_hyd_data_list)
                # concatenate all small spectrograms to
                # obtain final spectrogram
                specgram = []
                time_specgram = []
                for i in range(len(specgram_list)):
                    time_specgram.extend(specgram_list[i].time)
                    specgram.extend(specgram_list[i].values)
                self.spectrogram = Spectrogram(
                    np.array(time_specgram), specgram_list[0].freq, np.array(specgram)
                )
                return self.spectrogram
            except Exception:
                if verbose:
                    print("Cannot compute spectrogram")
                self.spectrogram = None
                return None

    def compute_psd_welch(
        self,
        win="hann",
        L=4096,
        overlap=0.5,
        avg_method="median",
        interpolate=None,
        scale="log",
        verbose=True,
    ):
        """
        Compute power spectral density estimates of noise data using
        Welch's method.

        Parameters
        ----------
        win : str, optional
            Window function used to taper the data. See scipy.signal.get_window
            for a list of possible window functions (Default is Hann-window.)
        L : int, optional
            Length of each data block for computing the FFT (Default is 4096).
        overlap : float, optional
            Percentage of overlap between adjacent blocks if Welch's method is
            used. Parameter is ignored if avg_time is None. (Default is 50%)
        avg_method : str, optional
            Method for averaging the periodograms when using Welch's method.
            Either 'mean' or 'median' (default) can be used
        interpolate : float, optional
            Resolution in frequency domain in Hz. If None (default), the
            resolution will be sampling frequency fs divided by L. If
            interpolate is smaller than fs/L, the PSD will be interpolated
            using zero-padding
        scale : str, optional
            If 'log' (default) PSD in logarithmic scale (dB re 1µPa^2/H) is
            returned. If 'lin', PSD in linear scale
            (1µPa^2/H) is returned
        verbose : bool, optional
            If true (default), exception messages and some comments are
            printed.

        Returns
        -------
        Psd
            A Psd object that contains frequency bins and PSD values. If no
            noise date is available, None is returned.
        """
        # get noise data segment for each entry in rain_event
        # each noise data segment contains usually 1 min of data
        if any(self.data) is None:
            if verbose:
                print("Data object is empty. PSD cannot be computed")
            self.psd = None
            return None
        fs = self.stats.sampling_rate

        # compute nfft if zero padding is desired
        if interpolate is not None:
            if fs / L > interpolate:
                nfft = int(fs / interpolate)
            else:
                nfft = L
        else:
            nfft = L

        # compute Welch median for entire data segment
        f, Pxx = signal.welch(
            x=self.data,
            fs=fs,
            window=win,
            nperseg=L,
            noverlap=int(L * overlap),
            nfft=nfft,
            average=avg_method,
        )

        if len(Pxx) != int(nfft / 2) + 1:
            if verbose:
                print("PSD cannot be computed.")
            self.psd = None
            return None

        sense_corr = -self.frequency_calibration(int(nfft / 2 + 1))
        print(sense_corr)
        if scale == "log":
            Pxx = 10 * np.log10(Pxx * np.power(10, sense_corr / 10))
        elif scale == "lin":
            Pxx = Pxx * np.power(10, sense_corr / 10)
        else:
            raise Exception('scale has to be either "lin" or "log".')

        self.psd = Psd(f, Pxx)
        return self.psd

    def compute_psd_welch_mp(
        self,
        split,
        n_process=None,
        win="hann",
        L=4096,
        overlap=0.5,
        avg_method="median",
        interpolate=None,
        scale="log",
        verbose=True,
    ):
        """
        Same as compute_psd_welch but using the multiprocessing library.

        Parameters
        ----------
        split : int, float, or array of datetime.datetime
            Time period for each PSD estimate. The time between start_time and
            end_time is split into parts of length
            split seconds (if float). The last segment can be shorter than
            split seconds. Alternatively split can
            be set as an list, where each entry is a start-end time tuple.
        n_process : int, optional
            Number of processes in the pool. None (default) means that
            n_process is equal to the number
            of CPU cores.
        win : str, optional
            Window function used to taper the data. See scipy.signal.get_window
            for a list of possible window functions (Default is Hann-window.)
        L : int, optional
            Length of each data block for computing the FFT (Default is 4096).
        overlap : float, optional
            Percentage of overlap between adjacent blocks if Welch's method is
            used. Parameter is ignored if avg_time is None. (Default is 50%)
        avg_method : str, optional
            Method for averaging the periodograms when using Welch's method.
            Either 'mean' or 'median' (default) can be used
        interpolate : float, optional
            Resolution in frequency domain in Hz. If None (default), the
            resolution will be sampling frequency fs divided by L. If
            interpolate is smaller than fs/L, the PSD will be interpolated
            using zero-padding
        scale : str, optional
            If 'log' (default) PSD in logarithmic scale (dB re 1µPa^2/H) is
            returned. If 'lin', PSD in linear scale (1µPa^2/H) is returned
        verbose : bool, optional
            If true (default), exception messages and some comments are
            printed.

        Returns
        -------
        list of Psd
            A list of Psd objects where each entry represents the PSD of the
            respective noise segment. If no noise data is available, None is
            returned.
        """

        # create array with N start and end time values
        if n_process is None:
            n_process = mp.cpu_count()

        ooi_hyd_data_list = []
        # do segmentation from scratch
        if isinstance(split, int) or isinstance(split, float):
            n_seg = int(np.ceil((self.stats.endtime - self.stats.starttime) / split))

            seconds_per_process = (self.stats.endtime - self.stats.starttime) / n_seg

            for k in range(n_seg - 1):
                starttime = self.stats.starttime + datetime.timedelta(
                    seconds=k * seconds_per_process
                )
                endtime = self.stats.starttime + datetime.timedelta(
                    seconds=(k + 1) * seconds_per_process
                )
                temp_slice = self.slice(
                    starttime=UTCDateTime(starttime), endtime=UTCDateTime(endtime)
                )
                tmp_obj = HydrophoneData(
                    data=temp_slice.data,
                    header=temp_slice.stats,
                    node=self.stats.location,
                )
                ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))

            # treat last segment separately as its length may differ from other
            # segments
            starttime = self.stats.starttime + datetime.timedelta(
                seconds=(n_seg - 1) * seconds_per_process
            )
            temp_slice = self.slice(
                starttime=UTCDateTime(starttime),
                endtime=UTCDateTime(self.stats.endtime),
            )
            tmp_obj = HydrophoneData(
                data=temp_slice.data, header=temp_slice.stats, node=self.stats.location
            )
            ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))
        # use segmentation specified by split
        else:
            ooi_hyd_data_list = []
            for row in split:
                temp_slice = self.slice(starttime=UTCDateTime(row[0]), endtime=UTCDateTime(row[1]))
                tmp_obj = HydrophoneData(
                    data=temp_slice.data,
                    header=temp_slice.stats,
                    node=self.stats.location,
                )
                ooi_hyd_data_list.append((tmp_obj, win, L, overlap, avg_method, interpolate, scale))

        with mp.get_context("spawn").Pool(n_process) as p:
            try:
                self.psd_list = p.starmap(_psd_mp_helper, ooi_hyd_data_list)
            except Exception:
                if verbose:
                    print("Cannot compute PSd list")
                self.psd_list = None

        return self.psd_list

    def wav_write(self, filename, norm=False, new_sample_rate=None):
        """
        method that stores HydrophoneData into .wav file

        Parameters
        ----------
        filename : str
            filename to store .wav file as
        norm : bool
            specifices whether data should be normalized to 1
        new_sample_rate : float
            specifices new sample rate of wav file to be saved. (Resampling is
            done with scipy.signal.resample()). Default is None which keeps
            original sample rate of data.
        """
        if norm:
            data = self.data / np.abs(np.max(self.data))
        else:
            data = self.data

        if new_sample_rate is None:
            sampling_rate = self.stats.sampling_rate
        else:
            if new_sample_rate > self.stats.sampling_rate:
                upsamp_fac = new_sample_rate / self.stats.sampling_rate
                new_npts = self.stats.npts * upsamp_fac
                data = signal.resample(data, int(new_npts))
                sampling_rate = new_sample_rate
            elif new_sample_rate == self.stats.sampling_rate:
                warnings.warn("New sample rate is same as original data. " "No resampling done.")
                sampling_rate = self.stats.sampling_rate
            elif new_sample_rate < self.stats.sampling_rate:
                warnings.warn(
                    "New sample rate is lower than original sample"
                    " rate. Chebychev 1 anti-aliasing filter used"
                )
                if self.stats.sampling_rate % new_sample_rate != 0:
                    raise Exception("New Sample Rate is not factor of original sample rate")
                else:
                    data = signal.decimate(data, int(self.stats.sampling_rate / new_sample_rate))
                    sampling_rate = new_sample_rate

        wavfile.write(filename, int(sampling_rate), data)

    def get_asset_ID(self):
        """
        get_asset_ID returns the hydrophone asset ID for a given data sample.
        This data can be foun `here <https://raw.githubusercontent.com/
        OOI-CabledArray/deployments/main/HYDBBA_deployments.csv'>`_ for
        broadband hydrophones. Since Low frequency hydrophones remain
        constant with location and time, if the hydrophone is low frequency,
        the node ID is returned
        """
        # Low frequency hydrophone
        if round(self.stats.sampling_rate) == 200:
            asset_ID = self.stats.location

        elif round(self.stats.sampling_rate) == 64000:
            url = (
                "https://raw.githubusercontent.com/OOI-CabledArray/"
                "deployments/main/HYDBBA_deployments.csv"
            )
            hyd_df = pd.read_csv(url)

            # LJ01D'Oregon Shelf Base Seafloor
            if self.stats.location == "LJ01D":
                ref = "CE02SHBP-LJ01D-11-HYDBBA106"
            # LJ01AOregon Slope Base Seafloor
            if self.stats.location == "LJ01A":
                ref = "RS01SLBS-LJ01A-09-HYDBBA102"
            # Oregan Slope Base Shallow
            if self.stats.location == "PC01A":
                ref = "RS01SBPS-PC01A-08-HYDBBA103"
            # Axial Base Shallow Profiler
            if self.stats.location == "PC03A":
                ref = "RS03AXPS-PC03A-08-HYDBBA303"
            # Oregon Offshore Base Seafloor
            if self.stats.location == "LJ01C":
                ref = "CE04OSBP-LJ01C-11-HYDBBA105"
            # Axial Base Seafloor
            if self.stats.location == "LJ03A":
                ref = "RS03AXBS-LJ03A-09-HYDBBA302"

            hyd_df["referenceDesignator"]

            df_ref = hyd_df.loc[hyd_df["referenceDesignator"] == ref]

            df_start = df_ref.loc[
                (df_ref["startTime"] < self.stats.starttime)
                & (df_ref["endTime"] > self.stats.starttime)
            ]
            df_end = df_ref.loc[
                (df_ref["startTime"] < self.stats.endtime)
                & (df_ref["endTime"] > self.stats.endtime)
            ]

            if df_start.index.to_numpy() == df_end.index.to_numpy():
                idx = df_start.index.to_numpy()
                asset_ID = df_start["assetID"][int(idx)]
            else:
                raise Exception(
                    "Hydrophone Data involves multiple" "deployments. Feature to be added later"
                )
        else:
            raise Exception("Invalid hydrophone sampling rate")

        return asset_ID


def node_id(node):
    """
    mapping of name of hydrophone node to ID

    Parameter
    ---------
    node : str
        name or ID of the hydrophone node

    Returns
    -------
    str
        ID of hydrophone node
    """
    # broadband hydrophones
    if node == "Oregon_Shelf_Base_Seafloor" or node == "LJ01D":
        return "LJ01D"
    if node == "Oregon_Slope_Base_Seafloor" or node == "LJ01A":
        return "LJ01A"
    if node == "Oregon_Slope_Base_Shallow" or node == "PC01A":
        return "PC01A"
    if node == "Axial_Base_Shallow" or node == "PC03A":
        return "PC03A"
    if node == "Oregon_Offshore_Base_Seafloor" or node == "LJ01C":
        return "LJ01C"
    if node == "Axial_Base_Seafloor" or node == "LJ03A":
        return "LJ03A"

    # low frequency hydrophones
    if node == "Slope_Base" or node == "HYSB1":
        return "HYSB1"
    if node == "Southern_Hydrate" or node == "HYS14":
        return "HYS14"
    if node == "Axial_Base" or node == "AXBA1":
        return "AXBA1"
    if node == "Central_Caldera" or node == "AXCC1":
        return "AXCC1"
    if node == "Eastern_Caldera" or node == "AXEC2":
        return "AXEC2"

    else:
        print("No node exists for name or ID " + node)
        return ""


def node_name(node):
    """
    mapping of ID of hydrophone node to name

    Parameter
    ---------
    node : str
        ID or name of the hydrophone node

    Returns
    -------
    str
        name of hydrophone node
    """
    # broadband hydrophones
    if node == "Oregon_Shelf_Base_Seafloor" or node == "LJ01D":
        return "Oregon_Shelf_Base_Seafloor"
    if node == "Oregon_Slope_Base_Seafloor" or node == "LJ01A":
        return "Oregon_Slope_Base_Seafloor"
    if node == "Oregon_Slope_Base_Shallow" or node == "PC01A":
        return "Oregon_Slope_Base_Shallow"
    if node == "Axial_Base_Shallow" or node == "PC03A":
        return "Axial_Base_Shallow"
    if node == "Oregon_Offshore_Base_Seafloor" or node == "LJ01C":
        return "Oregon_Offshore_Base_Seafloor"
    if node == "Axial_Base_Seafloor" or node == "LJ03A":
        return "Axial_Base_Seafloor"

    # low frequency hydrophones
    if node == "Slope_Base" or node == "HYSB1":
        return "Slope_Base"
    if node == "Southern_Hydrate" or node == "HYS14":
        return "Southern_Hydrate"
    if node == "Axial_Base" or node == "AXBA1":
        return "Axial_Base"
    if node == "Central_Caldera" or node == "AXCC1":
        return "Central_Caldera"
    if node == "Eastern_Caldera" or node == "AXEC2":
        return "Eastern_Caldera"

    else:
        print("No node exists for ID or name " + node)
        return ""


def _spectrogram_mp_helper(ooi_hyd_data_obj, win, L, avg_time, overlap):
    """
    Helper function for compute_spectrogram_mp
    """
    ooi_hyd_data_obj.compute_spectrogram(win, L, avg_time, overlap)
    return ooi_hyd_data_obj.spectrogram


def _psd_mp_helper(ooi_hyd_data_obj, win, L, overlap, avg_method, interpolate, scale):
    """
    Helper function for compute_psd_welch_mp
    """
    ooi_hyd_data_obj.compute_psd_welch(win, L, overlap, avg_method, interpolate, scale)
    return ooi_hyd_data_obj.psd


class Spectrogram:
    """
    A class used to represent a spectrogram object.

    Attributes
    ----------
    time : 1-D array of float or datetime objects
        Indices of time-bins of spectrogram.
    freq : 1-D array of float
        Indices of frequency-bins of spectrogram.
    values : 2-D array of float
        Values of the spectrogram. For each time-frequency-bin pair there has
        to be one entry in values. That is, if time has  length N and freq
        length M, values is a NxM array.

    """

    def __init__(self, time, freq, values):
        self.time = time
        self.freq = freq
        self.values = values

    def visualize(
        self,
        plot_spec=True,
        save_spec=False,
        filename="spectrogram.png",
        title="spectrogram",
        xlabel="time",
        xlabel_rot=70,
        ylabel="frequency",
        fmin=0,
        fmax=32000,
        vmin=20,
        vmax=80,
        vdelta=1.0,
        vdelta_cbar=5,
        figsize=(16, 9),
        dpi=96,
        res_reduction_time=1,
        res_reduction_freq=1,
        time_limits=None,
    ):
        """
        This function will be depreciated into a differnt module in the future.
        The current documentation might not be accurate.

        To plot spectrograms please see
        :meth:`ooipy.hydrophone.basic.plot_spectrogram`

        Basic visualization of spectrogram based on matplotlib. The function
        offers two options: Plot spectrogram in Python (plot_spec = True) and
        save spectrogram plot in directory (save_spec = True). Spectrograms are
        plotted in dB re 1µ Pa^2/Hz.

        plot_spec (bool): whether or not spectrogram is plotted using Python
        save_spec (bool): whether or not spectrogram plot is saved
        filename (str): directory where spectrogram plot is saved. Use ending
        ".png" or ".pdf" to save as PNG or PDF file. This value will be ignored
        if save_spec=False
        title (str): title of plot
        ylabel (str): label of vertical axis
        xlabel (str): label of horizontal axis
        xlabel_rot (float): rotation of xlabel. This is useful if xlabel are
        longer strings for example when using datetime.datetime objects.
        fmin (float): minimum frequency (unit same as f) that is displayed
        fmax (float): maximum frequency (unit same as f) that is displayed
        vmin (float): minimum value (dB) of spectrogram that is colored.
        All values below are displayed in white.
        vmax (float): maximum value (dB) of spectrogram that is colored.
        All values above are displayed in white.
        vdelta (float): color resolution
        vdelta_cbar (int): label ticks in colorbar are in vdelta_cbar steps
        figsize (tuple(int)): size of figure
        dpi (int): dots per inch
        time_limits : list
            specifices xlimits on spectrogram. List contains two
            datetime.datetime objects
        """
        import warnings

        raise warnings.warn(
            "will be depricated in future. Please see " "ooipy.tools.ooiplotlib.plot_spectrogram()"
        )
        # set backend for plotting/saving:
        if not plot_spec:
            matplotlib.use("Agg")

        font = {"size": 22}
        matplotlib.rc("font", **font)

        v = self.values[::res_reduction_time, ::res_reduction_freq]

        if len(self.time) != len(self.values):
            t = np.linspace(0, len(self.values) - 1, int(len(self.values) / res_reduction_time))
        else:
            t = self.time[::res_reduction_time]

        # Convert t to np.array of datetime.datetime
        if type(t[0]) == UTCDateTime:
            for k in range(len(t)):
                t[k] = t[k].datetime

        if len(self.freq) != len(self.values[0]):
            f = np.linspace(
                0,
                len(self.values[0]) - 1,
                int(len(self.values[0]) / res_reduction_freq),
            )
        else:
            f = self.freq[::res_reduction_freq]

        cbarticks = np.arange(vmin, vmax + vdelta, vdelta)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.contourf(
            t,
            f,
            np.transpose(v),
            cbarticks,
            norm=colors.Normalize(vmin=vmin, vmax=vmax),
            cmap=plt.cm.jet,
        )
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.ylim([fmin, fmax])
        if time_limits is not None:
            plt.xlim(time_limits)
        plt.xticks(rotation=xlabel_rot)
        plt.title(title)
        plt.colorbar(im, ax=ax, ticks=np.arange(vmin, vmax + vdelta, vdelta_cbar))
        plt.tick_params(axis="y")

        if type(t[0]) == datetime.datetime:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d %H:%M"))

        if save_spec:
            plt.savefig(filename, dpi=dpi, bbox_inches="tight")

        if plot_spec:
            plt.show()
        else:
            plt.close(fig)

    def save(self, filename="spectrogram.pickle"):
        """
        !!!!! This function will be moved into a different module in the
        future. The current documentation might not be accurate !!!!!

        Save spectrogram in pickle file.

        filename (str): directory where spectrogram data is saved. Ending has
        to be ".pickle".
        """

        dct = {"t": self.time, "f": self.freq, "spectrogram": self.values}
        with open(filename, "wb") as outfile:
            pickle.dump(dct, outfile)

    def plot(self, **kwargs):
        """
        redirects to ooipy.ooiplotlib.plot_spectrogram()
        please see :meth:`ooipy.hydrophone.basic.plot_spectrogram`
        """
        ooipy.tools.ooiplotlib.plot_spectrogram(self, **kwargs)


class Psd:
    """
    A calss used to represent a PSD object

    Attributes
    ----------
    freq : array of float
        Indices of frequency-bins of PSD.
    values : array of float
        Values of the PSD.

    TODO:
    Methods
    -------
    visualize(plot_psd=True, save_psd=False, filename='psd.png', title='PSD',
    xlabel='frequency', xlabel_rot=0, ylabel='spectral level', fmin=0,
    fmax=32, vmin=20, vmax=80, figsize=(16,9), dpi=96)
        Visualizes PSD estimate using matplotlib.
    save(filename='psd.json', ancillary_data=[], ancillary_data_label=[])
        Saves PSD estimate and ancillary data in .json file.
    """

    def __init__(self, freq, values):
        self.freq = freq
        self.values = values

    def visualize(
        self,
        plot_psd=True,
        save_psd=False,
        filename="psd.png",
        title="PSD",
        xlabel="frequency",
        xlabel_rot=0,
        ylabel="spectral level",
        fmin=0,
        fmax=32000,
        vmin=20,
        vmax=80,
        figsize=(16, 9),
        dpi=96,
    ):
        """
        !!!!! This function will be moved into a different module in the
        future. The current documentation might not be accurate !!!!!

        Basic visualization of PSD estimate based on matplotlib. The function
        offers two options: Plot PSD in Python (plot_psd = True) and save PSD
        plot in directory (save_psd = True). PSDs are plotted in dB re 1µ
        Pa^2/Hz.

        plot_psd (bool): whether or not PSD is plotted using Python
        save_psd (bool): whether or not PSD plot is saved
        filename (str): directory where PSD plot is saved. Use ending ".png"
        or ".pdf" to save as PNG or PDF
            file. This value will be ignored if save_psd=False
        title (str): title of plot
        ylabel (str): label of vertical axis
        xlabel (str): label of horizontal axis
        xlabel_rot (float): rotation of xlabel. This is useful if xlabel are
        longer strings.
        fmin (float): minimum frequency (unit same as f) that is displayed
        fmax (float): maximum frequency (unit same as f) that is displayed
        vmin (float): minimum value (dB) of PSD.
        vmax (float): maximum value (dB) of PSD.
        figsize (tuple(int)): size of figure
        dpi (int): dots per inch
        """
        import warnings

        raise warnings.warn(
            "will be depricated in future. Please see " "ooipy.tools.ooiplotlib.plot_psd()"
        )
        # set backend for plotting/saving:
        if not plot_psd:
            matplotlib.use("Agg")

        font = {"size": 22}
        matplotlib.rc("font", **font)

        if len(self.freq) != len(self.values):
            f = np.linspace(0, len(self.values) - 1, len(self.values))
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
            plt.savefig(filename, dpi=dpi, bbox_inches="tight")

        if plot_psd:
            plt.show()
        else:
            plt.close(fig)

    def save(self, filename="psd.json", ancillary_data=[], ancillary_data_label=[]):
        """
        !!!!! This function will be moved into a different module in the
        future. The current documentation might not be accurate !!!!!

        Save PSD estimates along with with ancillary data
        (stored in dictionary) in json file.

        filename (str): directory for saving the data
        ancillary_data ([array like]): list of ancillary data
        ancillary_data_label ([str]): labels for ancillary data used as keys
        in the output dictionary.
            Array has same length as ancillary_data array.
        """

        if len(self.freq) != len(self.values):
            f = np.linspace(0, len(self.values) - 1, len(self.values))
        else:
            f = self.freq

        if type(self.values) != list:
            values = self.values.tolist()

        if type(f) != list:
            f = f.tolist()

        dct = {"psd": values, "f": f}

        if len(ancillary_data) != 0:
            for i in range(len(ancillary_data)):
                if type(ancillary_data[i]) != list:
                    dct[ancillary_data_label[i]] = ancillary_data[i].tolist()
                else:
                    dct[ancillary_data_label[i]] = ancillary_data[i]

        with open(filename, "w+") as outfile:
            json.dump(dct, outfile)

    def plot(self, **kwargs):
        """
        redirects to ooipy.ooiplotlib.plot_psd()
        please see :meth:`ooipy.hydrophone.basic.plot_psd`
        """
        ooipy.tools.ooiplotlib.plot_psd(self, **kwargs)
