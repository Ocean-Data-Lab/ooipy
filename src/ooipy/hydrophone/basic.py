"""
The {py:meth}`ooipy.get_acoustic_data` and {py:meth}`ooipy.get_acoustic_data_LF`
functions return the {py:class}`ooipy.HydrophoneData` object.

The :class:`ooipy.HydrophoneData` objects inherits from obspy.Trace, and methods for
computing calibrated spectrograms and power spectral densities are added.
"""

import datetime
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from obspy import Trace
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import savemat, wavfile

import ooipy


class HydrophoneData(Trace):
    """
    Object that stores hydrophone data

    Attributes
    ----------
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
        """
        Apply a frequency dependent sensitivity correction to the
        acoustic data based on the information from the calibration
        sheets.
        Hydrophone deployments are found at
        https://github.com/OOI-CabledArray/deployments
        Hydrophone calibration sheets are found at
        https://github.com/OOI-CabledArray/calibrationFiles
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
            if self.stats.channel == "HDH":
                sens_calib = 20 * np.log10(sens_calib * 1e-6)
            elif (
                (self.stats.channel == "HNZ")
                | (self.stats.channel == "HNE")
                | (self.stats.channel == "HNN")
            ):
                sens_calib = 20 * np.log10(sens_calib)
                # units for seismograms are in dB rel to m/s^2
        elif round(self.stats.sampling_rate) == 64000:
            sens_calib = sens_calib + 128.9
        else:
            raise Exception("Invalid sampling rate")

        sens_interpolated = interp1d(f_calib, sens_calib)

        f_calib = sens_interpolated(f)
        return f_calib

    def compute_spectrogram(
        self,
        win="hann",
        L=4096,
        avg_time=None,
        overlap=0.5,
        verbose=True,
        average_type="median",
    ):
        """
        Compute spectrogram of acoustic signal. For each time step of the
        spectrogram either a modified periodogram (avg_time=None)
        or a power spectral density estimate using Welch's method with median
        or mean averaging is computed.

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
            If true (default), exception messages and some comments are printed.
        average_type : str
            type of averaging if Welch PSD estimate is used. options are
            'median' (default) and 'mean'.

        Returns
        -------
        spectrogram : xr.DataArray
            An ``xarray.DataArray`` object that contains time and frequency bins as
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
                    x=self.data[n * n_hop : n * n_hop + L],
                    fs=fs,
                    window=win,  # noqa
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
                        self.stats.starttime.datetime + datetime.timedelta(seconds=n * L / fs / 2)
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
                    average=average_type,
                )

                if len(Pxx) != int(L / 2) + 1:
                    if verbose:
                        print(
                            "Error while computing " "Welch estimate for segment",
                            n,
                        )
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
                    average=average_type,
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
            spec_xr = xr.DataArray(
                np.array(specgram),
                dims=["time", "frequency"],
                coords={"time": np.array(time), "frequency": np.array(f)},
                attrs=dict(
                    start_time=self.stats.starttime.datetime,
                    end_time=self.stats.endtime.datetime,
                    nperseg=L,
                    units="dB rel µ Pa^2 / Hz",
                ),
                name="spectrogram",
            )
            return spec_xr

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
        psd : xr.DataArray
           An ``xarray.DataArray`` object that contains frequency bins and PSD values. If no
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
        if scale == "log":
            Pxx = 10 * np.log10(Pxx * np.power(10, sense_corr / 10))
        elif scale == "lin":
            Pxx = Pxx * np.power(10, sense_corr / 10)
        else:
            raise Exception('scale has to be either "lin" or "log".')

        psd_xr = xr.DataArray(
            np.array(Pxx),
            dims=["frequency"],
            coords={"frequency": np.array(f)},
            attrs=dict(
                start_time=self.stats.starttime.datetime,
                end_time=self.stats.endtime.datetime,
                nperseg=L,
                units="dB rel µ Pa^2 / Hz",
            ),
            name="psd",
        )
        return psd_xr

    def wav_write(self, filename, norm=False, new_sample_rate=None):
        """
        method that stores HydrophoneData into .wav file

        Parameters
        ----------
        filename : str
            filename to store .wav file as
        norm : bool
            specifies whether data should be normalized to 1
        new_sample_rate : float
            specifies new sample rate of wav file to be saved. (Resampling is
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
        This data can be found `here <https://raw.githubusercontent.com/
        OOI-CabledArray/deployments/main/HYDBBA_deployments.csv'>`_ for
        broadband hydrophones. Since Low frequency hydrophones remain
        constant with location and time, if the hydrophone is low frequency,
        {location}-{channel} string combination is returned
        """
        # Low frequency hydrophone
        if round(self.stats.sampling_rate) == 200:
            asset_ID = f"{self.stats.location}-{self.stats.channel}"

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
            elif (len(df_start) == 0) | (len(df_end) == 0):
                """^ covers case where currently deployed hydrophone is the
                one that is used in data segment.
                """
                asset_ID = df_ref["assetID"][df_ref.index.to_numpy()[-1]]
            else:
                raise Exception(
                    "Hydrophone Data involves multiple" "deployments. Feature to be added later"
                )
        else:
            raise Exception("Invalid hydrophone sampling rate")

        return asset_ID

    def save(self, file_format, filename, wav_kwargs={}) -> None:
        """
        save hydrophone data in specified method. Supported methods are:
        - pickle - saves the HydrophoneData object as a pickle file
        - netCDF - saves HydrophoneData object as netCDF. Time coordinates are not included
        - mat - saves HydrophoneData object as a .mat file
        - wav - calls wav_write method to save HydrophoneData object as a .wav file

        Parameters
        ----------
        file_format : str
            format to save HydrophoneData object as. Supported formats are
            ['pkl', 'nc', 'mat', 'wav']
        filepath : str
            filepath to save HydrophoneData object. file extension should not be included
        wav_kwargs : dict
            dictionary of keyword arguments to pass to wav_write method

        Returns
        -------
        None
        """

        try:
            self.data
        except AttributeError:
            raise AttributeError("HydrophoneData object does not contain any data")

        if file_format == "pkl":
            # save HydrophoneData object as pickle file

            print(filename + ".pkl")
            with open(filename + ".pkl", "wb") as f:
                pickle.dump(self, f)
        elif file_format == "nc":
            # save HydrophoneData object as netCDF file
            attrs = dict(self.stats)
            attrs["starttime"] = self.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")
            attrs["endtime"] = self.stats.endtime.strftime("%Y-%m-%dT%H:%M:%S.%f")
            attrs["mseed"] = str(attrs["mseed"])
            hdata_x = xr.DataArray(self.data, dims=["time"], attrs=attrs)
            hdata_x.to_netcdf(filename + ".nc")
        elif file_format == "mat":
            # save HydrophoneData object as .mat file
            data_dict = dict(self.stats)
            data_dict["data"] = self.data
            data_dict["starttime"] = self.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")
            data_dict["endtime"] = self.stats.endtime.strftime("%Y-%m-%dT%H:%M:%S.%f")
            savemat(filename + ".mat", {self.stats.location: data_dict})

        elif file_format == "wav":
            # save HydrophoneData object as .wav file
            self.wav_write(filename + ".wav", **wav_kwargs)
        else:
            raise Exception(
                "Invalid file format. Supported formats are: ['pkl', 'nc', 'mat', 'wav']"
            )


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

    # 200 Hz Seismometers
    if node == "AXAS1":
        return "AXAS1"
    if node == "AXAS2":
        return "AXAS2"
    if node == "AXEC1":
        return "AXEC1"
    if node == "AXEC3":
        return "AXEC3"
    if node == "AXID1":
        return "AXID1"
    if node == "HYS11":
        return "HYS11"
    if node == "HYS12":
        return "HYS12"
    if node == "HYS13":
        return "HYS13"

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


def _spectrogram_mp_helper(ooi_hyd_data_obj, win, L, avg_time, overlap, verbose, average_type):
    """
    Helper function for compute_spectrogram_mp
    """
    ooi_hyd_data_obj.compute_spectrogram(win, L, avg_time, overlap, verbose, average_type)
    return ooi_hyd_data_obj.spectrogram


def _psd_mp_helper(ooi_hyd_data_obj, win, L, overlap, avg_method, interpolate, scale):
    """
    Helper function for compute_psd_welch_mp
    """
    ooi_hyd_data_obj.compute_psd_welch(win, L, overlap, avg_method, interpolate, scale)
    return ooi_hyd_data_obj.psd
