"""
Module for CTD data objects
"""

import datetime

import numpy as np

import ooipy


class CtdData:
    """
    Object that stores conductivity, temperature, depth (CTD) data, and
    provides functions for calculating sound speed, temperature,
    pressure, and salinity profiles. When a CtdData object is created
    and extract_parameters = True (default), then temperature, pressure,
    salinity, and time are automatically extracted from the raw data.

    Attributes
    ----------
    raw_data : list of dict
        list containing sample from CTD. Each sample is a dictionary
        containing all parameters measured by the CTD.
    temperature : numpy.ndarray
        array containing temperature samples in degree celsius.
    pressure : numpy.ndarray
        array containing pressure samples in dbar.
    salinity : numpy.ndarray
        array containing salinity samples in parts per thousand.
    depth : numpy.ndarray
        array containing depth samples in meter.
    density : numpy.ndarray
        array containing density samples in kg/cubic meter.
    conductivity : numpy.ndarray
        array containing conductivity samples in siemens/meter.
    sound_speed : numpy.ndarray
        array containing sound speed samples in meter/second.
    time : numpy.ndarray
        array containing time samples as datetime.datetime objects.
    sound_speed_profile : :class:`ooipy.ctd.basic.CtdProfile`
        object for sound speed profile.
    temperature_profile : :class:`ooipy.ctd.basic.CtdProfile`
        object for temperature profile.
    salinity_profile : :class:`ooipy.ctd.basic.CtdProfile`
        object for salinity profile.
    pressure_profile : :class:`ooipy.ctd.basic.CtdProfile`
        object for pressure profile.
    density_profile : :class:`ooipy.ctd.basic.CtdProfile`
        object for density profile.
    conductivity_profile : :class:`ooipy.ctd.basic.CtdProfile`
        object for conductivity profile.

    """

    def __init__(self, raw_data=None, extract_parameters=True):
        self.raw_data = raw_data

        if self.raw_data is not None and extract_parameters:
            self.temperature = self.get_parameter_from_rawdata("temperature")
            self.pressure = self.get_parameter_from_rawdata("pressure")
            self.salinity = self.get_parameter_from_rawdata("salinity")
            self.time = self.get_parameter_from_rawdata("time")
        else:
            self.temperature = None
            self.pressure = None
            self.salinity = None
            self.time = None

        self.sound_speed = None
        self.depth = None
        self.density = None
        self.conductivity = None
        self.sound_speed_profile = None
        self.temperature_profile = None
        self.salinity_profile = None
        self.pressure_profile = None
        self.density_profile = None
        self.conductivity_profile = None

    def ntp_seconds_to_datetime(self, ntp_seconds):
        """
        Converts timestamp into dattime object.
        """
        ntp_epoch = datetime.datetime(1900, 1, 1)
        unix_epoch = datetime.datetime(1970, 1, 1)
        ntp_delta = (unix_epoch - ntp_epoch).total_seconds()
        return datetime.datetime.utcfromtimestamp(ntp_seconds - ntp_delta).replace(microsecond=0)

    def get_parameter_from_rawdata(self, parameter):
        """
        Extracts parameters from raw data dictionary.
        """
        param_arr = []
        for item in self.raw_data:
            if parameter == "temperature":
                if "seawater_temperature" in item:
                    param_arr.append(item["seawater_temperature"])
                elif "temperature" in item:
                    param_arr.append(item["temperature"])
                else:
                    param_arr.append(item["temp"])
            if parameter == "pressure":
                if "ctdbp_no_seawater_pressure" in item:
                    param_arr.append(item["ctdbp_no_seawater_pressure"])
                elif "seawater_pressure" in item:
                    param_arr.append(item["seawater_pressure"])
                else:
                    param_arr.append(item["pressure"])

            if parameter == "salinity":
                if "practical_salinity" in item:
                    param_arr.append(item["practical_salinity"])
                else:
                    param_arr.append(item["salinity"])

            if parameter == "density":
                if "seawater_density" in item:
                    param_arr.append(item["seawater_density"])
                else:
                    param_arr.append(item["density"])

            if parameter == "conductivity":
                if "ctdbp_no_seawater_conductivity" in item:
                    param_arr.append(item["ctdbp_no_seawater_conductivity"])
                elif "seawater_conductivity" in item:
                    param_arr.append(item["seawater_conductivity"])
                else:
                    param_arr.append(item["conductivity"])

            if parameter == "time":
                param_arr.append(self.ntp_seconds_to_datetime(item["pk"]["time"]))

        return np.array(param_arr)

    def get_parameter(self, parameter):
        """
        Extension of get_parameters_from_rawdata. Also sound speed and
        depth can be requested.
        """
        if parameter in [
            "temperature",
            "pressure",
            "salinity",
            "time",
            "density",
            "conductivity",
        ]:
            param = self.get_parameter_from_rawdata(parameter)
        elif parameter == "sound_speed":
            param = self.calc_sound_speed()
        elif parameter == "depth":
            param = self.calc_depth_from_pressure()
        else:
            param = None

        return param

    def calc_depth_from_pressure(self):
        """
        Calculates depth from pressure array
        """

        if self.pressure is None:
            self.pressure = self.get_parameter_from_rawdata("pressure")

        press_MPa = 0.01 * self.pressure

        # TODO: adapt for each hydrophone
        lat = 44.52757  # deg

        # Calculate gravity constant for given latitude
        g_phi = 9.780319 * (
            1
            + 5.2788e-3 * (np.sin(np.deg2rad(lat)) ** 2)
            + 2.36e-5 * (np.sin(np.deg2rad(lat)) ** 4)
        )

        # Calculate Depth for Pressure array
        self.depth = (
            9.72659e2 * press_MPa
            - 2.512e-1 * press_MPa ** 2
            + 2.279e-4 * press_MPa ** 3
            - 1.82e-7 * press_MPa ** 4
        ) / (g_phi + 1.092e-4 * press_MPa)

        return self.depth

    def calc_sound_speed(self):
        """
        Calculates sound speed from temperature, salinity and pressure
        array. The equation for calculating the sound speed is from:
        Chen, C. T., & Millero, F. J. (1977). Speed of sound in seawater
        at high pressures. Journal of the Acoustical Society of America,
        62(5), 1129–1135. https://doi.org/10.1121/1.381646
        """
        if self.pressure is None:
            self.pressure = self.get_parameter_from_rawdata("pressure")
        if self.temperature is None:
            self.temperature = self.get_parameter_from_rawdata("temperature")
        if self.salinity is None:
            self.salinity = self.get_parameter_from_rawdata("salinity")

        press_MPa = 0.01 * self.pressure

        C00 = 1402.388
        A02 = 7.166e-5
        C01 = 5.03830
        A03 = 2.008e-6
        C02 = -5.81090e-2
        A04 = -3.21e-8
        C03 = 3.3432e-4
        A10 = 9.4742e-5
        C04 = -1.47797e-6
        A11 = -1.2583e-5
        C05 = 3.1419e-9
        A12 = -6.4928e-8
        C10 = 0.153563
        A13 = 1.0515e-8
        C11 = 6.8999e-4
        A14 = -2.0142e-10
        C12 = -8.1829e-6
        A20 = -3.9064e-7
        C13 = 1.3632e-7
        A21 = 9.1061e-9
        C14 = -6.1260e-10
        A22 = -1.6009e-10
        C20 = 3.1260e-5
        A23 = 7.994e-12

        C21 = -1.7111e-6
        A30 = 1.100e-10
        C22 = 2.5986e-8
        A31 = 6.651e-12
        C23 = -2.5353e-10
        A32 = -3.391e-13
        C24 = 1.0415e-12
        B00 = -1.922e-2
        C30 = -9.7729e-9
        B01 = -4.42e-5
        C31 = 3.8513e-10
        B10 = 7.3637e-5
        C32 = -2.3654e-12
        B11 = 1.7950e-7
        A00 = 1.389
        D00 = 1.727e-3
        A01 = -1.262e-2
        D10 = -7.9836e-6

        T = 3
        S = 1
        P = 700
        T = self.temperature
        S = self.salinity
        P = press_MPa * 10

        D = D00 + D10 * P
        B = B00 + B01 * T + (B10 + B11 * T) * P
        A = (
            (A00 + A01 * T + A02 * T ** 2 + A03 * T ** 3 + A04 * T ** 4)
            + (A10 + A11 * T + A12 * T ** 2 + A13 * T ** 3 + A14 * T ** 4) * P
            + (A20 + A21 * T + A22 * T ** 2 + A23 * T ** 3) * P ** 2
            + (A30 + A31 * T + A32 * T ** 2) * P ** 3
        )
        Cw = (
            (C00 + C01 * T + C02 * T ** 2 + C03 * T ** 3 + C04 * T ** 4 + C05 * T ** 5)
            + (C10 + C11 * T + C12 * T ** 2 + C13 * T ** 3 + C14 * T ** 4) * P
            + (C20 + C21 * T + C22 * T ** 2 + C23 * T ** 3 + C24 * T ** 4) * P ** 2
            + (C30 + C31 * T + C32 * T ** 2) * P ** 3
        )

        # Calculate Speed of Sound
        self.sound_speed = Cw + A * S + B * S ** (3 / 2) + D * S ** 2

        return self.sound_speed

    def get_profile(self, max_depth, parameter):
        """
        Compute the profile for sound speed, temperature, pressure, or
        salinity over the vater column.

        Parameters
        ----------
        max_depth : int
            The profile will be computed from 0 meters to max_depth
            meters in 1-meter increments
        parameter : str
            * 'sound_speed'
            * 'temperature'
            * 'salinity'
            * 'pressure'
            * 'density'
            * 'conductivity'
        """

        param_dct = {}
        for k in range(max_depth):
            param_dct[str(k)] = {"param": [], "d": []}

        param_arr = self.get_parameter(parameter)

        if self.depth is None:
            self.depth = self.get_parameter("depth")

        for d, p in zip(self.depth, param_arr):
            if str(int(d)) in param_dct:
                param_dct[str(int(d))]["d"].append(d)
                param_dct[str(int(d))]["param"].append(p)

        param_mean = []
        depth_mean = []
        param_var = []
        depth_var = []

        n_samp = []

        for key in param_dct:
            param_mean.append(np.mean(param_dct[key]["param"]))
            depth_mean.append(np.mean(param_dct[key]["d"]))
            param_var.append(np.var(param_dct[key]["param"]))
            depth_var.append(np.var(param_dct[key]["d"]))
            n_samp.append(len(param_dct[key]["d"]))

        idx = np.argsort(depth_mean)

        depth_mean = np.array(depth_mean)[idx]
        param_mean = np.array(param_mean)[idx]
        depth_var = np.array(depth_var)[idx]
        param_var = np.array(param_var)[idx]
        n_samp = np.array(n_samp)[idx]

        param_profile = CtdProfile(param_mean, param_var, depth_mean, depth_var, n_samp)

        if parameter == "temperature":
            self.temperature_profile = param_profile
        elif parameter == "salinity":
            self.salinity_profile = param_profile
        elif parameter == "pressure":
            self.pressure_profile = param_profile
        elif parameter == "sound_speed":
            self.sound_speed_profile = param_profile
        elif parameter == "density":
            self.density_profile = param_profile
        elif parameter == "conductivity":
            self.conductivity_profile = param_profile

        return param_profile


class CtdProfile:
    """
    Simple object that stores a parameter profile over the water column.
    For each 1-meter interval, there is one data point in the profile.

    Attributes
    ----------
    parameter_mean : array of float
        mean of paramter within each 1-meter depth interval
    parameter_var : array of float
        variance of paramter within each 1-meter depth interval
    depth_mean : array of float
        mean of depth within each 1-meter depth interval
    depth_var : array of float
        variance of depth within each 1-meter depth interval
    n_samp : array of int
        number of samples within each 1-meter depth interval
    """

    def __init__(self, parameter_mean, parameter_var, depth_mean, depth_var, n_samp):
        self.parameter_mean = parameter_mean
        self.parameter_var = parameter_var
        self.depth_mean = depth_mean
        self.depth_var = depth_var
        self.n_samp = n_samp

    def plot(self, **kwargs):
        """
        redirects to ooipy.ooiplotlib.plot_ctd_profile()
        please see :meth:`ooipy.hydrophone.basic.plot_psd`
        """
        ooipy.tools.ooiplotlib.plot_ctd_profile(self, **kwargs)

    def convert_to_ssp(self):
        """
        converts to numpy array with correct format for arlpy simulation

        Returns
        -------
        ssp : numpy array
            2D numpy array containing sound speed profile column 0 is depth,
            column 1 is sound speed (in m/s)
        """
        ssp = np.vstack((self.depth_mean, self.parameter_mean)).T
        # insert 0 depth term
        ssp = np.insert(ssp, 0, np.array((0, ssp[0, 1])), 0)

        # remove NaN Terms
        first_nan = np.where(np.isnan(ssp))[0][0]
        ssp = ssp[: first_nan - 1, :]

        return ssp
