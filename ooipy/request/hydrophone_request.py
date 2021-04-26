"""
This modules handles the downloading of OOI Data. As of current, the supported
OOI sensors include all broadband hydrophones (Fs = 64 kHz) and all low
frequency hydrophones (Fs = 200 Hz). All supported hydrophone nodes are listed
in the Hydrophone Nodes section below.

.. sidebar:: Hydrophone Request Jupyter Notebook

        For a demo of the hydrophone request module, see the `Hydrophone
        Request Jupyter Notebook <_static/test_request.html>`_.

Hydrophone Nodes
^^^^^^^^^^^^^^^^
* `Oregon Shelf Base Seafloor (Fs = 64 kHz)
<https://ooinet.oceanobservatories.org/data_access/
?search=CE02SHBP-LJ01D-11-HYDBBA106>`_
    * 'LJ01D'
* `Oregon Slope Base Seafloor (Fs = 64 kHz)
<https://ooinet.oceanobservatories.org/data_access/
?search=RS01SLBS-LJ01A-09-HYDBBA102>`_
    * 'LJ01A'
* `Slope Base Shallow (Fs = 64 kHz)
<https://ooinet.oceanobservatories.org/data_access/
?search=RS01SBPS-PC01A-08-HYDBBA103>`_
    * 'PC01A'
* `Axial Base Shallow Profiler (Fs = 64 kHz)
<https://ooinet.oceanobservatories.org/data_access/
?search=RS03AXPS-PC03A-08-HYDBBA303>`_
    * 'PC03A'
* `Offshore Base Seafloor (Fs = 64 kHz)
<https://ooinet.oceanobservatories.org/data_access/
?search=CE04OSBP-LJ01C-11-HYDBBA105>`_
    * 'LJ01C'
* `Axial Base Seafloor (Fs = 64 kHz)
<https://ooinet.oceanobservatories.org/data_access/
?search=RS03AXBS-LJ03A-09-HYDBBA302>`_
    * 'LJ03A'
* `Axial Base Seaflor (Fs = 200 Hz)
<https://ooinet.oceanobservatories.org/data_access/
?search=RS03AXBS-MJ03A-05-HYDLFA301>`_
    * 'Axial_Base'
    * 'AXABA1'
* `Central Caldera (Fs = 200 Hz)
<https://ooinet.oceanobservatories.org/data_access/
?search=RS03CCAL-MJ03F-06-HYDLFA305>`_
    * 'Central_Caldera'
    * 'AXCC1'
* `Eastern Caldera (Fs = 200 Hz)
<https://ooinet.oceanobservatories.org/data_access/
?search=RS03ECAL-MJ03E-09-HYDLFA304>`_
    * 'Eastern_Caldera'
    * 'AXEC2'
* `Southern Hydrate (Fs = 200 Hz)
<https://ooinet.oceanobservatories.org/data_access/
?search=RS01SUM1-LJ01B-05-HYDLFA104>`_
    * 'Southern_Hydrate'
    * 'HYS14'
* `Oregon Slope Base Seafloor (Fs = 200 Hz)
<https://ooinet.oceanobservatories.org/data_access/
?search=RS01SLBS-MJ01A-05-HYDLFA101>`_
    * 'Slope_Base'
    * 'HYSB1'

Hydrophone Request Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

import concurrent.futures
import multiprocessing as mp
from datetime import timedelta

import fsspec
import numpy as np
from obspy import Stream, read
from obspy.core import UTCDateTime

# Import all dependancies
from ooipy.hydrophone.basic import HydrophoneData


def get_acoustic_data(
    starttime,
    endtime,
    node,
    fmin=None,
    fmax=None,
    max_workers=-1,
    append=True,
    verbose=False,
    data_gap_mode=0,
    mseed_file_limit=None,
    large_gap_limit=1800.0,
):
    """
    Get broadband acoustic data for specific time frame and sensor node. The
    data is returned as a :class:`.HydrophoneData` object. This object is
    based on the obspy data trace. Example usage is shown below. For a more in
    depth tutorial, see the `Hydrophone Request Jupyter Notebook Example
    <_static/test_request.html>`_.

    >>> import ooipy
    >>> start_time = datetime.datetime(2017,3,10,0,0,0)
    >>> end_time = datetime.datetime(2017,3,10,0,5,0)
    >>> node = 'PC01A'
    >>> data = ooipy.request.get_acoustic_data(start_time, end_time, node)
    >>> # To access stats for retrieved data:
    >>> print(data.stats)
    >>> # To access numpy array of data:
    >>> print(data.data)

    Parameters
    ----------
    start_time : datetime.datetime
        time of the first noise sample
    end_time : datetime.datetime
        time of the last noise sample
    node : str
        hydrophone name or identifier
    fmin : float, optional
        lower cutoff frequency of hydrophone's bandpass filter. Default
        is None which results in no filtering.
    fmax : float, optional
        higher cutoff frequency of hydrophones bandpass filter. Default
        is None which results in no filtering.
    print_exceptions : bool, optional
        whether or not exeptions are printed in the terminal line
    max_workers : int, optional
        number of maximum workers for concurrent processing
    append : bool, optional
        specifies if extra mseed files should be appended at beginning
        and end in case of boundary gaps in data
    verbose : bool, optional
        specifies whether print statements should occur or not
    data_gap_mode : int, optional
        How gaps in the raw data will be handled. Options are:
        '0': gaps will be linearly interpolated
        '1': no interpolation; mask array is returned
        '2': subtract mean of data and fill gap with zeros; mask array
        is returned
    mseed_file_limit: int, optional
        If the number of mseed files to be merged exceed this value, the
        function returns None. For some days the mseed files contain
        only a few seconds or milli seconds of data and merging a huge
        amount of files can dramatically slow down the program. if None
        (default), the number of mseed files will not be limited.
    large_gap_limit: float, optional
        Defines the length in second of large gaps in the data.
        Sometimes, large data gaps are present on particular days. This
        can cause long interpolation times if data_gap_mode 0 or 2 are
        used, possibly resulting in a memory overflow. If a data gap is
        longer than large_gap_limit, data are only retrieved before (if
        the gap stretches beyond the requested time) or after (if the gap
        starts prior to the requested time) the gap, or not at all (if
        the gap is within the requested time).

    Returns
    -------
    HydrophoneData

    """

    # data_gap = False
    sampling_rate = 64000.0

    if verbose:
        print("Fetching URLs...")

    # get URL for first day
    day_start = UTCDateTime(starttime.year, starttime.month, starttime.day, 0, 0, 0)
    data_url_list = __get_mseed_urls(starttime.strftime("/%Y/%m/%d/"), node, verbose)

    if data_url_list is None:
        if verbose:
            print(
                "No data available for specified day and node. "
                "Please change the day or use a different node"
            )
        return None

    # increment day start by 1 day
    day_start = day_start + 24 * 3600

    # get all urls for each day until endtime is reached
    while day_start < endtime:
        urls_list_next_day = __get_mseed_urls(day_start.strftime("/%Y/%m/%d/"), node, verbose)
        if urls_list_next_day is None:
            day_start = day_start + 24 * 3600
        else:
            data_url_list.extend(urls_list_next_day)
            day_start = day_start + 24 * 3600

    if append:
        # Save last mseed of previous day to data_url_list if not None
        prev_day = starttime - timedelta(days=1)
        data_url_list_prev_day = __get_mseed_urls(prev_day.strftime("/%Y/%m/%d/"), node, verbose)
        if data_url_list_prev_day is not None:
            data_url_list = [data_url_list_prev_day[-1]] + data_url_list

        # get 1 more day of urls
        data_url_last_day_list = __get_mseed_urls(day_start.strftime("/%Y/%m/%d/"), node, verbose)
        if data_url_last_day_list is not None:
            data_url_list = data_url_list + [data_url_last_day_list[0]]

    if verbose:
        print("Sorting valid URLs for Time Window...")
    # Create list of urls for specified time range

    valid_data_url_list = []
    first_file = True

    # Create List of mseed urls for valid time range
    for i in range(len(data_url_list)):

        # get UTC time of current and next item in URL list
        # extract start time from ith file
        utc_time_url_start = UTCDateTime(data_url_list[i].split("YDH")[1][1:].split(".mseed")[0])

        # this line assumes no gaps between current and next file
        if i != len(data_url_list) - 1:
            utc_time_url_stop = UTCDateTime(
                data_url_list[i + 1].split("YDH")[1][1:].split(".mseed")[0]
            )
        else:
            utc_time_url_stop = UTCDateTime(data_url_list[i].split("YDH")[1][1:].split(".mseed")[0])
            utc_time_url_stop.hour = 23
            utc_time_url_stop.minute = 59
            utc_time_url_stop.second = 59
            utc_time_url_stop.microsecond = 999999

        # if current segment contains desired data, store data segment
        if (
            (utc_time_url_start >= starttime and utc_time_url_start < endtime)
            or (utc_time_url_stop >= starttime and utc_time_url_stop < endtime)
            or (utc_time_url_start <= starttime and utc_time_url_stop >= endtime)
        ):
            if append:
                if i == 0:
                    first_file = False
                    valid_data_url_list.append(data_url_list[i])

                elif first_file:
                    first_file = False
                    valid_data_url_list = [data_url_list[i - 1], data_url_list[i]]
                else:
                    valid_data_url_list.append(data_url_list[i])
            else:
                if i == 0:
                    first_file = False
                valid_data_url_list.append(data_url_list[i])

        # adds one more mseed file to st_ll
        else:
            # Checks if last file has been downloaded within time period
            if first_file is False:
                first_file = True
                if append:
                    valid_data_url_list.append(data_url_list[i])
                break

    if isinstance(mseed_file_limit, int):
        if len(valid_data_url_list) > mseed_file_limit:
            if verbose:
                print("Number of mseed files to be merged exceed limit.")
            return None

    # handle large data gaps within one day
    if len(valid_data_url_list) >= 2:
        # find gaps
        gaps = []
        for i in range(len(valid_data_url_list) - 1):
            utc_time_url_first = UTCDateTime(
                valid_data_url_list[i].split("YDH")[1][1:].split(".mseed")[0]
            )
            utc_time_url_second = UTCDateTime(
                valid_data_url_list[i + 1].split("YDH")[1][1:].split(".mseed")[0]
            )
            if utc_time_url_second - utc_time_url_first >= large_gap_limit:
                gaps.append(i)

        gap_cnt = 0
        # check if gap at beginning
        if 0 in gaps:
            del valid_data_url_list[0]
            gap_cnt += 1
            if verbose:
                print("Removed large data gap at beginning of requested time")
        # check if gap at the end
        if len(valid_data_url_list) - 2 in gaps:
            del valid_data_url_list[-1]
            gap_cnt += 1
            if verbose:
                print("Removed large data gap at end of requested time")
        # check if gap within requested time
        if len(gaps) > gap_cnt:
            if verbose:
                print("Found large data gap within requested time")
            return None

    if verbose:
        print("Downloading mseed files...")

    # Code Below from Landung Setiawan
    # removed max workers argument in following statement
    st_list = __map_concurrency(__read_mseed, valid_data_url_list)
    st_all = None
    for st in st_list:
        if st:
            if st[0].stats.sampling_rate != sampling_rate:
                if verbose:
                    print("Some data have different sampling rate")
            else:
                if not isinstance(st_all, Stream):
                    st_all = st
                else:
                    st_all += st

    if st_all is None:
        if verbose:
            print("No data available for selected time")
        return None

    # Merge all traces together
    # Interpolation
    if data_gap_mode == 0:
        st_all.merge(fill_value="interpolate", method=1)
    # Masked Array
    elif data_gap_mode == 1:
        st_all.merge(method=1)
    # Masked Array, Zero-Mean, Zero Fill
    elif data_gap_mode == 2:
        st_all.merge(method=1)
        st_all[0].data = st_all[0].data - np.mean(st_all[0].data)

        try:
            st_all[0].data.fill_value = 0
            st_all[0].data = np.ma.filled(st_all[0].data)
        except Exception:
            if verbose:
                print("data has no minor gaps")

    else:
        if verbose:
            print("Invalid Data Gap Mode")
        return None
    # Slice data to desired window
    st_all = st_all.slice(UTCDateTime(starttime), UTCDateTime(endtime))

    if len(st_all) == 0:
        if verbose:
            print("No data available for selected time frame.")
        return None

    if isinstance(st_all[0].data, np.ma.core.MaskedArray):
        # data_gap = True
        if verbose:  # Note this will only trip if masked array is returned
            print("Data has Gaps")
        # interpolated is treated as if there is no gap

    # Filter Data
    try:
        if fmin is not None and fmax is not None:
            st_all = st_all.filter("bandpass", freqmin=fmin, freqmax=fmax)
            if verbose:
                print("Signal Filtered")
        # return st_all[0]
        return HydrophoneData(st_all[0].data, st_all[0].stats, node)
    except Exception:
        if st_all is None:
            if verbose:
                print("No data available for selected time frame.")
        else:
            if verbose:
                print(Exception)
        return None


def get_acoustic_data_LF(
    starttime, endtime, node, fmin=None, fmax=None, verbose=False, zero_mean=False
):
    """
    Get low frequency acoustic data for specific time frame and sensor
    node. The data is returned as a :class:`.HydrophoneData` object.
    This object is based on the obspy data trace. Example usage is shown
    below. For a more in depth tutorial, see the `Hydrophone Request
    Jupyter Notebook Example <_static/test_request.html>`_. This
    function does not include the full functionality provided by the
    `IRIS data portal
    <https://service.iris.edu/irisws/timeseries/docs/1/builder/>`_.


    >>> starttime = datetime.datetime(2017,3,10,7,0,0)
    >>> endtime = datetime.datetime(2017,3,10,7,1,30)
    >>> location = 'Axial_Base'
    >>> fmin = None
    >>> fmax = None
    >>> # Returns ooipy.ooipy.hydrophone.base.HydrophoneData Object
    >>> data_trace = hydrophone_request.get_acoustic_data_LF(
            starttime, endtime, location, fmin, fmax, zero_mean=True)
    >>> # Access data stats
    >>> data_trace.stats
    >>> # Access numpy array containing data
    >>> data_trace.data

    Parameters
    ----------
    start_time : datetime.datetime
        time of the first noise sample
    end_time : datetime.datetime
        time of the last noise sample
    node : str
        hydrophone
    fmin : float, optional
        lower cutoff frequency of hydrophone's bandpass filter. Default
        is None which results in no filtering.
    fmax : float, optional
        higher cutoff frequency of hydrophones bandpass filter. Default
        is None which results in no filtering.
    verbose : bool, optional
        specifies whether print statements should occur or not
    zero_mean : bool, optional
        specifies whether the mean should be removed. Default to False
    """

    if fmin is None and fmax is None:
        bandpass_range = None
    else:
        bandpass_range = [fmin, fmax]

    url = __build_LF_URL(
        node, starttime, endtime, bandpass_range=bandpass_range, zero_mean=zero_mean
    )
    if verbose:
        print("Downloading mseed file...")

    # Try downloading data 5 times. If fails every time raise exception
    for k in range(5):
        try:
            data_stream = read(url)
            break
        except Exception:
            if k == 4:
                print("   Specific Time window timed out.")
                return None

            # raise Exception ('Problem Requesting Data from OOI Server')

    hydrophone_data = HydrophoneData(data_stream[0].data, data_stream[0].stats, node)
    return hydrophone_data


def ooipy_read(
    device,
    node,
    starttime,
    endtime,
    fmin=None,
    fmax=None,
    verbose=False,
    data_gap_mode=0,
    zero_mean=False,
):
    """
    General Purpose OOIpy read function. Parses input parameters to
    appropriate, device specific, read function. This function is under
    development but is included as is. There is no gurentee that this
    function works as expected.

    Parameters
    ----------
    device : str
        Specifies device type. Valid option are 'broadband_hydrohpone'
        and 'low_frequency_hydrophone'
    node : str
        Specifies data acquisition device location. TODO add available
        options
    starttime : datetime.datetime
        Specifies start time of data requested
    endtime : datetime.datetime
        Specifies end time of data requested
    fmin : float
        Low frequency corner for filtering. If None are give, then no
        filtering happens. Broadband hydrophone data is filtered using
        Obspy. Low frequency hydrophone uses IRIS filtering.
    fmax : float
        High frequency corner for filtering.
    verbose : bool
        Specifies whether or not to print status update statements.
    data_gap_mode : int
        specifies how gaps in data are handled see documentation for
        get_acoustic_data

    Returns
    -------
    hydrophone_data : HydrophoneData
        Object that stores hydrophone data. Similar to obspy trace.
    """

    if device == "broadband_hydrophone":
        hydrophone_data = get_acoustic_data(
            starttime,
            endtime,
            node,
            fmin,
            fmax,
            verbose=verbose,
            data_gap_mode=data_gap_mode,
        )
    elif device == "low_frequency_hydrophone":
        hydrophone_data = get_acoustic_data_LF(
            starttime,
            endtime,
            node,
            fmin=fmin,
            fmax=fmax,
            verbose=verbose,
            zero_mean=zero_mean,
        )
    else:
        raise Exception("Invalid Devic String")

    return hydrophone_data


def __map_concurrency(func, iterator, args=(), max_workers=-1):
    # automatically set max_workers to 2x(available cores)
    if max_workers == -1:
        max_workers = 2 * mp.cpu_count()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(func, i, *args): i for i in iterator}
        for future in concurrent.futures.as_completed(future_to_url):
            data = future.result()
            results.append(data)
    return results


def __read_mseed(url):
    # fname = os.path.basename(url)
    try:
        st = read(url, apply_calib=True)
    except Exception:
        print(f"Data Segment {url} Broken")
        return None
    if isinstance(st, Stream):

        return st
    else:
        print(f"Problem Reading {url}")

        return None


def __get_mseed_urls(day_str, node, verbose):
    """
    get URLs for a specific day from OOI raw data server

    Parameters
    ----------
    day_str : str
        date for which URLs are requested; format: yyyy/mm/dd,
        e.g. 2016/07/15
    node : str
        identifier or name of the hydrophone node
    verbose : bool
        print exceptions if True

    Returns
    -------
    ([str], str)
        list of URLs, each URL refers to one data file. If no data is
        available for specified date, None is returned.
    """

    try:
        if node == "LJ01D" or node == "Oregon_Shelf_Base_Seafloor":
            array = "/CE02SHBP"
            instrument = "/11-HYDBBA106"
            node_id = "/LJ01D"
        if node == "LJ01A" or node == "Oregon_Slope_Base_Seafloor":
            array = "/RS01SLBS"
            instrument = "/09-HYDBBA102"
            node_id = "/LJ01A"
        if node == "PC01A" or node == "Oregon_Slope_Base_Shallow":
            array = "/RS01SBPS"
            instrument = "/08-HYDBBA103"
            node_id = "/PC01A"
        if node == "PC03A" or node == "Axial_Base_Shallow":
            array = "/RS03AXPS"
            instrument = "/08-HYDBBA303"
            node_id = "/PC03A"
        if node == "LJ01C" or node == "Oregon_Offshore_Base_Seafloor":
            array = "/CE04OSBP"
            instrument = "/11-HYDBBA105"
            node_id = "/LJ01C"
        if node == "LJ03A" or node == "Axial_Base_Seafloor":
            array = "/RS03AXBS"
            instrument = "/09-HYDBBA302"
            node_id = "/LJ03A"

        mainurl = (
            "https://rawdata.oceanobservatories.org/files" + array + node_id + instrument + day_str
        )
    except Exception:
        raise Exception(
            "Invalid Location String "
            + node
            + ". Please use one "
            + "of the following node strings: "
            + "'Oregon_Shelf_Base_Seafloor' ('LJ01D'); ",
            "'Oregon_Slope_Base_Seafloor' ('LJ01A'); ",
            "'Oregon_Slope_Base_Shallow' ('PC01A'); ",
            "'Axial_Base_Shallow' ('PC03A'); ",
            "'Oregon_Offshore_Base_Seafloor' ('LJ01C'); ",
            "'Axial_Base_Seafloor' ('LJ03A')",
        )

    FS = fsspec.filesystem("http")

    try:
        data_url_list = sorted(
            [
                f["name"]
                for f in FS.ls(mainurl)
                if f["type"] == "file" and f["name"].endswith(".mseed")
            ]
        )
    except Exception as e:
        if verbose:
            print("Client response: ", e)
        return None

    if not data_url_list:
        if verbose:
            print("No Data Available for Specified Time")
        return None

    return data_url_list


def __build_LF_URL(node, starttime, endtime, bandpass_range=None, zero_mean=False, correct=False):
    """
    Build URL for Lowfrequency Data given the start time, end time, and
    node

    Parameters
    ----------
    node : str
        node of low frequency hydrophone. Options include
        'Easter_Caldera', ...
    starttime : datetime.datetime
        start of data segment requested
    endtime : datetime.datetime
        end of data segment requested
    bandpass_range : list
        list of length two specifying [flow, fhigh] in Hertz. If None
        are given, no bandpass will be added to data.
    zero_mean : bool
        specified whether mean should be removed from data
    correct : bool
        specifies whether to do sensitivity correction on hydrophone data

    Returns
    -------
    url : str
        url of specified data segment. Format will be in miniseed.
    """

    network, station, location, channel = __get_LF_locations_stats(node)

    starttime = starttime.strftime("%Y-%m-%dT%H:%M:%S")
    endtime = endtime.strftime("%Y-%m-%dT%H:%M:%S")
    base_url = "http://service.iris.edu/irisws/timeseries/1/query?"
    netw_url = "net=" + network + "&"
    stat_url = "sta=" + station + "&"
    chan_url = "cha=" + channel + "&"
    strt_url = "start=" + starttime + "&"
    end_url = "end=" + endtime + "&"
    form_url = "format=miniseed&"
    loca_url = "loc=" + location
    if correct:
        corr_url = "&correct=true"
    else:
        corr_url = ""

    if bandpass_range is None:
        band_url = ""
    else:
        band_url = "bp=" + str(bandpass_range[0]) + "-" + str(bandpass_range[1]) + "&"
    if zero_mean:
        mean_url = "demean=true&"
    else:
        mean_url = ""
    url = (
        base_url
        + netw_url
        + stat_url
        + chan_url
        + strt_url
        + end_url
        + mean_url
        + band_url
        + form_url
        + loca_url
        + corr_url
    )
    return url


def __get_LF_locations_stats(node):
    try:
        if node == "Slope_Base" or node == "HYSB1":
            network = "OO"
            station = "HYSB1"
            location = "--"
            channel = "HDH"

        if node == "Southern_Hydrate" or node == "HYS14":
            network = "OO"
            station = "HYS14"
            location = "--"
            channel = "HDH"

        if node == "Axial_Base" or node == "AXBA1":
            network = "OO"
            station = "AXBA1"
            location = "--"
            channel = "HDH"

        if node == "Central_Caldera" or node == "AXCC1":
            network = "OO"
            station = "AXCC1"
            location = "--"
            channel = "HDH"

        if node == "Eastern_Caldera" or node == "AXEC2":
            network = "OO"
            station = "AXEC2"
            location = "--"
            channel = "HDH"

        # Create error if node is invalid
        network = network

    except Exception:
        raise Exception(
            "Invalid Location String "
            + node
            + ". Please use one "
            + "of the following node strings: "
            + "'Slope_Base' ('HYSB1'); ",
            "'Southern_Hydrate' ('HYS14'); ",
            "'Axial_Base' ('AXBA1'); ",
            "'Central_Caldera' ('AXCC1'); ",
            "'Eastern_Caldera' ('AXEC2')",
        )

    return network, station, location, channel
