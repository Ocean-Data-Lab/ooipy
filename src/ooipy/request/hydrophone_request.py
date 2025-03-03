"""
This modules handles the downloading of OOI Data. As of current, the supported
OOI sensors include all broadband hydrophones (Fs = 64 kHz), all low
frequency hydrophones (Fs = 200 Hz), and bottom mounted OBSs.
All supported hydrophone nodes are listed in the Hydrophone Nodes section below.
"""

import concurrent.futures
import multiprocessing as mp
import sys
from datetime import datetime, timedelta
from functools import partial

import fsspec
import numpy as np
import obspy
import requests
from obspy import Stream, Trace, read
from obspy.core import UTCDateTime
from tqdm import tqdm

# Import all dependencies
from ooipy.hydrophone.basic import HydrophoneData


def get_acoustic_data(
    starttime: datetime,
    endtime: datetime,
    node: str,
    fmin: float = None,
    fmax: float = None,
    max_workers: int = -1,
    append: bool = True,
    verbose: bool = False,
    mseed_file_limit: int = None,
    large_gap_limit: float = 1800.0,
    obspy_merge_method: int = 0,
    gapless_merge: bool = True,
    single_ms_buffer: bool = False,
):
    """
    Get broadband acoustic data for specific time frame and sensor node. The
    data is returned as a :class:`.HydrophoneData` object. This object is
    based on the obspy data trace.

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
        whether or not exceptions are printed in the terminal line
    max_workers : int, optional
        number of maximum workers for concurrent processing.
        Default is -1 (uses number of available cores)
    append : bool, optional
        specifies if extra mseed files should be appended at beginning
        and end in case of boundary gaps in data. Default is True
    verbose : bool, optional
        specifies whether print statements should occur or not
    mseed_file_limit: int, optional
        If the number of mseed traces to be merged exceed this value, the
        function returns None. For some days the mseed files contain
        only a few seconds or milli seconds of data and merging a huge
        amount of files can dramatically slow down the program. if None
        (default), the number of mseed files will not be limited. This also
        limits the number of traces in a single file.
    large_gap_limit: float, optional
        Defines the length in second of large gaps in the data.
        Sometimes, large data gaps are present on particular days. This
        can cause long interpolation times if data_gap_mode 0 or 2 are
        used, possibly resulting in a memory overflow. If a data gap is
        longer than large_gap_limit, data are only retrieved before (if
        the gap stretches beyond the requested time) or after (if the gap
        starts prior to the requested time) the gap, or not at all (if
        the gap is within the requested time).
    obspy_merge_method : int, optional
        either [0,1], see [obspy documentation](https://docs.obspy.org/packages/autogen/
            obspy.core.trace.Trace.html#handling-overlaps)
        for description of merge methods
    gapless_merge: bool, optional
        OOI BB hydrophones have had problems with data fragmentation, where
        individual files are only fractions of seconds long. Before June 2023,
        these were saved as separate mseed files. after 2023 (and in some cases,
        but not all retroactively), 5 minute mseed files contain many fragmented
        traces. These traces are essentially not possible to merge with
        obspy.merge. If True, then method to merge traces without
        consideration of gaps will be attempted. This will only be done if there
        is full data coverage over 5 min file length, but could still result in
        unalligned data. Default value is True. You should probably not use
        this method for data before June 2023 because it will likely cause an error.
    single_ms_buffer : bool
        If true, than 5 minute samples that have ± 1ms of data will also be allowed
        when using gapless merge. There is an issue in the broadband hydrophone
        data where there is occasionally ± 1 ms of data for a 5 minute segment
        (64 samples). This is likely due to the GPS clock errors that cause the
        data fragmentation in the first place.

    Returns
    -------
    HydrophoneData

    """
    # set number of workers
    if max_workers == -1:
        max_workers = mp.cpu_count()

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
            base_time = UTCDateTime(data_url_list[i].split("YDH")[1][1:].split(".mseed")[0])
            utc_time_url_stop = UTCDateTime(
                year=base_time.year,
                month=base_time.month,
                day=base_time.day,
                hour=23,
                minute=59,
                second=59,
                microsecond=999999,
            )

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
                    valid_data_url_list = [
                        data_url_list[i - 1],
                        data_url_list[i],
                    ]
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

    # Check if number of mseed files exceed limit
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

    # removed max workers argument in following statement
    st_list = __map_concurrency(
        __read_mseed, valid_data_url_list, verbose=verbose, max_workers=max_workers
    )

    st_list_new = []
    # combine traces from single files into one trace if gapless merge is set to true
    # if a single 5 minute file is is not compatible with gapless merge, it is currently removed
    if gapless_merge:
        for k, st in enumerate(st_list):

            # count total number of points in stream
            npts_total = 0
            for tr in st:
                npts_total += tr.stats.npts

            # if valid npts, merge traces w/o consideration to gaps
            if single_ms_buffer:
                allowed_lengths = [300, 299.999, 300.001]
            else:
                allowed_lengths = [300]
            if npts_total / sampling_rate in allowed_lengths:
                # NOTE npts_total is nondeterminstically off by ± 64 samples. I have

                # if verbose:
                #    print(f"gapless merge for {valid_data_url_list[k]}")

                data = []
                for tr in st:
                    data.append(tr.data)
                data_cat = np.concatenate(data)

                stats = dict(st[0].stats)
                stats["starttime"] = UTCDateTime(valid_data_url_list[k][-33:-6])
                stats["endtime"] = UTCDateTime(stats["starttime"] + timedelta(minutes=5))
                stats["npts"] = len(data_cat)
                st_list_new.append(Stream(traces=Trace(data_cat, header=stats)))
            else:
                # if verbose:
                #    print(
                #        f"Data segment {valid_data_url_list[k]}, \
                #            with npts {npts_total}, is not compatible with gapless merge"
                #    )

                # check if start times contain unique values
                start_times = []
                for tr in st_list[k]:
                    start_times.append(tr.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S"))
                un_starttimes = set(start_times)
                if len(un_starttimes) == len(st_list[k]):
                    if verbose:
                        print("file fragmented but timestamps are unique. Segment kept")
                    st_list_new.append(st_list[k])
                else:
                    if verbose:
                        print("file fragmented and timestamps are corrupt. Segment thrown out")
                    pass
        st_list = st_list_new

    # check if number of traces in st_list exceeds limit
    if mseed_file_limit is not None:
        for k, st in enumerate(st_list):
            if len(st) > mseed_file_limit:
                if verbose:
                    print(
                        f"Number of traces in mseed file, {valid_data_url_list[k]}\n\
                          exceed mseed_file_limit: {mseed_file_limit}."
                    )
                return None

    # combine list of single traces into stream of straces
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

    st_all = st_all.sort()

    # Merging Data - this section distributes obspy.merge to available cores
    if verbose:
        print(f"Merging {len(st_all)} Traces...")

    if len(st_all) < max_workers * 3:
        # don't use multiprocessing if there are less than 3 traces per worker
        st_all = st_all.merge(method=obspy_merge_method)
    else:
        # break data into num_worker segments
        num_segments = max_workers
        segment_size = len(st_all) // num_segments
        segments = [st_all[i : i + segment_size] for i in range(0, len(st_all), segment_size)]

        with mp.Pool(max_workers) as p:
            segments_merged = p.map(
                partial(__merge_singlecore, merge_method=obspy_merge_method), segments
            )
        # final pass with just 4 cores
        if len(segments_merged) > 12:
            with mp.Pool(4) as p:
                segments_merged = p.map(
                    partial(__merge_singlecore, merge_method=obspy_merge_method), segments_merged
                )

        # merge merged segments
        for k, tr in enumerate(segments_merged):
            if k == 0:
                stream_merge = tr
            else:
                stream_merge += tr
        st_all = stream_merge.merge()

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
    starttime,
    endtime,
    node,
    fmin=None,
    fmax=None,
    verbose=False,
    zero_mean=False,
    channel="HDH",
    correct=False,
):
    """
    Get low frequency acoustic data for specific time frame and sensor
    node. The data is returned as a :class:`.HydrophoneData` object.
    This object is based on the obspy data trace. Example usage is shown
    below. This function does not include the full functionality
    provided by the `IRIS data portal
    <https://service.iris.edu/irisws/timeseries/docs/1/builder/>`_.

    If there is no data for the specified time window, then None is returned

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
    channel : str
        Channel of hydrophone to get data from. Currently supported options
        are 'HDH' - hydrophone, 'HNE' - east seismometer, 'HNN' - north
        seismometer, 'HNZ' - z seismometer. NOTE calibration is only valid for
        'HDH' channel. All other channels are for raw data only at this time.
    correct : bool
        whether or not to use IRIS calibration code. NOTE: when this is true,
        computing PSDs is currently broken as calibration is computed twice

    Returns
    -------
    hydrophone_data : :class:`.HydrophoneData`
        Hyrophone data object. If there is no data in the time window, None
        is returned
    """

    if fmin is None and fmax is None:
        bandpass_range = None
    else:
        bandpass_range = [fmin, fmax]

    url = __build_LF_URL(
        node,
        starttime,
        endtime,
        bandpass_range=bandpass_range,
        zero_mean=zero_mean,
        channel=channel,
        correct=correct,
    )
    if verbose:
        print("Downloading mseed file...")

    try:
        data_stream = read(url)
    except requests.HTTPError:
        if verbose:
            print("   error loading data from OOI server.")
            print("      likely that time window doesn't have data")
        return None

    # removing this (John 9/29/22) not sure if this will caused unknown errors...
    # Try downloading data 5 times. If fails every time raise exception
    # for k in range(5):
    #    try:
    #        data_stream = read(url)
    #        break
    #    except Exception:
    #        if k == 4:
    #            print("   Specific Time window timed out.")
    #            return None

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
    **this function is under development**

    General Purpose OOIpy read function. Parses input parameters to
    appropriate, device specific, read function.

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
        raise Exception("Invalid Device String")

    return hydrophone_data


def __map_concurrency(func, iterator, args=(), max_workers=-1, verbose=False):
    # automatically set max_workers to 2x(available cores)
    if max_workers == -1:
        max_workers = 2 * mp.cpu_count()

    results = [None] * len(iterator)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the load operations and mark each future with its index
        future_to_index = {executor.submit(func, i, *args): idx for idx, i in enumerate(iterator)}
        # Disable progress bar
        is_disabled = not verbose
        for future in tqdm(
            concurrent.futures.as_completed(future_to_index),
            total=len(iterator),
            disable=is_disabled,
            file=sys.stdout,
        ):
            idx = future_to_index[future]
            results[idx] = future.result()
    return results


def __merge_singlecore(ls: list, merge_method: int = 0):
    """
    merge a list of obspy traces into a single trace

    Parameters
    ----------
    stream : list
        list of obspy traces
    merge_method : int
        see `obspy.Stream.merge() <https://docs.obspy.org/packages/autogen/obspy.core.\
            stream.Stream.merge.html>`__ passed to obspy.merge
    """

    stream = obspy.Stream(ls)
    stream_merge = stream.merge(method=merge_method)
    stream_merge.id = ls[0].id
    return stream_merge


def __read_mseed(url):
    # fname = os.path.basename(url)

    # removing try statement that abstracts errors
    # try:
    st = read(url, apply_calib=True)
    # except Exception:
    #    print(f"Data Segment {url} Broken")
    #    return None
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
            instrument = "/HYDBBA106"
            node_id = "/LJ01D"
        if node == "LJ01A" or node == "Oregon_Slope_Base_Seafloor":
            array = "/RS01SLBS"
            instrument = "/HYDBBA102"
            node_id = "/LJ01A"
        if node == "PC01A" or node == "Oregon_Slope_Base_Shallow":
            array = "/RS01SBPS"
            instrument = "/HYDBBA103"
            node_id = "/PC01A"
        if node == "PC03A" or node == "Axial_Base_Shallow":
            array = "/RS03AXPS"
            instrument = "/HYDBBA303"
            node_id = "/PC03A"
        if node == "LJ01C" or node == "Oregon_Offshore_Base_Seafloor":
            array = "/CE04OSBP"
            instrument = "/HYDBBA105"
            node_id = "/LJ01C"
        if node == "LJ03A" or node == "Axial_Base_Seafloor":
            array = "/RS03AXBS"
            instrument = "/HYDBBA302"
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
            f["name"]
            for f in FS.ls(mainurl)
            if f["type"] == "file" and f["name"].endswith(".mseed")
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


def __build_LF_URL(
    node,
    starttime,
    endtime,
    bandpass_range=None,
    zero_mean=False,
    correct=False,
    channel=None,
):
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
    channel : str
        channel string specifier ('HDH', 'HNE', 'HNN', 'HNZ')

    Returns
    -------
    url : str
        url of specified data segment. Format will be in miniseed.
    """

    network, station, location = __get_LF_locations_stats(node, channel)

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


def __get_LF_locations_stats(node, channel):
    network = "OO"
    location = "--"

    # only 200 Hz channels are supported
    if node == "Slope_Base" or node == "HYSB1":
        station = "HYSB1"
        if channel not in ["HDH", "HHN", "HHE", "HHZ", "HNN", "HNE", "HNZ"]:
            raise Exception(
                f"Invalid Channel String {channel} for node {node}.\n\
                    see https://ds.iris.edu/mda/OO/ for available channels and nodes for OOI"
            )
    elif node == "Southern_Hydrate" or node == "HYS14":
        station = "HYS14"
        if channel not in ["HDH", "HHN", "HHE", "HHZ", "HNN", "HNE", "HNZ"]:
            raise Exception(
                f"Invalid Channel String {channel} for node {node}.\n\
                    see https://ds.iris.edu/mda/OO/ for available channels and nodes for OOI"
            )
    elif node == "Axial_Base" or node == "AXBA1":
        station = "AXBA1"
        if channel not in ["HDH", "HHN", "HHE", "HHZ", "HNN", "HNE", "HNZ"]:
            raise Exception(
                f"Invalid Channel String {channel} for node {node}.\n\
                    see https://ds.iris.edu/mda/OO/ for available channels and nodes for OOI"
            )
    elif node == "Central_Caldera" or node == "AXCC1":
        station = "AXCC1"
        if channel not in ["HDH", "HHN", "HHE", "HHZ", "HNN", "HNE", "HNZ"]:
            raise Exception(
                f"Invalid Channel String {channel} for node {node}.\n\
                    see https://ds.iris.edu/mda/OO/ for available channels and nodes for OOI"
            )
    elif node == "Eastern_Caldera" or node == "AXEC2":
        station = "AXEC2"
        if channel not in ["HDH", "HHN", "HHE", "HHZ", "HNN", "HNE", "HNZ"]:
            raise Exception(
                f"Invalid Channel String {channel} for node {node}.\n\
                    see https://ds.iris.edu/mda/OO/ for available channels and nodes for OOI"
            )
    elif node == "AXAS1":
        station = "AXAS1"
        if channel not in ["EHN", "EHE", "EHZ"]:
            raise Exception(
                f"Invalid Channel String {channel} for node {node}.\n\
                    see https://ds.iris.edu/mda/OO/ for available channels and nodes for OOI"
            )
    elif node == "AXAS2":
        station = "AXAS2"
        if channel not in ["EHN", "EHE", "EHZ"]:
            raise Exception(
                f"Invalid Channel String {channel} for node {node}.\n\
                    see https://ds.iris.edu/mda/OO/ for available channels and nodes for OOI"
            )
    elif node == "AXEC1":
        station = "AXEC1"
        if channel not in ["EHN", "EHE", "EHZ"]:
            raise Exception(
                f"Invalid Channel String {channel} for node {node}.\n\
                    see https://ds.iris.edu/mda/OO/ for available channels and nodes for OOI"
            )
    elif node == "AXEC3":
        station = "AXEC3"
        if channel not in ["EHN", "EHE", "EHZ"]:
            raise Exception(
                f"Invalid Channel String {channel} for node {node}.\n\
                    see https://ds.iris.edu/mda/OO/ for available channels and nodes for OOI"
            )
    elif node == "AXID1":
        station = "AXID1"
        if channel not in ["EHN", "EHE", "EHZ"]:
            raise Exception(
                f"Invalid Channel String {channel} for node {node}.\n\
                    see https://ds.iris.edu/mda/OO/ for available channels and nodes for OOI"
            )
    elif node == "HYS11":
        station = "HYS11"
        if channel not in ["EHN", "EHE", "EHZ"]:
            raise Exception(
                f"Invalid Channel String {channel} for node {node}.\n\
                    see https://ds.iris.edu/mda/OO/ for available channels and nodes for OOI"
            )
    elif node == "HYS12":
        station = "HYS12"
        if channel not in ["EHN", "EHE", "EHZ"]:
            raise Exception(
                f"Invalid Channel String {channel} for node {node}.\n\
                    see https://ds.iris.edu/mda/OO/ for available channels and nodes for OOI"
            )
    elif node == "HYS13":
        station = "HYS13"
        if channel not in ["EHN", "EHE", "EHZ"]:
            raise Exception(
                f"Invalid Channel String {channel} for node {node}.\n\
                    see https://ds.iris.edu/mda/OO/ for available channels and nodes for OOI"
            )

    else:
        raise Exception(
            f"Invalid Location String {node}. see https://ds.iris.edu/mda/OO/ for LF OOI nodes"
        )

    return network, station, location
