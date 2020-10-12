'''
This modules handles the downloading of OOI Data.

.. sidebar:: Hydrophone Request Jupyter Notebook

        For a demo of the hydrophone request module, see the `Hydrophone
        Request Jupyter Notebook <_static/test_request.html>`_.

'''
# Import all dependancies
from ooipy.hydrophone.basic import HydrophoneData
from obspy import read, Stream
from obspy.core import UTCDateTime
from lxml import html
from datetime import timedelta
import numpy as np
import sys
import requests
import multiprocessing as mp
import concurrent.futures
import fsspec
import aiohttp

sys.path.append("..")  # TODO: remove this before publishing


def get_acoustic_data(starttime, endtime, node, fmin=None, fmax=None,
                      max_workers=-1, append=True, verbose=False,
                      data_gap_mode=0, mseed_file_limit=None):
    '''
    Get broadband acoustic data for specific time frame and sensor node:

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

    Returns
    -------
    obspy.core.stream.Stream
        obspy Stream object containing one Trace and date between
        start_time and end_time. Returns None if no data are available
        for specified time frame

    >>> start_time = datetime.datetime(2017,3,10,0,0,0)
    >>> end_time = datetime.datetime(2017,3,10,0,5,0)
    >>> node = '/PC01A'
    >>> data = hyd_request.get_acoustic_data_archive(start_time,
            end_time, node)
    >>> print(data.stats)

    '''

    # data_gap = False

    if verbose:
        print('Fetching URLs...')

    # Save last mseed of previous day to data_url_list if not None
    prev_day = starttime - timedelta(days=1)
    data_url_list_prev_day = __get_mseed_urls(prev_day.strftime("/%Y/%m/%d/"),
                                              node, verbose)
    if data_url_list_prev_day is not None:
        data_url_prev_day = data_url_list_prev_day[-1]

    # get URL for first day
    day_start = UTCDateTime(starttime.year, starttime.month, starttime.day,
                            0, 0, 0)
    data_url_list = __get_mseed_urls(starttime.strftime("/%Y/%m/%d/"), node,
                                     verbose)

    #for k in data_url_list:
    #    print(f'{k}\n') # Checks Out

    if data_url_list is None:
        if verbose:
            print('No data available for specified day and node. '
                  'Please change the day or use a different node')
        return None

    # increment day start by 1 day
    day_start = day_start + 24 * 3600

    # get all urls for each day until endtime is reached
    while day_start < endtime:
        data_url_list.extend(__get_mseed_urls(day_start.strftime("/%Y/%m/%d/"),
                                              node, verbose))
        day_start = day_start + 24 * 3600

    # get 1 more day of urls
    data_url_last_day_list = __get_mseed_urls((day_start).strftime("/%Y/%m/%d/"),
                                              node, verbose)
    data_url_last_day = data_url_last_day_list[0]

    # add 1 extra mseed file at beginning and end to handle gaps if append
    if append:
        data_url_list = [data_url_prev_day] + data_url_list + [data_url_last_day]

    if verbose:
        print('Sorting valid URLs for Time Window...')
    # Create list of urls for specified time range

    valid_data_url_list = []
    first_file = True

    # Create List of mseed urls for valid time range
    for i in range(len(data_url_list)):

        # get UTC time of current and next item in URL list
        # extract start time from ith file
        utc_time_url_start = UTCDateTime(
            data_url_list[i].split('YDH')[1][1:].split('.mseed')[0])

        # this line assumes no gaps between current and next file
        if i != len(data_url_list) - 1:
            utc_time_url_stop = UTCDateTime(
                data_url_list[i + 1].split('YDH')[1][1:].split('.mseed')[0])
        else:
            utc_time_url_stop = UTCDateTime(
                data_url_list[i].split('YDH')[1][1:].split('.mseed')[0])
            utc_time_url_stop.hour = 23
            utc_time_url_stop.minute = 59
            utc_time_url_stop.second = 59
            utc_time_url_stop.microsecond = 999999

        # if current segment contains desired data, store data segment
        if (utc_time_url_start >= starttime
            and utc_time_url_start < endtime) \
            or (utc_time_url_stop >= starttime
                and utc_time_url_stop < endtime) \
            or (utc_time_url_start <= starttime
                and utc_time_url_stop >= endtime):
            if append:
                if i == 0:
                    first_file = False
                    valid_data_url_list.append(data_url_list[i])

                elif (first_file):
                    first_file = False
                    valid_data_url_list = [data_url_list[i - 1],
                                           data_url_list[i]]
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
                print('Number of mseed files to be merged exceed limit.')
            return None

    if verbose:
        print('Downloading mseed files...')

    # Code Below from Landung Setiawan
    # removed max workers argument in following statement
    st_list = __map_concurrency(__read_mseed, valid_data_url_list)
    st_all = None
    for st in st_list:
        if st:
            if not isinstance(st_all, Stream):
                st_all = st
            else:
                st_all += st

    if st_all is None:
        if verbose:
            print('No data available for selected time')
        return None

    # Merge all traces together
    # Interpolation
    if data_gap_mode == 0:
        st_all.merge(fill_value='interpolate', method=1)
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
                print('data has no minor gaps')

    else:
        if verbose:
            print('Invalid Data Gap Mode')
        return None
    # Slice data to desired window
    st_all = st_all.slice(UTCDateTime(starttime), UTCDateTime(endtime))

    if len(st_all) == 0:
        if verbose:
            print('No data available for selected time frame.')
        return None

    if isinstance(st_all[0].data, np.ma.core.MaskedArray):
        # data_gap = True
        if verbose:  # Note this will only trip if masked array is returned
            print('Data has Gaps')
        # interpolated is treated as if there is no gap

    # Filter Data
    try:
        if fmin is not None and fmax is not None:
            st_all = st_all.filter("bandpass", freqmin=fmin,
                                   freqmax=fmax)
            if verbose:
                print('Signal Filtered')
        # return st_all[0]
        return HydrophoneData(st_all[0].data, st_all[0].stats, node)
    except Exception:
        if st_all is None:
            if verbose:
                print('No data available for selected time frame.')
        else:
            if verbose:
                print('Other exception')
        return None


def __map_concurrency(func, iterator, args=(), max_workers=-1):
    # automatically set max_workers to 2x(available cores)
    if max_workers == -1:
        max_workers = 2 * mp.cpu_count()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) \
            as executor:
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
        print(f'Data Segment {url} Broken')
        return None
    if isinstance(st, Stream):

        return st
    else:
        print(f'Problem Reading {url}')

        return None


def __get_mseed_urls(day_str, node, verbose):
    '''
    get URLs for a specific day from OOI raw data server

    Parameters
    ----------
    day_str : str
        date for which URLs are requested; format: yyyy/mm/dd,
        e.g. 2016/07/15

    Returns
    -------
    [str]
        list of URLs, each URL refers to one data file. If no data is
        available for specified date, None is returned.
    '''

    if node == '/LJ01D':  # LJ01D'  Oregon Shelf Base Seafloor
        array = '/CE02SHBP'
        instrument = '/11-HYDBBA106'
    if node == '/LJ01A':  # LJ01A Oregon Slope Base Seafloore
        array = '/RS01SLBS'
        instrument = '/09-HYDBBA102'
    if node == '/PC01A':  # Oregan Slope Base Shallow
        array = '/RS01SBPS'
        instrument = '/08-HYDBBA103'
    if node == '/PC03A':  # Axial Base Shallow Profiler
        array = '/RS03AXPS'
        instrument = '/08-HYDBBA303'
    if node == '/LJ01C':  # Oregon Offshore Base Seafloor
        array = '/CE04OSBP'
        instrument = '/11-HYDBBA105'

    mainurl = 'https://rawdata.oceanobservatories.org/files' + array + node \
              + instrument + day_str

    FS = fsspec.filesystem('http')

    try:
        data_url_list = sorted([f['name'] for f in FS.ls(mainurl)
                               if f['type'] == 'file'
                               and f['name'].endswith('.mseed')])
    except Exception as e:
        if verbose:
            print('Client response: ', e)
        return None

    if not data_url_list:
        if verbose:
            print('No Data Available for Specified Time')
        return None

    return data_url_list


def build_LF_URL(node, starttime, endtime, bandpass_range=None,
                 zero_mean=False):
    '''
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
    Returns
    -------
    url : str
        url of specified data segment. Format will be in miniseed.
    '''

    network, station, location, channel = get_LF_location_stats(node)

    starttime = starttime.strftime("%Y-%m-%dT%H:%M:%S")
    endtime = endtime.strftime("%Y-%m-%dT%H:%M:%S")
    base_url = 'http://service.iris.edu/irisws/timeseries/1/query?'
    netw_url = 'net=' + network + '&'
    stat_url = 'sta=' + station + '&'
    chan_url = 'cha=' + channel + '&'
    strt_url = 'start=' + starttime + '&'
    end_url = 'end=' + endtime + '&'
    form_url = 'format=miniseed&'
    loca_url = 'loc=' + location
    if bandpass_range is None:
        band_url = ''
    else:
        band_url = 'bp=' + str(bandpass_range[0]) + '-' \
            + str(bandpass_range[1]) + '&'
    if zero_mean:
        mean_url = 'demean=true&'
    else:
        mean_url = ''
    url = base_url + netw_url + stat_url + chan_url + strt_url + end_url \
        + mean_url + band_url + form_url + loca_url
    return url


def get_LF_location_stats(node):
    try:
        if node == 'Slope_Base':
            network = 'OO'
            station = 'HYSB1'
            location = '--'
            channel = 'HHE'

        if node == 'Southern_Hydrate':
            network = 'OO'
            station = 'HYS14'
            location = '--'
            channel = 'HHE'

        if node == 'Axial_Base':
            network = 'OO'
            station = 'AXBA1'
            location = '--'
            channel = 'HHE'

        if node == 'Central_Caldera':
            network = 'OO'
            station = 'AXCC1'
            location = '--'
            channel = 'HHE'

        if node == 'Eastern_Caldera':
            network = 'OO'
            station = 'AXEC2'
            location = '--'
            channel = 'HHE'

        # Create error if node is invalid
        network = network

    except Exception:
        raise Exception('Invalid Location String')

    return network, station, location, channel


def get_acoustic_data_LF(starttime, endtime, node, fmin=None, fmax=None,
                         verbose=False, zero_mean=False):
    '''
    Get acoustic data from low frequency OOI Hydrophones
    '''

    if fmin is None and fmax is None:
        bandpass_range = None
    else:
        bandpass_range = [fmin, fmax]

    url = build_LF_URL(node, starttime, endtime, bandpass_range=bandpass_range,
                       zero_mean=zero_mean)
    if verbose:
        print('Downloading mseed file...')

    # Try downloading data 5 times. If fails every time raise exception
    for k in range(5):
        try:
            data_stream = read(url)
            break
        except Exception:
            if k == 4:
                print('   Specific Time window timed out.')
                return None

            # raise Exception ('Problem Requesting Data from OOI Server')

    hydrophone_data = HydrophoneData(data_stream[0].data, data_stream[0].stats,
                                     node)
    return hydrophone_data


def ooipy_read(device, node, starttime, endtime, fmin=None, fmax=None,
               verbose=False, data_gap_mode=0, zero_mean=False):
    '''
    General Purpose OOIpy read function. Parses input parameters to
    appropriate, device specific, read function

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
    '''

    if device == 'broadband_hydrophone':
        hydrophone_data = get_acoustic_data(starttime, endtime, node, fmin,
                                            fmax, verbose=verbose,
                                            data_gap_mode=data_gap_mode)
    elif device == 'low_frequency_hydrophone':
        hydrophone_data = get_acoustic_data_LF(starttime, endtime, node,
                                               fmin=fmin, fmax=fmax,
                                               verbose=verbose,
                                               zero_mean=zero_mean)
    else:
        raise Exception('Invalid Devic String')

    return hydrophone_data


# Archive
'''
def get_acoustic_data_archive_mp(starttime, endtime, node,n_process=None,
    fmin=None, fmax=None,
    append=True, verbose=False, limit_seed_files=True, data_gap_mode=0):

    Same as function get acoustic data but using multiprocessing.

    # entire time frame is divided into n_process parts of equal length
    if n_process == None:
        N  = mp.cpu_count()
    else:
        N = n_process

    seconds_per_process = (endtime - starttime).total_seconds() / N

    get_data_list = [(starttime + datetime.timedelta(
        seconds=i * seconds_per_process),
        starttime + datetime.timedelta(seconds=(i + 1) * seconds_per_process),
        node, fmin, fmax) for i in range(N)]

    # create pool of processes require one part of the data in each process
    with mp.get_context("spawn").Pool(N) as p:
        try:
            data_list = p.starmap(self.get_acoustic_data_archive,
            get_data_list)
        except:
            if verbose:
                print('Data cannot be requested.')
            #data_available = False
            return None

    #if all data is None, return None and set flags
    if (all(x==None for x in data_list)):
        if verbose:
            print('No data available for specified time and node')
        self.data = None
        self.data_available = False
        st_all = None


    #if only some of data is none, remove None entries
    if (None in data_list):
        if self.print_exceptions:
            print('Some mseed files missing or corrupted for time range')
        data_list = list(filter(None.__ne__, data_list))


    # merge data segments together
    st_all = data_list[0]
    if len(data_list) > 1:
        for d in data_list[1:]:
            st_all = st_all + d
    self._data_segmented = data_list
    st_all.merge(method=1)

    if isinstance(st_all[0].data, np.ma.core.MaskedArray):
        self.data_gap = True
        if self.print_exceptions: print('Data has gaps')

    # apply bandpass filter to st_all if desired
    if (self.fmin != None and self.fmax != None):
        st_all = st_all.filter("bandpass", freqmin=fmin, freqmax=fmax)
        print('data filtered')
    self.data = st_all[0]
    self.data_available = True

    self.starttime = starttime
    self.endtime = endtime
    return st_all
'''


def _web_crawler_acoustic_data(day_str, node):
    '''
    get URLs for a specific day from OOI raw data server

    Parameters
    ----------
    day_str : str
        date for which URLs are requested;
        format: yyyy/mm/dd, e.g. 2016/07/15

    Returns
    -------
    [str]
        list of URLs, each URL refers to one data file. If no data are
        available for specified date, None is returned.
    '''

    if node == '/LJ01D':  # LJ01D'  Oregon Shelf Base Seafloor
        array = '/CE02SHBP'
        instrument = '/11-HYDBBA106'
    if node == '/LJ01A':  # LJ01A Oregon Slope Base Seafloore
        array = '/RS01SLBS'
        instrument = '/09-HYDBBA102'
    if node == '/PC01A':  # Oregan Slope Base Shallow
        array = '/RS01SBPS'
        instrument = '/08-HYDBBA103'
    if node == '/PC03A':  # Axial Base Shallow Profiler
        array = '/RS03AXPS'
        instrument = '/08-HYDBBA303'
    if node == '/LJ01C':  # Oregon Offshore Base Seafloor
        array = '/CE04OSBP'
        instrument = '/11-HYDBBA105'

    mainurl = 'https://rawdata.oceanobservatories.org/files' + array + node \
              + instrument + day_str
    try:
        mainurlpage = requests.get(mainurl, timeout=60)
    except Exception:
        print('Timeout URL request')
        return None
    webpage = html.fromstring(mainurlpage.content)
    suburl = webpage.xpath('//a/@href')

    FileNum = len(suburl)
    data_url_list = []
    for filename in suburl[6:FileNum]:
        data_url_list.append(str(mainurl + filename[2:]))

    return data_url_list


def get_acoustic_data_archive(starttime, endtime, node, fmin=None, fmax=None,
                              append=True, verbose=False,
                              limit_seed_files=True, data_gap_mode=0):
    '''
    Get acoustic data for specific time frame and node:

    start_time (datetime.datetime): time of the first noise sample
    end_time (datetime.datetime): time of the last noise sample
    node (str): hydrophone
    fmin (float): lower cutoff frequency of hydrophone's bandpass filter.
        Default is None which results in no filtering.
    fmax (float): higher cutoff frequency of hydrophones bandpass filter.
        Default is None which results in no filtering.
    print_exceptions (bool): whether or not exeptions are printed in the
        terminal line
    verbose (bool) : Determines if information is printed to command line

    return (obspy.core.stream.Stream): obspy Stream object containing
        one Trace and date
        between start_time and end_time. Returns None if no data are
        available for specified time frame

    '''

    # data_gap = False

    # Save last mseed of previous day to data_url_list
    prev_day = starttime - timedelta(days=1)

    if append:
        data_url_list_prev_day = \
            _web_crawler_acoustic_data(prev_day.strftime("/%Y/%m/%d/"), node)
        # keep only .mseed files
        del_list = []
        for i in range(len(data_url_list_prev_day)):
            url = data_url_list_prev_day[i].split('.')
            if url[len(url) - 1] != 'mseed':
                del_list.append(i)
        data_url_prev_day = np.delete(data_url_list_prev_day, del_list)
        data_url_prev_day = data_url_prev_day[-1]

    # get URL for first day
    day_start = UTCDateTime(starttime.year, starttime.month,
                            starttime.day, 0, 0, 0)
    data_url_list = \
        _web_crawler_acoustic_data(starttime.strftime("/%Y/%m/%d/"), node)
    if data_url_list is None:
        if verbose:
            print('No data available for specified day and node. '
                  'Please change the day or use a different node')
        return None

    # increment day start by 1 day
    day_start = day_start + 24 * 3600

    # get all urls for each day untill endtime is reached
    while day_start < endtime:
        data_url_list.extend(_web_crawler_acoustic_data(
                             starttime.strftime("/%Y/%m/%d/"), node))
        day_start = day_start + 24 * 3600

    if limit_seed_files:
        # if too many files for one day -> skip day (otherwise program takes
        # too long to terminate)
        if len(data_url_list) > 1000:
            if verbose:
                print('Too many files for specified day. Cannot request data '
                      'as web crawler cannot terminate.')
            return None

    # keep only .mseed files
    del_list = []
    for i in range(len(data_url_list)):
        url = data_url_list[i].split('.')
        if url[len(url) - 1] != 'mseed':
            del_list.append(i)
    data_url_list = np.delete(data_url_list, del_list)

    if append:
        data_url_list = np.insert(data_url_list, 0, data_url_prev_day)

    st_all = None
    first_file = True
    # only acquire data for desired time

    for i in range(len(data_url_list)):
        # get UTC time of current and next item in URL list
        # extract start time from ith file
        utc_time_url_start = UTCDateTime(
            data_url_list[i].split('YDH')[1][1:].split('.mseed')[0])

        # this line assumes no gaps between current and next file
        if i != len(data_url_list) - 1:
            utc_time_url_stop = UTCDateTime(
                data_url_list[i + 1].split('YDH')[1][1:].split('.mseed')[0])
        else:
            utc_time_url_stop = UTCDateTime(
                data_url_list[i].split('YDH')[1][1:].split('.mseed')[0])
            utc_time_url_stop.hour = 23
            utc_time_url_stop.minute = 59
            utc_time_url_stop.second = 59
            utc_time_url_stop.microsecond = 999999

        # if current segment contains desired data, store data segment
        if (utc_time_url_start >= starttime
            and utc_time_url_start < endtime) \
            or (utc_time_url_stop >= starttime
                and utc_time_url_stop < endtime) \
            or (utc_time_url_start <= starttime
                and utc_time_url_stop >= endtime):

            if (first_file) and (i != 0):
                first_file = False
                try:
                    if append:
                        # add one extra file on front end
                        st = read(data_url_list[i - 1], apply_calib=True)
                        st += read(data_url_list[i], apply_calib=True)
                    else:
                        st = read(data_url_list[i], apply_calib=True)
                except Exception:
                    if verbose:
                        print(f"Data Segment, {data_url_list[i-1]} \
                              or {data_url_list[i] } Broken")
                    # self.data = None
                    # self.data_available = False
                    # return None
            # normal operation (not first or last file)
            else:
                try:
                    st = read(data_url_list[i], apply_calib=True)
                except Exception:
                    if verbose:
                        print(f"Data Segment, {data_url_list[i]} Broken")
                    # self.data = None
                    # self.data_available = False
                    # return None

            # Add st to acculation of all data st_all
            if st_all is None:
                st_all = st
            else:
                st_all += st
        # adds one more mseed file to st_ll
        else:
            # Checks if last file has been downloaded within time period
            if first_file is False:
                first_file = True
                try:
                    if append:
                        st = read(data_url_list[i], apply_calib=True)
                except Exception:
                    if verbose:
                        print(f"Data Segment, {data_url_list[i]} Broken")
                    # self.data = None
                    # self.data_available = False
                    # return None

                if append:
                    st_all += st

    # Merge all traces together
    if data_gap_mode == 0:
        st_all.merge(fill_value='interpolate', method=1)
    # Returns Masked Array if there are data gaps
    elif data_gap_mode == 1:
        st_all.merge(method=1)
    else:
        if verbose:
            print('Invalid Data Gap Mode')
        return None
    # Slice data to desired window
    st_all = st_all.slice(UTCDateTime(starttime), UTCDateTime(endtime))

    if len(st_all) == 0:
        if verbose:
            print('No data available for selected time frame.')
        return None

    if isinstance(st_all[0].data, np.ma.core.MaskedArray):
        # data_gap = True

        # Note this will only trip if masked array is returned
        if verbose:
            print('Data has Gaps')
        # interpolated is treated as if there is no gap

    # Filter Data
    try:
        if fmin is not None and fmax is not None:
            st_all = st_all.filter("bandpass", freqmin=fmin, freqmax=fmax)
            if verbose:
                print('Signal Filtered')
        data = st_all[0]
        # data_available = True
        return HydrophoneData(data.data, data.stats, node)
    except Exception:
        if st_all is None:
            if verbose:
                print('No data available for selected time frame.')
        else:
            if verbose:
                print('Other exception')
        return None
