import concurrent.futures
import datetime
import multiprocessing as mp

import requests

import ooipy.request.authentification
from ooipy.ctd.basic import CtdData


def get_ctd_data(
    start_datetime,
    end_datetime,
    location,
    limit=10000,
    only_profilers=False,
    delivery_method="auto",
):
    """
    Requests CTD data between start_detetime and end_datetime for the
    specified location if data is available. For each location, data of
    all available CTDs are requested and concatenated in a list. That is
    the final data list can consists of multiple segments of data, where
    each segment contains the data from one instrument. This means that
    the final list might not be ordered in time, but rather should be
    treated as an unordered list of CTD data points.

    Parameters
    ----------
    start_datetime : datetime.datetime
        time of first sample from CTD
    end_datetime : datetime.datetime
        time of last sample from CTD
    location : str
        location for which data are requested. Possible choices are:
        * 'oregon_inshore'
        * 'oregon_shelf'
        * 'oregon_offshore'
        * 'oregon_slope'
        * 'washington_inshore'
        * 'washington_shelf'
        * 'washington_offshore'
        * 'axial_base'
    limit : int
        maximum number of data points returned in one request. The limit
        applies for each instrument separately. That is the final list
        of data points can contain more samples then indicated by limit
        if data from multiple CTDs is available at the given location
        and time. Default is 10,0000.
    only_profilers : bool
        Specifies whether only data from the water column profilers
        should be requested. Default is False
    delivery_method : str
        Specifies which delivery method is considered. For details
        please refer to http://oceanobservatories.org/glossary/. Options
        are:
        * 'auto' (default): automatically uses method that has data
            avaiilable
        * 'streamed': only consideres data that are streamed to shore
            via cable
        * 'telemetered': only consideres data that are streamed to shore
            via satellite
        * 'recovered': only consideres data that were reteived when the
            instrument was retreived

    Returns
    -------
    ctd_data : :class:`ooipy.ctd.basic.CtdProfile`
        object, where the data array is stored in the raw_data attribute. Each
        data sample consists of a dictionary of parameters measured by the CTD.

    """

    USERNAME, TOKEN = ooipy.request.authentification.get_authentification()
    # Sensor Inventory
    DATA_API_BASE_URL = "https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/"

    # Oregon Shelf
    if location == "oregon_shelf":
        url_list = [
            "CE02SHSP/SP001/08-CTDPFJ000/telemetered/ctdpf_j_cspp_instrument?",
            "CE02SHSP/SP001/08-CTDPFJ000/recovered_cspp/ctdpf_j_cspp_instrument_recovered?",
        ]
        if not only_profilers:
            url_list.extend(
                [
                    "CE02SHBP/LJ01D/06-CTDBPN106/streamed/ctdbp_no_sample?",
                    "CE02SHSM/RID27/03-CTDBPC000/telemetered/ctdbp_cdef_dcl_instrument?",
                ]
            )

    elif location == "oregon_offshore":
        url_list = [
            "CE04OSPS/SF01B/2A-CTDPFA107/streamed/ctdpf_sbe43_sample?",
            "CE04OSPD/DP01B/01-CTDPFL105/recovered_inst/dpc_ctd_instrument_recovered?",
            "CE04OSPD/DP01B/01-CTDPFL105/recovered_wfp/dpc_ctd_instrument_recovered?",
        ]
        if not only_profilers:
            url_list.extend(
                [
                    "CE04OSPS/PC01B/4A-CTDPFA109/streamed/ctdpf_optode_sample?",
                    "CE04OSSM/RID27/03-CTDBPC000/telemetered/ctdbp_cdef_dcl_instrument?",
                    "CE04OSBP/LJ01C/06-CTDBPO108/streamed/ctdbp_no_sample?",
                ]
            )

    elif location == "oregon_slope":
        url_list = [
            "RS01SBPD/DP01A/01-CTDPFL104/recovered_inst/dpc_ctd_instrument_recovered?",
            "RS01SBPD/DP01A/01-CTDPFL104/recovered_wfp/dpc_ctd_instrument_recovered?",
            "RS01SBPS/SF01A/2A-CTDPFA102/streamed/ctdpf_sbe43_sample?",
        ]
        if not only_profilers:
            url_list.extend(
                [
                    "RS01SBPS/PC01A/4A-CTDPFA103/streamed/ctdpf_optode_sample?",
                    "RS01SLBS/LJ01A/12-CTDPFB101/streamed/ctdpf_optode_sample?",
                ]
            )

    elif location == "oregon_inshore":
        url_list = [
            "CE01ISSP/SP001/09-CTDPFJ000/recovered_cspp/ctdpf_j_cspp_instrument_recovered?",
            "CE01ISSP/SP001/09-CTDPFJ000/telemetered/ctdpf_j_cspp_instrument?",
        ]
        if not only_profilers:
            url_list.extend(
                [
                    "CE01ISSM/SBD17/06-CTDBPC000/telemetered/ctdbp_cdef_dcl_instrument?",
                    "CE01ISSM/SBD17/06-CTDBPC000/recovered_inst/"
                    + "ctdbp_cdef_instrument_recovered?",
                    "CE01ISSM/RID16/03-CTDBPC000/recovered_inst/"
                    + "ctdbp_cdef_instrument_recovered?",
                    "CE01ISSM/RID16/03-CTDBPC000/telemetered/ctdbp_cdef_dcl_instrument?",
                    "CE01ISSM/MFD37/03-CTDBPC000/recovered_inst/"
                    + "ctdbp_cdef_instrument_recovered?",
                    "CE01ISSM/MFD37/03-CTDBPC000/telemetered/ctdbp_cdef_dcl_instrument?",
                ]
            )

    elif location == "washington_inshore":
        url_list = [
            "CE06ISSP/SP001/09-CTDPFJ000/telemetered/ctdpf_j_cspp_instrument?",
            "CE06ISSP/SP001/09-CTDPFJ000/recovered_cspp/ctdpf_j_cspp_instrument_recovered?",
        ]
        if not only_profilers:
            url_list.extend(
                [
                    "CE06ISSM/SBD17/06-CTDBPC000/recovered_host/"
                    + "ctdbp_cdef_dcl_instrument_recovered?",
                    "CE06ISSM/SBD17/06-CTDBPC000/recovered_inst/"
                    + "ctdbp_cdef_instrument_recovered?",
                    "CE06ISSM/SBD17/06-CTDBPC000/telemetered/ctdbp_cdef_dcl_instrument?",
                    "CE06ISSM/RID16/03-CTDBPC000/recovered_inst/"
                    + "ctdbp_cdef_instrument_recovered?",
                    "CE06ISSM/RID16/03-CTDBPC000/telemetered/ctdbp_cdef_dcl_instrument?",
                    "CE06ISSM/MFD37/03-CTDBPC000/telemetered/ctdbp_cdef_dcl_instrument?",
                    "CE06ISSM/MFD37/03-CTDBPC000/recovered_inst/"
                    + "ctdbp_cdef_instrument_recovered?",
                ]
            )

    elif location == "washington_shelf":
        url_list = ["CE07SHSP/SP001/08-CTDPFJ000/recovered_cspp/ctdpf_j_cspp_instrument_recovered?"]
        if not only_profilers:
            url_list.extend(
                [
                    "CE07SHSM/RID27/03-CTDBPC000/telemetered/ctdbp_cdef_dcl_instrument?",
                    "CE07SHSM/RID27/03-CTDBPC000/recovered_inst/"
                    + "ctdbp_cdef_instrument_recovered?",
                    "CE07SHSM/MFD37/03-CTDBPC000/telemetered/ctdbp_cdef_dcl_instrument?",
                    "CE07SHSM/MFD37/03-CTDBPC000/recovered_inst/"
                    + "ctdbp_cdef_instrument_recovered?",
                ]
            )

    elif location == "washington_offshore":
        url_list = [
            "CE09OSPM/WFP01/03-CTDPFK000/telemetered/ctdpf_ckl_wfp_instrument?",
            "CE09OSPM/WFP01/03-CTDPFK000/recovered_wfp/ctdpf_ckl_wfp_instrument_recovered?",
        ]
        if not only_profilers:
            url_list.extend(
                [
                    "CE09OSSM/RID27/03-CTDBPC000/telemetered/ctdbp_cdef_dcl_instrument?",
                    "CE09OSSM/RID27/03-CTDBPC000/recovered_inst/"
                    + "ctdbp_cdef_instrument_recovered?",
                    "CE09OSSM/MFD37/03-CTDBPE000/telemetered/ctdbp_cdef_dcl_instrument?",
                    "CE09OSSM/MFD37/03-CTDBPE000/recovered_inst/"
                    + "ctdbp_cdef_instrument_recovered?",
                ]
            )

    elif location == "axial_base":
        url_list = [
            "RS03AXPD/DP03A/01-CTDPFL304/recovered_inst/dpc_ctd_instrument_recovered?",
            "RS03AXPD/DP03A/01-CTDPFL304/recovered_wfp/dpc_ctd_instrument_recovered?",
            "RS03AXPS/SF03A/2A-CTDPFA302/streamed/ctdpf_sbe43_sample?",
        ]
        if not only_profilers:
            url_list.extend(
                [
                    "RS03AXPS/PC03A/4A-CTDPFA303/streamed/ctdpf_optode_sample?",
                    "RS03AXBS/LJ03A/12-CTDPFB301/streamed/ctdpf_optode_sample?",
                ]
            )

    else:
        raise Exception("Do not know given location.")

    # remove URLs if delivery method is specified
    url_list_method = []
    if (
        delivery_method == "telemetered"
        or delivery_method == "streamed"
        or delivery_method == "recovered"
    ):
        for url in url_list:
            if delivery_method in url:
                url_list_method.append(url)
    elif delivery_method == "auto":
        url_list_method = url_list
    else:
        raise Exception(
            "deliveriy method need to be either 'auto', "
            + "'telemetered', 'streamed', or 'recovered', but not "
            + str(delivery_method)
        )

    beginDT = start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    endDT = end_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    dataraw = []
    processed_ctds = []

    for url in url_list_method:
        ctd_id = __get_instrument_id(url)
        if ctd_id in processed_ctds:
            continue

        data_request_url = (
            DATA_API_BASE_URL
            + url
            + "beginDT="
            + beginDT
            + "&endDT="
            + endDT
            + "&limit="
            + str(limit)
        )

        r = requests.get(data_request_url, auth=(USERNAME, TOKEN))
        dataraw2 = r.json()
        if "message" not in dataraw2:
            dataraw.extend(dataraw2)
            processed_ctds.append(ctd_id)
    ctd_data = CtdData(raw_data=dataraw)
    return ctd_data


def __get_instrument_id(url_str):
    ctd_id_arr = url_str.split("/")[:3]
    ctd_id = ctd_id_arr[0] + "/" + ctd_id_arr[1] + "/" + ctd_id_arr[2]

    return ctd_id


def get_ctd_data_daily(
    datetime_day, location, limit=10000, only_profilers=False, delivery_method="auto"
):
    """
    Requests CTD data for specified day and location. The day is split
    into 24 1-hour periods and for each 1-hour period
    :func:`ooipy.ctd_request.get_ctd_data is called. The data for all
    1-hour periods are then concatednated in stored in a
    :class:`ooipy.ctd.basic.CtdProfile` object.

    Parameters
    ----------
    datetime_day : datetime.datetime
        Day for which CTD data are requested
    location : str
        See :func:`ooipy.ctd_request.get_ctd_data
    limit : int
        See :func:`ooipy.ctd_request.get_ctd_data
    only_profilers : bool
        See :func:`ooipy.ctd_request.get_ctd_data
    delivery_method : str
        See :func:`ooipy.ctd_request.get_ctd_data

    Returns
    -------
    :class:`ooipy.ctd.basic.CtdProfile` object, where the data array is
    stored in the raw_data attribute. Each data sample consists of a
    dictionary of parameters measured by the CTD.
    """
    # get CTD data for one hour
    year = datetime_day.year
    month = datetime_day.month
    day = datetime_day.day

    start_end_list = []

    for hour in range(24):
        start = datetime.datetime(year, month, day, hour, 0, 0)
        end = datetime.datetime(year, month, day, hour, 59, 59, 999)
        start_end_list.append((start, end))

    raw_data_arr = __map_concurrency(
        __get_ctd_data_concurrent,
        start_end_list,
        {
            "location": location,
            "limit": limit,
            "only_profilers": only_profilers,
            "delivery_method": delivery_method,
        },
    )

    raw_data_falttened = []
    for item in raw_data_arr:
        if item is None:
            continue
        else:
            raw_data_falttened.extend(item.raw_data)

    return CtdData(raw_data=raw_data_falttened)


def __map_concurrency(func, iterator, args=(), max_workers=-1):
    """
    Helper function to support multiprocessing for
    :func:`ooipy.ctd_request.get_ctd_data_daily
    """
    # automatically set max_workers to 2x(available cores)
    if max_workers == -1:
        max_workers = 2 * mp.cpu_count()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(func, i, **args): i for i in iterator}
        for future in concurrent.futures.as_completed(future_to_url):
            data = future.result()
            results.append(data)
    return results


def __get_ctd_data_concurrent(start_end_tuple, location, limit, only_profilers, delivery_method):
    """
    Helper function to support multiprocessing for
    :func:`ooipy.ctd_request.get_ctd_data_daily
    """
    start = start_end_tuple[0]
    end = start_end_tuple[1]

    rawdata = get_ctd_data(
        start,
        end,
        location=location,
        limit=limit,
        only_profilers=only_profilers,
        delivery_method=delivery_method,
    )
    return rawdata
