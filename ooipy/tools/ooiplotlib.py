'''
This modules provides functions for plotting spectrograms and power
spectral density estimates. It extends the matplotlib.pyplot.plot
function.
'''

# Import all dependancies
import datetime
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from ooipy.hydrophone.basic import Spectrogram, Psd
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
from obspy.core import UTCDateTime


def plot(*args, scalex=True, scaley=True, data=None, **kwargs):
    """
    An extension to the matplotlib.pyplot.plot function that allows for
    the nice plotting of :calss:`~ooipy.hydrophone.basic.Spectrogram`
    and :calss:`~ooipy.hydrophone.basic.Psd` objects. For a description
    of the input parameters, please refer to the matplotlib
    documentation.

    Parameters
    ----------
    *args : :calss:`~ooipy.hydrophone.basic.Spectrogram`, or
    :calss:`~ooipy.hydrophone.basic.Psd`, or array
        object or array to be plotted
    scalex :
        see matplotlib documentation
    scaley :
        see matplotlib doccumentation
    data :
        see matplotlib doccumentation
    **kwargs :
        see matplotlib doccumentation,
        :func:`~ooipy.tools.ooiplotlib.plot_spectrogram`, and
        :func:`~ooipy.tools.ooiplotlib.plot_psd` for arguments

    >>> from ooipy.request import hydrophone_request
    >>> from ooipy.tools import ooiplotlib as ooiplt

    >>> # pull OOI data and compute spectrogram and PSD
    >>> start_time = datetime.datetime(2017,3,10,0,0,0)
    >>> end_time = datetime.datetime(2017,3,10,0,5,0)
    >>> node = '/PC01A'
    >>> hydrophone_data = hydrophone_request.get_acoustic_data(
            start_time, end_time, node)
    >>> hydrophone_data.compute_spectrogram()
    >>> hydrophone_data.compute_psd_welch()

    >>> # plot spectrogram and change some default plot settings
    >>> ooiplt.plot(hydrophone_data.spectrogram)
    >>> plt.title('my spectrogram')
    >>> plt.ylim([0, 20000])
    >>> plt.show()

    >>> # plot PSD and chnage some default plot settings
    >>> ooiplt.plot(hydrophone_data.psd)
    >>> plt.title('my PSD')
    >>> plt.show()
    """
    for arg in args:
        if isinstance(arg, Spectrogram):
            plot_spectrogram(arg, **kwargs)
        elif isinstance(arg, Psd):
            plot_psd(arg, **kwargs)
        else:
            plt.gca().plot(arg, scalex=scalex, scaley=scaley,
            **({"data": data} if data is not None else {}), **kwargs)

def plot_spectrogram(spec_obj, **kwargs):
    """
    Plot a :calss:`~ooipy.hydrophone.basic.Spectrogram` object using the
    matplotlib package.

    Parameters
    ----------
    spec_obj : :calss:`~ooipy.hydrophone.basic.Spectrogram`
        spectrogram object to be plotted
    **kwargs :
        See matplotlib doccumentation for list of arguments. Additional
        arguments are:
        * plot_spec : bool
            If False, figure will be closed. Can save time if only
            saving but not plotting is desired. Default is True
        * save_spec : bool
            If True, figure will be saved under **filename**. Default is
            False
        * filename : str
            filename of figure if saved. Default is "spectrogram.png"
        * xlabel_rot : int or float
            rotation angle (deg) of x-labels. Default is 70
        * fmin : int or float
            minimum frequency. Default is 0
        * fmax : int or float
            maximum frequency. Default is 32000
        * vmin : int or float
            lower limit of level axis (colormap). Default is 20
        * vmax : int or float
            upper limit of level axis (colormap). Default is 80
        * vdelta : int or float
            resolution of level axis (colormap). Default is 1
        * vdelta_cbar : int or float
            label distance of colorbar. Default is 5
        * figsize : (int, int)
            width and height of figure, Default is (16, 9)
        * res_reduction_time : int
            reduction factor of time domain resolution. This can
            facilitate faster plotting of large spectroagm objects.
            Default is 1 (no reduction)
        * res_reduction_freq : int
            reduction factor of frequency domain resolution. This can
            facilitate faster plotting of large spectroagm objects.
            Default is 1 (no reduction)
    """
    # check for keys
    if 'plot_spec' not in kwargs: plot_spec = True
    if 'save_spec' not in kwargs: save_spec = False
    if 'filename' not in kwargs: filename = 'spectrogram.png'
    if 'title' not in kwargs: title = 'Spectrogram'
    if 'xlabel' not in kwargs: xlabel = 'time'
    if 'xlabel_rot' not in kwargs: xlabel_rot = 70
    if 'ylabel' not in kwargs: ylabel = 'frequency'
    if 'fmin' not in kwargs: fmin = 0.0
    if 'fmax' not in kwargs: fmax = 32000.0
    if 'vmin' not in kwargs: vmin = 20.0
    if 'vmax' not in kwargs: vmax = 80.0
    if 'vdelta' not in kwargs: vdelta = 1.0
    if 'vdelta_cbar' not in kwargs: vdelta_cbar = 5.0
    if 'figsize' not in kwargs: figsize = (16, 9)
    if 'res_reduction_time' not in kwargs: res_reduction_time = 1
    if 'res_reduction_freq' not in kwargs: res_reduction_freq = 1

    # set backend for plotting/saving:
    if not plot_spec: matplotlib.use('Agg')
    font = {'size'   : 22}
    matplotlib.rc('font', **font)

    # reduce resolution in time and frequency
    v = spec_obj.values[::res_reduction_time,::res_reduction_freq]
    if len(spec_obj.time) != len(spec_obj.values):
        t = np.linspace(0, len(spec_obj.values) - 1,
                        int(len(spec_obj.values) / res_reduction_time))
    else:
        t = spec_obj.time[::res_reduction_time]
    if len(spec_obj.freq) != len(spec_obj.values[0]):
        f = np.linspace(0, len(spec_obj.values[0]) - 1,
                        int(len(spec_obj.values[0]) / res_reduction_freq))
    else:
        f = spec_obj.freq[::res_reduction_freq]

    # plot spectrogram object
    cbarticks = np.arange(vmin,vmax+vdelta,vdelta)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.contourf(t, f, np.transpose(v), cbarticks,
                     norm=Normalize(vmin=vmin, vmax=vmax), cmap=plt.cm.jet,
                     **kwargs)  
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim([fmin, fmax])
    plt.xticks(rotation=xlabel_rot)
    plt.title(title)
    plt.colorbar(im, ax=ax, ticks=np.arange(vmin, vmax+vdelta, vdelta_cbar))
    plt.tick_params(axis='y')

    if isinstance(t[0],  datetime.datetime) or isinstance(t[0], UTCDateTime):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H:%M'))
    
    if save_spec:
        plt.savefig(filename, bbox_inches='tight')

    if not plot_spec: plt.close(fig)

def plot_psd(psd_obj, **kwargs):
    """
    Plot a :calss:`~ooipy.hydrophone.basic.Psd` object using the
    matplotlib package.

    Parameters
    ----------
    spec_obj : :calss:`~ooipy.hydrophone.basic.Psd`
        Psd object to be plotted
    **kwargs :
        See matplotlib doccumentation for list of arguments. Additional
        arguments are:
        * plot_spec : bool
            If False, figure will be closed. Can save time if only
            saving but not plotting is desired. Default is True
        * save_spec : bool
            If True, figure will be saved under **filename**. Default is
            False
        * new_fig : bool
            If True, matplotlib will create a new fugure. Default is
            True
        * filename : str
            filename of figure if saved. Default is "spectrogram.png"
        * xlabel_rot : int or float
            rotation angle (deg) of x-labels. Default is 70
        * fmin : int or float
            minimum frequency. Default is 0
        * fmax : int or float
            maximum frequency. Default is 32000
        * vmin : int or float
            lower limit of level axis (colormap). Default is 20
        * vmax : int or float
            upper limit of level axis (colormap). Default is 80
        * figsize : (int, int)
            width and height of figure, Default is (16, 9)
    """

    # check for keys
    if 'plot_psd' not in kwargs: plot_psd = True
    if 'save_psd' not in kwargs: save_psd = False
    if 'new_fig' not in kwargs: new_fig = True
    if 'filename' not in kwargs: filename = 'psd.png'
    if 'title' not in kwargs: title = 'PSD'
    if 'xlabel' not in kwargs: xlabel = 'frequency'
    if 'xlabel_rot' not in kwargs: xlabel_rot = 0
    if 'ylabel' not in kwargs: ylabel = 'spectral level'
    if 'fmin' not in kwargs: fmin = 0.0
    if 'fmax' not in kwargs: fmax = 32000.0
    if 'vmin' not in kwargs: vmin = 20.0
    if 'vmax' not in kwargs: vmax = 80.0
    if 'figsize' not in kwargs: figsize = (16,9)

    # set backend for plotting/saving:
    if not plot_psd: matplotlib.use('Agg')
    font = {'size'   : 22}
    matplotlib.rc('font', **font)

    if len(psd_obj.freq) != len(psd_obj.values):
        f = np.linspace(0, len(psd_obj.values)-1, len(psd_obj.values))
    else:
        f = psd_obj.freq

    # plot PSD object
    if new_fig:
        fig, ax = plt.subplots(figsize=figsize)
    plt.semilogx(f, psd_obj.values)  
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim([fmin, fmax])
    plt.ylim([vmin, vmax])
    plt.xticks(rotation=xlabel_rot)
    plt.title(title)
    plt.grid(True)
    
    if save_psd:
        plt.savefig(filename, bbox_inches='tight')

    if not plot_psd: plt.close(fig)