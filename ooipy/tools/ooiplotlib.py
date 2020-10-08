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
    An extension to the matplotlib.pyplot.plot function that allows for the nice plotting of Spectrogram
    and PSD objects. For a description of the input parameters, please refer to matplotlib.
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
    Plot a spectrogram object using the matplotlib package. 
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
    if 'figsize' not in kwargs: figsize = (16,9)
    if 'res_reduction_time' not in kwargs: res_reduction_time = 1
    if 'res_reduction_freq' not in kwargs: res_reduction_freq = 1

    #set backend for plotting/saving:
    if not plot_spec: matplotlib.use('Agg')
    font = {'size'   : 22}
    matplotlib.rc('font', **font)

    v = spec_obj.values[::res_reduction_time,::res_reduction_freq]

    if len(spec_obj.time) != len(spec_obj.values):
        t = np.linspace(0, len(spec_obj.values) - 1, int(len(spec_obj.values) / res_reduction_time))
    else:
        t = spec_obj.time[::res_reduction_time]

    if len(spec_obj.freq) != len(spec_obj.values[0]):
        f = np.linspace(0, len(spec_obj.values[0]) - 1, int(len(spec_obj.values[0]) / res_reduction_freq))
    else:
        f = spec_obj.freq[::res_reduction_freq]

    cbarticks = np.arange(vmin,vmax+vdelta,vdelta)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.contourf(t, f, np.transpose(v), cbarticks, norm=Normalize(vmin=vmin, vmax=vmax), cmap=plt.cm.jet, **kwargs)  
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
    '''
    Plot a spectrogram object using the matplotlib package.

    Basic visualization of PSD estimate based on matplotlib. The function offers two options: Plot PSD
    in Python (plot_psd = True) and save PSD plot in directory (save_psd = True). PSDs are
    plotted in dB re 1µ Pa^2/Hz.

    plot_psd (bool): whether or not PSD is plotted using Python
    save_psd (bool): whether or not PSD plot is saved
    filename (str): directory where PSD plot is saved. Use ending ".png" or ".pdf" to save as PNG or PDF
        file. This value will be ignored if save_psd=False
    title (str): title of plot
    ylabel (str): label of vertical axis
    xlabel (str): label of horizontal axis
    xlabel_rot (float): rotation of xlabel. This is useful if xlabel are longer strings.
    fmin (float): minimum frequency (unit same as f) that is displayed
    fmax (float): maximum frequency (unit same as f) that is displayed
    vmin (float): minimum value (dB) of PSD.
    vmax (float): maximum value (dB) of PSD.
    figsize (tuple(int)): size of figure
    dpi (int): dots per inch
    '''

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

    #set backend for plotting/saving:
    if not plot_psd: matplotlib.use('Agg')
    font = {'size'   : 22}
    matplotlib.rc('font', **font)

    if len(psd_obj.freq) != len(psd_obj.values):
        f = np.linspace(0, len(psd_obj.values)-1, len(psd_obj.values))
    else:
        f = psd_obj.freq

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
