"""
This modules provides functions for plotting spectrograms and power
spectral density estimates. It extends the matplotlib.pyplot.plot
function.
"""

# Import all dependencies
import datetime

import matplotlib
import matplotlib.dates as mdates
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from obspy.core import UTCDateTime

from ooipy.ctd.basic import CtdProfile
from ooipy.hydrophone.basic import HydrophoneData

def plot_ctd_profile(ctd_profile, **kwargs):
    """
    Plot a :class:`ooipy.ctd.basic.CtdProfile` object using the
    matplotlib package.

    Parameters
    ----------
    ctd_profile : :class:`ooipy.ctd.basic.CtdProfile`
        CtdProfile object to be plotted
    kwargs :
        See matplotlib documentation for list of arguments. Additional
        arguments are

        * plot : bool
            If False, figure will be closed. Can save time if only
            saving but not plotting is desired. Default is True
        * save : bool
            If True, figure will be saved under **filename**. Default is
            False
        * new_fig : bool
            If True, matplotlib will create a new figure. Default is
            True
        * filename : str
            filename of figure if saved. Default is "spectrogram.png"
        * figsize : (int, int)
            width and height of figure. Default is (16, 9)
        * title : str
            Title of plot. Default is 'CTD profile'
        * xlabel : str
            x-axis label of plot. Default is 'parameter'
        * ylabel : str
            y-axis label of plot. Default is 'depth'
        * show_variance : bool
            Indicates whether the variance should be plotted or not.
            Default is True
        * min_depth : int or float
            upper limit of vertical axis (depth axis). Default is the
            maximum of max(min(ctd_profile.depth_mean) - 10, 0)
        * max_depth : int or float
            lower limit of vertical axis (depth axis). Default is
            max(ctd_profile.depth_mean) + 10
        * dpi : int
            dots per inch, passed to matplotlib figure.savefig()
        * fontsize : int
            fontsize of saved plot, passed to matplotlib figure
    """

    # check for keys
    if "plot" not in kwargs:
        kwargs["plot"] = True
    if "save" not in kwargs:
        kwargs["save"] = False
    if "new_fig" not in kwargs:
        kwargs["new_fig"] = True
    if "filename" not in kwargs:
        kwargs["filename"] = "ctd_profile.png"
    if "title" not in kwargs:
        kwargs["title"] = "CTD profile"
    if "xlabel" not in kwargs:
        kwargs["xlabel"] = "parameter"
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "depth"
    if "figsize" not in kwargs:
        kwargs["figsize"] = (16, 9)
    if "show_variance" not in kwargs:
        kwargs["show_variance"] = True
    if "linestyle" not in kwargs:
        kwargs["linestyle"] = "dashed"
    if "marker" not in kwargs:
        kwargs["marker"] = "o"
    if "markersize" not in kwargs:
        kwargs["markersize"] = 5
    if "color" not in kwargs:
        kwargs["color"] = "black"
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.5
    if "var_color" not in kwargs:
        kwargs["var_color"] = "gray"
    if "min_depth" not in kwargs:
        kwargs["min_depth"] = max(ctd_profile.depth_mean[0] - 10, 0)
    if "max_depth" not in kwargs:
        kwargs["max_depth"] = np.nanmax(ctd_profile.depth_mean) + 10
    if "dpi" not in kwargs:
        kwargs["dpi"] = 100
    if "fontsize" not in kwargs:
        kwargs["fontsize"] = 22

    # set backend for plotting/saving:
    if not kwargs["plot"]:
        matplotlib.use("Agg")
    font = {"size": kwargs["fontsize"]}
    matplotlib.rc("font", **font)

    # plot PSD object
    if kwargs["new_fig"]:
        fig, ax = plt.subplots(figsize=kwargs["figsize"])
    plt.plot(
        ctd_profile.parameter_mean,
        ctd_profile.depth_mean,
        linestyle=kwargs["linestyle"],
        marker=kwargs["marker"],
        markersize=kwargs["markersize"],
        color=kwargs["color"],
    )
    if kwargs["show_variance"]:
        y1 = ctd_profile.parameter_mean - 2 * np.sqrt(ctd_profile.parameter_var)
        y2 = ctd_profile.parameter_mean + 2 * np.sqrt(ctd_profile.parameter_var)
        plt.fill_betweenx(
            ctd_profile.depth_mean,
            y1,
            y2,
            alpha=kwargs["alpha"],
            color=kwargs["var_color"],
        )
    plt.ylim([kwargs["max_depth"], kwargs["min_depth"]])
    plt.ylabel(kwargs["ylabel"])
    plt.xlabel(kwargs["xlabel"])
    plt.title(kwargs["title"])
    plt.grid(True)

    if kwargs["save"]:
        plt.savefig(kwargs["filename"], bbox_inches="tight", dpi=kwargs["dpi"])

    if not kwargs["plot"]:
        plt.close(fig)
