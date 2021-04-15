"""
This modules provides some useful functions for saving spectrograms and
power spectral density objects.
"""

import json
import pickle

# Import all dependancies
import numpy as np

from ooipy.hydrophone.basic import Psd, Spectrogram


def save(save_obj, filename, **kwargs):
    """
    Save :class:`ooipy.hydrophone.basic.Spectrogram` and
    :class:`ooipy.hydrophone.basic.Psd` objects. Spectrograms along
    with ancillary data will be saved in a pickle file as a dictionary.
    PSD object along with ancillary data will be saved in a json file as
    a dictionary.

    Parameters
    ----------
    save_obj :
        :class:`ooipy.hydrophone.basic.Psd` or
        :class:`ooipy.hydrophone.basic.Spectrogram` object to be saved
    filename : str
        name of file.
    kwargs : int, float, or list for Psd and arbitrary for spectrogram
        Ancillary data saved with the objects. Keyword and value
        will be saved as dictionary entries
    """

    if isinstance(save_obj, Spectrogram):
        save_spectrogram(save_obj, filename, **kwargs)
    elif isinstance(save_obj, Psd):
        save_psd(save_obj, filename, **kwargs)
    else:
        raise Exception("Fuction only supports spectrogram and PSD objects.")


def save_psd(psd_obj, filename, **kwargs):
    """
    Save a :class:`ooipy.hydrophone.basic.Psd` object in a json file.

    Parameters
    ----------
    spec_obj : :class:`ooipy.hydrophone.basic.Psd`
        Psd object to be saved
    filename : str
        name of file.
    kwargs : int, float, or list
        Ancillary data saved with the PSD object. Keyword and value
        will be saved as dictionary entries
    """

    if len(psd_obj.freq) != len(psd_obj.values):
        f = np.linspace(0, len(psd_obj.values) - 1, len(psd_obj.values))
    else:
        f = psd_obj.freq

    if not isinstance(psd_obj.values, list):
        values = psd_obj.values.tolist()

    if not isinstance(f, list):
        f = f.tolist()

    dct = {"psd": values, "values": f}

    # store ancillary data in dictionary
    for key, value in kwargs.items():
        if isinstance(value, int) or isinstance(value, float):
            dct[key] = value
        elif not isinstance(value, list):
            dct[key] = value.tolist()
        else:
            dct[key] = value

    with open(filename, "w+") as outfile:
        json.dump(dct, outfile)


def save_spectrogram(spectrogram_obj, filename, **kwargs):
    """
    Save a :class:`ooipy.hydrophone.basic.Spectrogram` object in a
    pickle file.

    Parameters
    ----------
    spectrogram_obj : :class:`ooipy.hydrophone.basic.Spectrogram`
        Spectrogram object to be saved
    filename : str
        name of file.
    kwargs : any datatype
        Ancillary data saved with the Spectrogram object. Keyword and
        value will be saved as dictionary entries
    """

    dct = {
        "time": spectrogram_obj.time,
        "frequency": spectrogram_obj.freq,
        "values": spectrogram_obj.values,
    }

    for key, value in kwargs.items():
        dct[key] = value

    with open(filename, "wb") as outfile:
        pickle.dump(dct, outfile)
