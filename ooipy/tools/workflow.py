# Import all dependancies
import numpy as np
from ooipy.hydrophone.basic import Spectrogram, Psd
import json
import pickle

def save(save_obj, filename, **kwargs):
    """
    Save spectrogram and PSD object. Spectrograms will be saved in a pickle file as a dictionary.
    PSDs will be saved in a json file as a dictionary. Ancillary data can be added to both dictionaries
    """
    if isinstance(save_obj, Spectrogram):
        save_spectrogram(save_obj, filename, **kwargs)
    elif isinstance(save_obj, Psd):
        save_psd(save_obj, filename, **kwargs)
    else:
        print('Fuction only supports spectrogram and PSD objects.')

def save_psd(psd_obj, filename, **kwargs):
    '''
    Save PSD estimates along with with ancillary data (stored in dictionary) in json file.

    filename (str): directory for saving the data
    ancillary_data ([array like]): list of ancillary data
    ancillary_data_label ([str]): labels for ancillary data used as keys in the output dictionary.
        Array has same length as ancillary_data array.
    '''

    if len(psd_obj.freq) != len(psd_obj.values):
        f = np.linspace(0, len(psd_obj.values)-1, len(psd_obj.values))
    else:
        f = psd_obj.freq

    if  not isinstance(psd_obj.values, list):
        values = psd_obj.values.tolist()

    if not isinstance(f, list):
        f = f.tolist()

    dct = {
        'psd': values,
        'values': f
        }

    for key, value in kwargs.items():
        if isinstance(value, int) or isinstance(value, float):
            dct[key] = value
        elif not isinstance(value, list):
            dct[key] = value.tolist()
        else:
            dct[key] = value

    with open(filename, 'w+') as outfile:
        json.dump(dct, outfile)

def save_spectrogram(spectrogram_obj, filename, **kwargs):
        '''
        Save spectrogram in pickle file.

        filename (str): directory where spectrogram data is saved. Ending has to be ".pickle".
        '''

        dct = {
            'time': spectrogram_obj.time,
            'frequency': spectrogram_obj.freq,
            'values': spectrogram_obj.values
            }

        for key, value in kwargs.items():
            dct[key] = value

        with open(filename, 'wb') as outfile:
            pickle.dump(dct, outfile)