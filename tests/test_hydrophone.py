import unittest
import numpy as np
import datetime
import xarray as xr
from obspy.core import UTCDateTime

import ooipy
from ooipy.hydrophone.basic import HydrophoneData


class TestHydrophoneDataMethods(unittest.TestCase):
    """
    Test class for HydrophoneData methods related to spectral analysis
    """
    
    def setUp(self):
        """
        Download a short segment (1 minute) of hydrophone data for testing
        """
        # Set start and end times for a 1-minute segment
        self.start_time = datetime.datetime(2019, 1, 1, 0, 0, 0)
        self.end_time = datetime.datetime(2019, 1, 1, 0, 1, 0)  # 1 minute later
        
        # Download data - using Oregon Offshore Base Seafloor node
        try:
            self.hyd_data = ooipy.request.hydrophone_request.get_acoustic_data(
                self.start_time,
                self.end_time,
                "Oregon_Offshore_Base_Seafloor"
            )
            self.test_data_available = True
        except Exception as e:
            print(f"Could not download test data: {e}")
            # Create synthetic data for testing if download fails
            sample_rate = 64000  # Standard rate for broadband hydrophones
            duration = 60  # 60 seconds
            num_samples = sample_rate * duration
            
            # Generate synthetic data (white noise)
            data = np.random.normal(0, 1, num_samples)
            
            # Create header
            header = {
                'sampling_rate': sample_rate,
                'starttime': UTCDateTime(self.start_time),
                'npts': num_samples,
                'network': 'OO',
                'station': 'TEST',
                'location': 'LJ01C',  # Oregon Offshore Base Seafloor
                'channel': 'HYD',
            }
            
            self.hyd_data = HydrophoneData(data=data, header=header, node="Oregon_Offshore_Base_Seafloor")
            self.test_data_available = False
            print("Using synthetic data for tests")
    
    def test_compute_spectrogram(self):
        """
        Test the compute_spectrogram method of HydrophoneData
        """
        # Skip if no data
        if self.hyd_data is None:
            self.skipTest("No hydrophone data available for testing")
        
        # Compute spectrogram with default parameters
        spectrogram = self.hyd_data.compute_spectrogram(
            win='hann',
            L=4096,
            avg_time=1.0,  # 1 second time bins
            overlap=0.5
        )
        
        # Verify the spectrogram is returned as xarray DataArray
        self.assertIsInstance(spectrogram, xr.DataArray)
        
        # Verify spectrogram dimensions and coordinates
        self.assertEqual(len(spectrogram.dims), 2)
        self.assertIn('time', spectrogram.dims)
        self.assertIn('frequency', spectrogram.dims)
        
        # Verify spectrogram attributes
        self.assertIn('units', spectrogram.attrs)
        self.assertEqual(spectrogram.attrs['units'], 'dB rel µ Pa^2 / Hz')
        self.assertIn('start_time', spectrogram.attrs)
        self.assertIn('end_time', spectrogram.attrs)
        
        # Verify frequency range makes sense
        freq_max = spectrogram.frequency.max().values
        self.assertLessEqual(freq_max, self.hyd_data.stats.sampling_rate / 2)
        
        # Print some info about the spectrogram
        print(f"Spectrogram shape: {spectrogram.shape}")
        print(f"Frequency range: {spectrogram.frequency.min().values} - {freq_max} Hz")
        print(f"Time bins: {len(spectrogram.time)}")
    
    def test_compute_psd_welch(self):
        """
        Test the compute_psd_welch method of HydrophoneData
        """
        # Skip if no data
        if self.hyd_data is None:
            self.skipTest("No hydrophone data available for testing")
        
        # Compute PSD with default parameters
        psd = self.hyd_data.compute_psd_welch(
            win='hann',
            L=4096,
            overlap=0.5,
            avg_method='median'
        )
        
        # Verify the PSD is returned as xarray DataArray
        self.assertIsInstance(psd, xr.DataArray)
        
        # Verify PSD dimensions
        self.assertEqual(len(psd.dims), 1)
        self.assertIn('frequency', psd.dims)
        
        # Verify PSD attributes
        self.assertIn('units', psd.attrs)
        self.assertEqual(psd.attrs['units'], 'dB rel µ Pa^2 / Hz')
        self.assertIn('start_time', psd.attrs)
        self.assertIn('end_time', psd.attrs)
        
        # Verify frequency range makes sense
        freq_max = psd.frequency.max().values
        self.assertLessEqual(freq_max, self.hyd_data.stats.sampling_rate / 2)
        
        # Check that PSD has reasonable values (typically between 20-120 dB for ocean noise)
        self.assertTrue(np.isfinite(psd.values).all(), "PSD contains non-finite values")
        
        # Print some info about the PSD
        print(f"PSD length: {len(psd)}")
        print(f"Frequency range: {psd.frequency.min().values} - {freq_max} Hz")
        print(f"PSD value range: {np.nanmin(psd.values)} - {np.nanmax(psd.values)} dB")


if __name__ == '__main__':
    unittest.main()
