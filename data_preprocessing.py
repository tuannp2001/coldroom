import numpy as np
import pandas as pd
from numpy.fft import *
from scipy.signal import savgol_filter

def filter_signal(signal, threshold=1e8):
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)

# === Hàm lọc từng đoạn không làm trễ đỉnh/đáy ===
def filter_segment_savgol(segment, window=21, polyorder=3):
    if len(segment) < window:
        return savgol_filter(segment, window_length=len(segment), polyorder=polyorder)
    return savgol_filter(segment, window_length=window, polyorder=polyorder)

# === Hàm tách và ghép đoạn lọc lại theo status ===
def segment_and_filter(signal, status, window=21, polyorder=3):
    filtered_signal = np.copy(signal)
    n = len(signal)
    i = 0

    while i < n:
        current_status = status[i]
        j = i
        while j < n and status[j] == current_status:
            j += 1
        segment = signal[i:j]
        filtered_segment = filter_segment_savgol(segment, window, polyorder)
        filtered_signal[i:j] = filtered_segment
        i = j

    return filtered_signal



class Data():
    def __init__(self, data_path, start_time):
        self.data_path = data_path
        self.start_time = start_time
    def get_data(self):
        data = pd.read_excel(self.data_path)
        df =  data.iloc[self.start_time: ].copy()
        #h_in = df['h_in'].to_numpy()
        #h_out = df['h_out'].to_numpy()
        #top_water = df['top_water'].to_numpy()
        mi_water = df['center_water'].to_numpy()
        #bottom_water = df['bottom_water'].to_numpy()
        #water = (top_water + mi_water + bottom_water) / 3
        water = mi_water
        """
        inside_air0 = df['air_pres_eau0'].to_numpy()
        
        inside_air1 = df['air_pres_eau1'].to_numpy()
        inside_air2 = df['air_pres_eau2'].to_numpy()
        """
        inside_air0 = df['inside_air'].to_numpy()


        """
        air_fond0 = df['air_fond0'].to_numpy()
        air_fond1 = df['air_fond1'].to_numpy()
        air_fond2 = df['air_fond2'].to_numpy()
        
        air_fond3 = df['air_fond3'].to_numpy()
        
        #inside_air1 = df['inside_air1'].to_numpy()
        """
        outside_air = df['outside_air_window'].to_numpy()
        
        supply_air = df['supply_air'].to_numpy()
        return_air  = df['return_air'].to_numpy()
        #return_air1  = df['return_air1'].to_numpy()
        
    
        status_raw = df['status'].apply(lambda x: 1 if x >=25 else 0 )
        
        binary_status = status_raw.to_numpy()
        #inside_air0 = self.denoise_air_with_status(inside_air0, binary_status, window_size=5)

        #inside_air0_denoised = filter_signal(inside_air0, threshold=1e4)
        #inside_air0_denoised = segment_and_filter(inside_air0, binary_status, window=21, polyorder=3)
        
        

        return inside_air0, water, supply_air, return_air, outside_air, binary_status