
# ===================== YOUR SCRIPT =====================
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from data_preprocessing import Data
import pywt

# === Load data ===
data_path = ""
start = 0
dt = 1


Data_instance = Data(data_path, start)
inside_air_org, inside_water_org, T_supply_org, T_return_org, T_outside_org, status_org = Data_instance.get_data()  # numpy arrays

mask = inside_water_org >= 4.0


inside_air, inside_water, T_supply, T_return, T_outside, status = inside_air_org[mask], inside_water_org[mask], T_supply_org[mask], T_return_org[mask], T_outside_org[mask], status_org[mask]



data_path = 'data_train0'
dt = 1
# Save
np.save("./" + data_path + "/inside_air_denoised.npy",   inside_air[::dt])
np.save("./" + data_path + "/inside_water_denoised.npy", inside_water[::dt])
np.save("./" + data_path + "/T_supply_denoised.npy",     T_supply[::dt])
np.save("./" + data_path + "/T_return_denoised.npy",     T_return[::dt])
np.save("./" + data_path + "/T_outside.npy",             T_outside[::dt])
np.save("./" + data_path + "/status.npy", status[::dt])


