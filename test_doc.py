"""
This script walks through how to use the experiment manager to load, retrieve
and analyze data. Note that the data is loaded lazily which means that the data
is not load into the memory until needed. Some functions return the data from
the file but do not keep the data. This is primarily for plotting purposes.
"""
#%%
from pyphotometry.acquisition.photometry import ExpManager
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from scipy import signal
import numpy as np
from numpy.polynomial import Polynomial
import scipy as scp

#%%
path1 = r"D:/Photometry/2022_07_07_photometry/1039085_N_Session25_PhotoData.csv"
path2 = r"D:/Photometry/2022_07_07_photometry/1039085_N_Session5_PhotoData.csv"
path = (
    r"/Volumes/Backup/Photometry/2022_07_07_photometry/1039085_N_Session5_PhotoData.csv"
)

#%%
manager = ExpManager()
manager.load_data(path)
# manager.load_data(path2)

#%%
# Show the experiments contained within the experiment manager
exps = manager.get_exps()
print(exps)

#%%
# Show the data the is available to analyze
acq_list = manager.get_acq_list("1039085_N_Session5_PhotoData")
print(acq_list)

# #%%
# # Retrieve the raw data for plotting.
# data = manager.get_raw_data("1039085_N_Session25_PhotoData", "timestamp")

# %%
manager.analyze_acq(
    exp="1039085_N_Session5_PhotoData",
    array="RawF_560_F1",
    analysis="min_peak",
    sec_array="None",
    time_array="timestamp",
    smooth=10,
    find_peaks=True,
    find_tau=True,
    peak_threshold=2,
    peak_direction="positive",
    z_score=False,
    z_threshold=150,
    filter_array="df_f",
    filter_type="ewma",
    input_1=10,
    input_2=0.85,
)

#%%
ntuple5 = manager.get_analyzed_data("1039085_N_Session5_PhotoData", "RawF_560_F1")
# ntuple25 = manager.get_analyzed_data("1039085_N_Session25_PhotoData", "RawF_465_F1")

#%%
f, t, sx = scp.signal.spectrogram(ntuple5.df_f, 30)
