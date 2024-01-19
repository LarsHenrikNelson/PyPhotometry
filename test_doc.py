"""
This script walks through how to use the experiment manager to load, retrieve
and analyze data. Note that the data is loaded lazily which means that the data
is not load into the memory until needed. Some functions return the data from
the file but do not keep the data. This is primarily for plotting purposes.
"""
# %%
from pyphotometry.acquisition.photometry import ExpManager
import matplotlib.pyplot as plt
from scipy import signal


# %%
path1 = r"<insert path to csv file here>"
path2 = r"<insert path to csv file here>"
path = r"<insert path to csv file here>"

# %%
manager = ExpManager()

# %%
manager.load_data(path1)
# manager.load_data(path2)

# %%
# Show the experiments contained within the experiment manager
exps = manager.get_exps()
print(exps)

# %%
# Show the data the is available to analyze
acq_list = manager.get_acq_list(exps[0])
print(acq_list)

# %%
# Retrieve the raw data for plotting.
data = manager.get_raw_data("FMR1 openfield_Session1_PhotoData", "RawF_465_F1")
x = manager.get_raw_data("FMR1 openfield_Session1_PhotoData", "timestamp")

# %%
manager.analyze_acq(
    exp=exps[0],
    array="RawF_560_F1",
    analysis="lsq_spline",
    sec_array="None",
    time_array="timestamp",
    smooth=10,
    find_peaks=True,
    find_tau=False,
    peak_threshold=2,
    peak_direction="positive",
    z_score=True,
    z_threshold=150,
    filter_array="df_f",
    filter_type="ewma",
    input_1=10,
    input_2=0.85,
)
# %%
of_560 = manager.get_analyzed_data(exps[0], "RawF_560_F1")
# of_560 = manager.get_analyzed_data(exps[1], "RawF_560_F1")

# %%
# w_465 = manager.get_analyzed_data(exps[0], "RawF_465_F1")
w_560 = manager.get_analyzed_data(exps[0], "RawF_560_F1")

# %%
# w2_465 = manager.get_analyzed_data(exps[0], "RawF_465_F1")
w2_560 = manager.get_analyzed_data(exps[0], "RawF_560_F1")

# %%
plt.plot(of_560.z_array[5000:10000], color="black")
plt.title("Openfield")

# %%
plt.plot(w2_560.z_array[5000:10000], color="black")
plt.title("Wheel")

# %%
f, t, sx = signal.spectrogram(ntuple5.df_f, 30)
