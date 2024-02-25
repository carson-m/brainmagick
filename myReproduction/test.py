import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import torch
import multiprocessing
from utils import to_idx
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

USE_CUDA = True
use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
maximum_thread = multiprocessing.cpu_count()
print('Available threads:', maximum_thread)
num_workers = 1
print('Num workers:', num_workers)

# params
delta_val = 0.5
tmin = -0.5 # start of each epoch (500ms before the trigger)
tmax = 2.5 # end of each epoch (2500ms after the trigger)
time_shift = 0.15 # shift the EEG signal 150ms to account for the delay in the audio
condition = 3.0

# Load Data
data = mne.io.read_raw_fif('../../Data/brennan2019/S01/meg-sr120-hp0-raw.fif')
sample_rate = data.info['sfreq']
times = np.arange(0, data.times[-1], condition)
mask = np.logical_and((times + tmin) >= 0, (times + tmax) < data.times[-1])
samples = to_idx(times[mask], sample_rate)
# print(raw.info)

delta = delta_val / sample_rate

mne_events = np.concatenate([samples[:, None], np.ones((len(samples), 2), dtype=np.int64)], 1)

epochs = mne.Epochs(data, mne_events, tmin=tmin, tmax=tmax, baseline=None, event_repeated='drop', preload=True)

epochs.plot(events = mne_events, event_color="red")

epochs.shift_time(tshift = time_shift, relative = True)

epochs.copy().plot(events = mne_events, event_color="blue")

input('Press Enter to continue...')