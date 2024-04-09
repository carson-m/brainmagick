import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne

data = mne.io.read_raw_fif('../../Data/brennan2019/S01/meg-sr120-hp0-raw.fif', preload=True)
mne_info = data.info
scores = np.load('sa_scores.npy')
scores = scores.squeeze().mean(axis=0)
mne.viz.plot_topomap(scores, mne_info)