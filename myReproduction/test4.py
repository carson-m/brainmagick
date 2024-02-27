import mne

data = mne.io.read_raw_fif('../../Data/brennan2019/S01/meg-sr120-hp0-raw.fif')
layout = mne.channels.find_layout(data.info)
layout.plot()
print(data.info.ch_names)