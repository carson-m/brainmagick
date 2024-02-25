import numpy as np
import mne as mne
import pandas as pd
import torchaudio

class dataHandler:
    def __init__(self, raw_pth, events_pth, tmin, tmax, time_shift, window_duration):
        self.tmin = tmin # s
        self.tmax = tmax # s
        self.time_shift = time_shift # s
        self.window_duration = window_duration
        self.data = mne.io.read_raw_fif(raw_pth)
        self.audio = None
        self.mne_events = pd.read_csv(events_pth)
        self.brain_srate = self.data.info['sfreq']
        self.audio_times = np.array([])
        self.brain_times = np.array([])
        self.brain_epochs = None
        self.audio_epochs = None
        self.audio_embeddings = None
        self.brain_segments = None # EEG/MEG segments (# epochs, # channels, # samples)
        
    def get_times(self): # get the times of the audio and brain data
        sound_mask = self.mne_events['kind']=='sound'
        sound_starts = self.mne_events['start'].values[sound_mask]
        sound_durations = self.mne_events['duration'].values[sound_mask]
        sound_stops = sound_starts + sound_durations
        sound_starts = sound_starts - self.time_shift # shift the EEG signal 150ms to account for latency
        sound_stops = sound_stops - self.time_shift # shift the EEG signal 150ms to account for latency
        for i in range(len(sound_starts)):
            brain_times_tmp = np.arange(sound_starts[i], sound_stops[i], self.window_duration)
            audio_times_tmp = np.arange(0, sound_durations[i], self.window_duration)
            mask_tmp = np.logical_and((brain_times_tmp + self.tmin) >= sound_starts[i], (brain_times_tmp + self.tmax) < sound_stops[i])
            brain_times_tmp = brain_times_tmp[mask_tmp]
            audio_times_tmp = audio_times_tmp[mask_tmp]
            self.brain_times = np.append(self.brain_times, brain_times_tmp)
            self.audio_times = np.append(self.audio_times, audio_times_tmp)
    
    def get_brain_segments(self): # get EEG/MEG segments
        events = np.concatenate([self.brain_times[:, None], np.ones((len(self.brain_times), 2), dtype=np.int64)], 1)
        self.brain_epochs = mne.Epochs(self.data, events, tmin=self.tmin, tmax=self.tmax, baseline=(self.tmin, 0), event_repeated='drop', preload=False)
        self.brain_segments = self.brain_epochs.get_data()
    
    def get_audio_epochs(self, audio_pth):
        