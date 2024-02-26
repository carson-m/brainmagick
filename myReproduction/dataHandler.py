import numpy as np
import mne as mne
import pandas as pd
import torchaudio
import torch
from utils import to_idx

class dataHandler:
    def __init__(self, bundle, device, raw_pth, events_pth, tmin, tmax, time_shift, window_duration):
        self.model = bundle.get_model().to(device)
        self.model_srate = bundle.sample_rate
        self.device = device
        self.tmin = tmin # s
        self.tmax = tmax # s
        self.time_shift = time_shift # s
        self.window_duration = window_duration
        self.data = mne.io.read_raw_fif(raw_pth)
        # self.audio = None
        self.mne_events = pd.read_csv(events_pth)
        self.brain_srate = self.data.info['sfreq']
        self.audio_times = list([])
        self.brain_times = np.array([])
        self.brain_epochs = None
        # self.audio_epochs = None
        self.audio_embeddings = np.array([]) # Wav2Vec2 embeddings (# epochs, # features, # samples)
        self.brain_segments = None # EEG/MEG segments (# epochs, # channels, # samples)
        
    def get_times(self): # get the times of the audio and brain data
        sound_mask = self.mne_events['kind']=='sound'
        sound_starts = self.mne_events['start'].values[sound_mask]
        sound_durations = self.mne_events['duration'].values[sound_mask]
        sound_stops = sound_starts + sound_durations
        sound_starts = sound_starts + self.time_shift # shift the EEG signal 150ms to account for latency
        sound_stops = sound_stops + self.time_shift # shift the EEG signal 150ms to account for latency
        for i in range(len(sound_starts)):
            brain_times_tmp = np.arange(sound_starts[i], sound_stops[i], self.window_duration)
            brain_times_tmp = brain_times_tmp - self.tmin # align the brain times with the audio times
            audio_times_tmp = np.arange(0, sound_durations[i], self.window_duration)
            mask_tmp = np.logical_and((brain_times_tmp + self.tmin) >= sound_starts[i], (brain_times_tmp + self.tmax) < sound_stops[i])
            brain_times_tmp = brain_times_tmp[mask_tmp]
            audio_times_tmp = audio_times_tmp[mask_tmp]
            self.brain_times = np.append(self.brain_times, brain_times_tmp)
            self.audio_times.append(audio_times_tmp)
    
    def get_brain_segments(self): # get EEG/MEG segments
        events = np.concatenate([self.brain_times[:, None], np.ones((len(self.brain_times), 2), dtype=np.int64)], 1)
        self.brain_epochs = mne.Epochs(self.data, events, tmin=self.tmin, tmax=self.tmax, baseline=(self.tmin, 0), event_repeated='drop', preload=False)
        self.brain_segments = self.brain_epochs.get_data()
    
    def get_audio_embeddings(self, audio_pth, audio_order): # load the audio file and get the audio embeddings
        # audio_pth: path to the audio file
        # audio_order: order of the audio file
        waveform, sample_rate = torchaudio.load(audio_pth)
        waveform = waveform.to(self.device)
        
        if sample_rate != self.model_srate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.model_srate)
        
        audio_times_tmp = self.audio_times[audio_order]
        for i in range(len(audio_times_tmp)):
            waveform_tmp = waveform[:, to_idx(audio_times_tmp[i], self.bundle.sample_rate): to_idx(audio_times_tmp[i] + self.window_duration, self.bundle.sample_rate)]
            with torch.inference_mode():
                features, _ = self.model.extract_features(waveform_tmp)
                
                feat_tmp = np.ndarray([])
                for i in [-4,-3,-2,-1]:
                    tmp = features[i].detach().cpu().numpy()
                    if i == -4:
                        feat_tmp = tmp
                    else:
                        feat_tmp = np.vstack((feat_tmp, tmp))
                embedding_tmp = np.transpose(np.mean(feat_tmp, axis=0))
                if self.audio_embeddings.size != 0:
                    self.audio_embeddings = np.append(self.audio_embeddings, embedding_tmp[None, :, :], axis = 0)
                else:
                    self.audio_embeddings = embedding_tmp[None, :, :]