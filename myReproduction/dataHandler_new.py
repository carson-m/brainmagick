import numpy as np
from utils import to_idx
import norm
import mne
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from torch.nn import functional as F

class event_getter:
    def __init__(self, tmin, tmax, tshift, events):
        self.tmin = tmin # Time before word onset
        self.tmax = tmax # Time after word onset
        self.tshift = tshift # Time shift to account for latency
        self.events = events # All Events recorded in CSV File
        self.word_mask = self.events['kind'] == type # All the rows that have type: 'word' are marked with True, others False
        self.all_words = np.empty(shape=sum(self.word_mask), dtype=bool) # A list that has length of the number of words(len of word_mask)
    
    def get_event_mask(self, type):
        return self.events['kind'] == type
    
    def get_sound_time(self):
        mask = self.get_event_mask('sound')
        sound_start = self.events['start'].values[mask]
        sound_duration = self.events['duration'].values[mask]
        sound_end = sound_start + sound_duration
        return sound_start, sound_end
    
    def get_sound_duration(self):
        mask = self.get_event_mask('sound')
        return self.events['duration'].values[mask]
    
    def choose_words(self):
        self.word_times = self.events['start'].values[self.word_mask]
        self.sound_starts, self.sound_ends = self.get_sound_time()
        for start, end in zip(self.sound_starts, self.sound_ends):
            mask = np.logical_and(self.word_times >= start, self.word_times <= end)
            edge_mask = np.logical_and(self.word_times[mask] + self.tmin >= start + self.tshift, self.word_times[mask] + self.tmax <= end + self.tshift) # Check if the word times are within the sound times
            self.all_words[mask] = edge_mask
    
    def get_word_time(self):
        return self.word_times[self.all_words]
    
    def get_audio_time(self, word_times):
        # create a list for each audio file, the number of audio files is the same as the number of sound events
        audio_times = list([])
        
        for start, end in zip(self.sound_starts, self.sound_ends):
            mask = np.logical_and(self.word_times >= start, self.word_times <= end)
            audio_times.append(word_times[mask] + (self.tmin - self.tshift - start)) # Audio times for each audio file, at the start of every segment
        return audio_times
    
    def get_times(self): # get the times of the brain data and audio data
        self.choose_words()
        word_times = self.get_word_time()
        audio_times = self.get_audio_time(word_times)
        return word_times, audio_times
    
    def get_word_mask(self): # For Debug Use, check if all the masks are identical
        return self.all_words
    
class brainHandler:
    def __init__(self, data, tmin, tmax, device):
        self.tmin = tmin
        self.tmax = tmax # window duration = tmax - tmin
        self.data = data
        self.srate = data.info['sfreq']
        self.device = device
        self.brain_len = int((tmax-tmin) * self.srate)
    
    def get_mne_info(self):
        return self.data.info
    
    def get_srate(self):
        return self.srate
    
    def normalize_brain_data(self, brain_segments):
        brain_seg_tmp = torch.tensor(brain_segments).to(self.device)
        sca = norm.RobustScaler(lowq=0.25, highq = 0.75, subsample=1., device=self.device)
        for i in range(brain_seg_tmp.shape[0]):
            brain_seg_tmp[i] = sca.transform(brain_seg_tmp[i])
        brain_segments = brain_seg_tmp.detach().cpu().numpy()
        return brain_segments
    
    def get_brain_segments(self, times, do_normalization = True):
        """_summary_

        Args:
            times: Time array of events
            do_normalization (bool, optional): Do Robust Scalar Normalization. Defaults to True.
        """
        events = np.concatenate([to_idx(times, self.srate)[:, None], np.ones((len(times), 2), dtype=np.int64)], 1)
        brain_epochs = mne.Epochs(self.data, events, tmin=self.tmin, tmax=self.tmax, baseline=(self.tmin, 0), event_repeated='drop', preload=False)
        brain_segments = brain_epochs.get_data()
        if brain_segments.shape[-1] > self.brain_len:
            brain_segments = brain_segments[..., :self.brain_len]
        if do_normalization:
            return self.normalize_brain_data(brain_segments)
        return brain_segments
    
    def save_brain_segments(self, pth, times, do_normalization = True):
        """_summary_

        Args:
            pth (_type_): Save path
            times: Time array of events
            do_normalization (bool, optional): Do Robust Scalar Normalization. Defaults to True.
        """
        brain_segments = self.get_brain_segments(times, do_normalization)
        np.savez(pth, brain_segments, mne_info = self.get_mne_info())
        
class audioHandler:
    def __init__(self, window_duration, target_len, tshift, device):
        self.window_duration = window_duration
        self.target_len = target_len
        self.tshift = tshift
        self.device = device
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(device)
        self.model.eval()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.model_srate = self.feature_extractor.sampling_rate
    
    def get_sound_embedding(self, wave, orig_srate, duration):
        wave = wave.mean(dim = 0) # Stereo to Mono
        wave = wave.to(self.device)
        if orig_srate != self.model_srate:
            wave = torchaudio.functional.resample(wave, orig_srate, self.model_srate)
        
        input_vals = self.feature_extractor(wave, sampling_rate = self.model_srate, return_tensors = "pt")
        with torch.no_grad():
            model_output = self.model(input_vals.to(self.device), output_hidden_states = True)
            hidden_states = model_output.get('hidden_states')
        
        wav2vec_emb_sr = hidden_states[0].shape[-2] / duration # Get the sample rate of the embeddings
        
        hidden_states = torch.stack(hidden_states)
        hidden_states = hidden_states[-4:].mean(dim=0).squeeze()
        embedding_avg = hidden_states.detach().cpu().numpy()
        return embedding_avg, wav2vec_emb_sr
    
    def get_segment_embeddings(self):