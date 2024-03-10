import numpy as np
import mne as mne
import pandas as pd
import torchaudio
import torch
from torch.nn import functional as F
from utils import to_idx

class brainHandler:
    def __init__(self, subject_num, raw_pth, events_pth, tmin, tmax, time_shift, window_duration, mask):
        self.subject_num = subject_num
        self.data = mne.io.read_raw_fif(raw_pth)
        self.mne_events = pd.read_csv(events_pth)
        self.brain_srate = self.data.info['sfreq']
        self.tmin = tmin # s
        self.tmax = tmax # s
        self.time_shift = time_shift # s
        self.window_duration = window_duration
        self.brain_times = np.array([])
        self.brain_epochs = None
        self.brain_segments = None
        self.mask = mask
    
    def get_mne_info(self):
        return self.data.info
    
    def get_brain_len(self):
        return self.brain_segments.shape[-1]
    
    def get_times(self): # get the times of the brain data
        sound_mask = self.mne_events['kind']=='sound'
        self.audio_num = np.sum(sound_mask)
        sound_starts = self.mne_events['start'].values[sound_mask]
        sound_durations = self.mne_events['duration'].values[sound_mask]
        sound_stops = sound_starts + sound_durations
        sound_starts = sound_starts + self.time_shift # shift the EEG signal 150ms to account for latency
        sound_stops = sound_stops + self.time_shift # shift the EEG signal 150ms to account for latency
        for i in range(len(sound_starts)):
            brain_times_tmp = np.arange(sound_starts[i], sound_stops[i], self.window_duration)
            brain_times_tmp = brain_times_tmp - self.tmin # align the brain times with the audio times
            mask_tmp = self.mask[i]
            brain_times_tmp = brain_times_tmp[mask_tmp]
            self.brain_times = np.append(self.brain_times, brain_times_tmp)
            
    def get_brain_segments(self): # get EEG/MEG segments
        events = np.concatenate([to_idx(self.brain_times, self.brain_srate)[:, None], np.ones((len(self.brain_times), 2), dtype=np.int64)], 1)
        self.brain_epochs = mne.Epochs(self.data, events, tmin=self.tmin, tmax=self.tmax, baseline=(self.tmin, 0), event_repeated='drop', preload=False)
        self.brain_segments = self.brain_epochs.get_data()
    
    def return_data(self):
        return self.brain_segments
        

class audioHandler:
    def __init__(self, audio_num, bundle, device, tmin, tmax, time_shift, window_duration):
        self.model = bundle.get_model().to(device)
        self.model_srate = bundle.sample_rate
        self.device = device
        self.window_duration = window_duration
        self.tmax = tmax
        self.tmin = tmin
        self.time_shift = time_shift
        self.audio_num = audio_num # number of audio files
        self.audio_embeddings = np.array([]) # Wav2Vec2 embeddings (# epochs, # features, # samples)
        self.mask = list([])
    
    def get_audio_embeddings(self, audio_pth): # load the audio file and get the audio embeddings
        # audio_pth: path to the audio file
        waveform, sample_rate = torchaudio.load(audio_pth)
        audio_duration = waveform.shape[-1] / sample_rate
        audio_times = np.arange(0, audio_duration, self.window_duration)
        mask_tmp = np.logical_and((audio_times + self.time_shift) >= 0, (audio_times + self.time_shift) < audio_duration)
        audio_times = audio_times[mask_tmp] # make sure the audio times have corresponding EEG/MEG data
        self.mask.append(mask_tmp) # save the mask for brain data alignment
        waveform = waveform.to(self.device)
        features = None
        wav2vec_emb_sr = 0
        
        if sample_rate != self.model_srate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.model_srate)
        
        with torch.inference_mode():
                features, _ = self.model.extract_features(waveform) # Extract features of the whole recording
        wav2vec_emb_sr = features[0].shape[-2] / (waveform.shape[-1]/self.model_srate) # Get the sample rate of the embeddings
        
        feat_tmp = np.ndarray([])
        for i in [-4,-3,-2,-1]:
            tmp = features[i].detach().cpu().numpy()
            if i == -4:
                feat_tmp = tmp
            else:
                feat_tmp = np.vstack((feat_tmp, tmp))
        embedding_avg = np.transpose(np.mean(feat_tmp, axis=0))
        
        for i in range(len(audio_times)):
            embedding_tmp = embedding_avg[:, to_idx(audio_times[i], wav2vec_emb_sr): to_idx(audio_times[i] + self.window_duration, wav2vec_emb_sr)]
            # print(wav2vec_emb_sr)
            # print(embedding_tmp.shape)
            embedding_tmp = F.interpolate(torch.tensor(embedding_tmp[None]), size = self.get_brain_len()).detach().cpu().numpy()[0]
            if self.audio_embeddings.size != 0:
                self.audio_embeddings = np.append(self.audio_embeddings, embedding_tmp[None, :, :], axis = 0)
            else:
                self.audio_embeddings = embedding_tmp[None, :, :]
    
    def return_data(self):
        return self.audio_embeddings
    
    def return_mask(self):
        return self.mask
        
def dataFactory():
    # Example usage
    print(torch.__version__)
    print(torchaudio.__version__)
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
    print("Sample Rate:", bundle.sample_rate)

    raw_pth = '../../Data/brennan2019/S01/meg-sr120-hp0-raw.fif'
    events_pth = '../../Data/brennan2019/S01/events.csv'
    audio_pth = '../../Data/Brennan/audio/DownTheRabbitHoleFinal_SoundFile'
    tmin = -0.5
    tmax = 2.5
    time_shift = 0.15
    window_duration = 3
    pth = 'S01.npy'
    
    # dh = audioHandler(0, bundle, device, raw_pth, events_pth, tmin, tmax, time_shift, window_duration)
    # dh.get_times()
    # dh.get_brain_segments()
    # for i in range(dh.get_audio_num()):
    #     audio_pth_tmp = audio_pth + str(i + 1) + '.wav'
    #     dh.get_audio_embeddings(audio_pth_tmp, i)
    # dh.save_data(pth)

if __name__ == '__main__':
    dataFactory()