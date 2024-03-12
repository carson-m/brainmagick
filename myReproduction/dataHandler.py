import numpy as np
import mne as mne
import pandas as pd
import torchaudio
import torch
from torch.nn import functional as F
from utils import to_idx, get_file
import os.path

def get_srate(raw_pth):
    data = mne.io.read_raw_fif(raw_pth)
    return data.info['sfreq']
    
class brainHandler:
    def __init__(self, raw_pth, events_pth, tmin, tmax, time_shift, window_duration, mask):
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
        self.brain_len = int(window_duration * self.brain_srate)
    
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
        if self.brain_segments.shape[-1] > self.brain_len:
            self.brain_segments = self.brain_segments[..., :self.brain_len]
    
    def return_data(self):
        return self.brain_segments
    
    def save_data(self, pth):
        np.savez(pth, brain_segments=self.brain_segments, mne_info=self.data.info)
        

class audioHandler:
    def __init__(self, bundle, device, tmin, tmax, time_shift, window_duration, brain_srate):
        self.model = bundle.get_model().to(device)
        self.model_srate = bundle.sample_rate
        self.device = device
        self.window_duration = window_duration
        self.tmax = tmax
        self.tmin = tmin
        self.time_shift = time_shift
        #self.audio_num = audio_num # number of audio files
        self.audio_embeddings = np.array([]) # Wav2Vec2 embeddings (# epochs, # features, # samples)
        self.mask = list([])
        self.brain_len = int(brain_srate * self.window_duration)
    
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
            embedding_tmp = F.interpolate(torch.tensor(embedding_tmp[None]), size = self.brain_len).detach().cpu().numpy()[0]
            if self.audio_embeddings.size != 0:
                self.audio_embeddings = np.append(self.audio_embeddings, embedding_tmp[None, :, :], axis = 0)
            else:
                self.audio_embeddings = embedding_tmp[None, :, :]
    
    def return_data(self):
        return self.audio_embeddings
    
    def return_mask(self):
        return self.mask
        
def dataFactory(audio_num):
    # Example usage
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(torch.__version__)
    print(torchaudio.__version__)
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
    print("Sample Rate:", bundle.sample_rate)

    brain_pth = '../cache/studies/brennan2019/'
    audio_pth = '../data/brennan2019/download/audio/DownTheRabbitHoleFinal_SoundFile'
    tmin = -0.5
    tmax = 2.5
    time_shift = 0.15
    window_duration = 3
    save_pth = '../../Data/brennanProcessed/'
    
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
        os.makedirs(save_pth + 'brain/')
        os.makedirs(save_pth + 'audio/')
    
    brain_dir = os.listdir(brain_pth)
    brain_dir = [d for d in brain_dir if os.path.isdir(brain_pth + d)]
    
    # Get the brain data sample rate
    raw_pth_tmp = brain_pth + brain_dir[0] + '/'
    raw_pth_tmp = raw_pth_tmp + get_file(raw_pth_tmp, '.fif')
    
    brain_srate = get_srate(raw_pth_tmp)
    print('Brain Data Sample Rate:', brain_srate)
    
    # Get Audio Embeddings
    print('Getting Audio Embeddings')
    ah = audioHandler(bundle=bundle, device=device, tmin=tmin, tmax=tmax, time_shift=time_shift, window_duration=window_duration, brain_srate=brain_srate)
    for i in range(audio_num):
        audio_pth_tmp = audio_pth + str(i + 1) + '.wav'
        ah.get_audio_embeddings(audio_pth_tmp)
    mask = ah.return_mask()
    audio_embeddings = ah.return_data()
    print('Got Audio Embeddings')
    np.savez(save_pth + '/audio/wav2vecEmb', audio_embeddings=audio_embeddings)
    
    subject_num = 1
    # Get Brain Data
    print('Getting Brain Data')
    for sub_folder in brain_dir:
        raw_pth = brain_pth + sub_folder + '/'
        raw_pth = raw_pth + get_file(raw_pth, '.fif')
        events_pth = brain_pth + sub_folder + '/'
        events_pth = events_pth + get_file(events_pth, '.csv')
        subject_num = int(sub_folder[1:])
        print('Loading Subject', subject_num)
        bh = brainHandler(raw_pth, events_pth, tmin, tmax, time_shift, window_duration, mask)
        bh.get_times()
        bh.get_brain_segments()
        print('Saving Data for Subject ', subject_num)
        bh.save_data(save_pth + '/brain/S' + str(subject_num))
    print('Got Brain Data')

if __name__ == '__main__':
    dataFactory(12)