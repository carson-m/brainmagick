import scipy.io as sio
import numpy as np
import torch
import torchaudio
# import IPython
import matplotlib.pyplot as plt
import os

print(torch.__version__)
print(torchaudio.__version__)
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
print("Sample Rate:", bundle.sample_rate)

model = bundle.get_model().to(device)

torch.save(model.state_dict(), 'wav2vec2XLSR53.pth')

print(model.__class__)

# load wav file
dir = '../../Data/Brennan/audio/DownTheRabbitHoleFinal_SoundFile'
# sfreq = 0
# audio = np.ndarray([])
# # for i in range(1, 13, 1):
# #     dir_tmp = dir + str(i) + '.wav'
# #     wav = sio.wavfile.read(dir_tmp)
# #     sfreq = wav[0]
# #     audio = np.append(audio, wav[1])

i = 1
dir_tmp = dir + str(i) + '.wav'
# wav = sio.wavfile.read(dir_tmp)
# sfreq = wav[0]
# audio = np.append(audio, wav[1])
waveform, sample_rate = torchaudio.load(dir_tmp)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

with torch.inference_mode():
    features, _ = model.extract_features(waveform)

# create a vstack of the features to calculate the mean
feat_tmp = np.ndarray([])
for i in [-4,-3,-2,-1]:
    tmp = features[i].detach().cpu().numpy()
    if i == -4:
        feat_tmp = tmp
    else:
        feat_tmp = np.vstack((feat_tmp, tmp))
    
embedding = np.mean(feat_tmp, axis=0)
imshow = plt.imshow(embedding, aspect='auto', origin='lower', interpolation='nearest')
plt.show()

# exit
input("Press Enter to continue...")