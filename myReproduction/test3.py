import numpy as np

with np.load('../../Data/brennanProcessed/brain/S1.npz') as data:
    print(data)
    brain_seg = data['brain_segments']
    print(brain_seg.shape)
    
with np.load('../../Data/brennanProcessed/audio/wav2vecEmb.npz') as data:
    print(data)
    audio_emb = data['audio_embeddings']
    print(audio_emb.shape)

input('Press Enter to continue...')