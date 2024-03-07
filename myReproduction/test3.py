import numpy as np

with np.load('S01.npy.npz') as data:
    print(data)
    brain_seg = data['brain_segments']
    audio_emb = data['audio_embeddings']
    print(brain_seg.shape)
    print(audio_emb.shape)

input('Press Enter to continue...')