import numpy as np

with np.load('S01.npz') as data:
    brain_seg = data['brain_segments']
    audio_emb = data['audio_embeddings']

input('Press Enter to continue...')