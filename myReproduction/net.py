import mne
import torch
from torch import nn
import numpy as np

class PositionObtainer: #**To be fixed for Tensor operations
    # Get the channel positions of the recording
    def __init__(self):
        pass
    
    def get_channel_layout(self, mne_info: mne.Info):
        layout = mne.channels.find_layout(mne_info)
        x, y = layout.pos[:, :2].T
        x = (x - x.min()) / (x.max() - x.min()) # Let x be in the range [0, 1]
        y = (y - y.min()) / (y.max() - y.min()) # Let y be in the range [0, 1]
        positions = np.array([x, y])
        return positions
    
    def get_positions(self, batch): # Get the channel positions of the batch
        brain_data = batch.brain_data
        B, C, T = brain_data.shape
        positions = torch.full((B, C, 2), device=brain_data.device)
        for idx in range(len(batch)):
            mne_info_tmp = batch.mne_info[idx]
            rec_pos = self.get_channel_layout(mne_info_tmp)
            positions[idx, :len(rec_pos)] = rec_pos.to(brain_data.device)
        return positions

class FourierEmbedding():
    # Get the fourier positional embedding
        

def __main__():
    pass

if __name__ == "__main__":
    __main__()