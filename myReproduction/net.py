import numpy as np
import math
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionObtainer:
    '''
    INVALID: a value to represent invalid position
    '''
    INVALID = -0.1
    
    def __init__(self):
        pass
    
    def get_channel_layout(self, mne_info: mne.Info):
        # Get the channel layout
        channel_layout = mne.channels.find_layout(mne_info)
        x, y = channel_layout.pos[:, :2].T
        x = (x - x.min()) / (x.max() - x.min()) # Let x be in the range [0, 1]
        y = (y - y.min()) / (y.max() - y.min()) # Let y be in the range [0, 1]
        positions = np.array([x, y]) # (C, 2)
        return positions
    
    def get_positions(self, brain_data, batch): # Get the channel positions of the batch
        B, C, T = brain_data.shape
        positions = torch.full((B, C, 2), self.INVALID, device=brain_data.device) # init with INVALID marker
        for idx in range(len(batch)):
            mne_info_tmp = batch.mne_info[idx]
            rec_pos = self.get_channel_layout(mne_info_tmp)
            positions[idx, :len(rec_pos)] = rec_pos.to(brain_data.device)
        return positions # 归一化的channel位置 (B, C, 2)
    
    def is_invalid(self, positions):
        return (positions == self.INVALID).all(dim=-1)
    
class FourierEmb(nn.Module):
    def __init__(self, n_freqs: int = 32, margin: float = 0.1):
        super().__init__()
        self.n_freqs = n_freqs
        self.margin = margin
        self.width = 1 + 2 * margin
    
    def forward(self, positions):
        *O, D = positions.shape
        positions = (positions + self.margin) / self.width
        freqs_y = torch.arange(self.n_freqs, device = positions.device)
        freqs_x = freqs_y[:, None]
        p_x = 2 * math.pi * freqs_x
        p_y = 2 * math.pi * freqs_y
        positions = positions[..., None, None, :] # (B, C, 1, 1, D) # here 'C' denotes number of physical channels
        phase = (positions[..., 0]*p_x + positions[..., 1]*p_y).view(*O, -1) # (B, C, n_freqs**2)
        emb = torch.cat([phase.sin(), phase.cos()], dim = -1) # (B, C, pos_dim = n_freqs**2 * 2)
        return emb

class ChannelDropout(nn.Module):
    def __init__(self, r_drop: float = 0.2):
        """
        Args:
            r_drop (float, optional): dropout radius in normalized [0, 1] coordinates. Defaults to 0.2.
        """
        super().__init__()
        self.r_drop = r_drop
        self.position_obtainer = PositionObtainer()
        
    def forward(self, brain_dat, batch):
        if not self.r_drop:
            return brain_dat
        
        B, C, T = brain_dat.shape
        positions = self.position_obtainer.get_positions(brain_dat, batch)
        
        if self.training:
            drop_center = torch.rand(2, device=brain_dat.device) # center of dropout (x, y)
            mask = (positions - drop_center).norm(dim=-1) > self.r_drop # (B, C)
            brain_dat = brain_dat * mask.float()[:, :, None]
        
        return brain_dat

class SpatialAttention(nn.Module):
    def __init__(self, chout, n_freqs, r_drop = 0.2, margin = 0.1, usage_penalty: float = 0.):
        '''
        inputs:
            chout: number of output channels
            n_freqs: number of frequencies in Fourier Emb
            r_drop: dropout radius in normalized [0, 1] coordinates
        variables:
            pos_dim: number of input channels of heads (=2*(n_freqs**2))
            heads: learnable parameter 'z' in the paper
            drop_center: center of dropout (x, y)
            mask: mask for dropout, dropped channels are set to True
            score_offset: offset to the attention score, remove dropped channels from softmax, see paper
        '''
        super().__init__()
        self.pos_dim = 2 * (n_freqs ** 2)
        self.position_obtainer = PositionObtainer()
        self.heads = nn.Parameter(torch.randn(chout, self.pos_dim, requires_grad = True))
        self.heads.data /= self.pos_dim ** 0.5 # Why?
        self.r_drop = r_drop
        self.embedding = FourierEmb(n_freqs, margin = margin)
        self.usage_penalty = usage_penalty
        self._penalty = torch.tensor(0.)
        
    @property
    def training_penalty(self): # what is this?
        return self._penalty.to(next(self.parameters()).device)
        
    def forward(self, brain_data, batch):
        B, C, T = brain_data.shape
        positions = self.position_obtainer.get_positions(brain_data, batch)
        emb = self.embedding(positions)
        score_offset = torch.zeros(B, C, device=brain_data.device)
        if self.training and self.r_drop:
            drop_center = torch.rand(2, device=brain_data.device)
            mask = (positions - drop_center).norm(dim=-1) <= self.r_drop
            score_offset[mask] = float('-inf')
        
        heads = self.heads[None].expand(B, -1, -1) # (B, chout, pos_dim)
        # (B, C, pos_dim) * (B, chout, pos_dim) -> (B, C, chout)
        scores = torch.einsum("bcd, bod -> bco", emb, heads)
        scores += score_offset[:, :, None]
        weights = scores.softmax(dim = -1)
        # Merge Channel (B, C, T) * (B, C, chout) -> (B, chout, T)
        out = torch.einsum("bct, bco -> bot", brain_data, weights)
        if self.training and self.usage_penalty > 0.: # what is this?
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage
        return out

class SubjectLayers(nn.Module):
    def __init__(self, n_channels, n_subjects, init_id: bool = True):
        # weights: (n_subjects, n_channels(chin), n_channels(chout))
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, n_channels, n_channels), requires_grad = True)
        if init_id:
            self.weights.data[:] = torch.eye(n_channels)[None]
        self.weights.data /= n_channels ** 0.5
    
    def forward(self, x, subjects):
        '''
        inputs:
            x: (B, C, T)
            subjects: (B,)
        '''
        __, D, __ = self.weights.shape
        weights_tmp = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, D, D)) # (B, D, D)
        return torch.einsum("bct, bcd -> bdt", x, weights_tmp)
    
    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"

class ConvSequence(nn.Module):
    def __init__(self, k: int, dilation_period: int, groups: int = 1):
        super().init()
        in_channels = 320
        if k == 0: # if first block, input channels are 270
            in_channels = 270
        dilation1 = 2 ** ((2 * k) % dilation_period)
        padding1 = dilation1
        dilation2 = 2 ** ((2 * (k + 1)) % dilation_period)
        padding2 = dilation2
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=320, kernel_size=3, stride=1, padding=padding1\
            , dilation=dilation1, groups=groups)
        self.bn1 = nn.BatchNorm1d(320)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=padding2\
            , dilation=dilation2, groups=groups)
        self.bn2 = nn.BatchNorm1d(320)
        self.gelu2 = nn.GELU()
        self.conv3 = nn.Conv1d(in_channels=320, out_channels=640, kernel_size=3, stride=1, padding=1, groups=groups)
        self.glu = nn.GLU(dim=1) # output (B, 320, T)
    
    def forward(self, x):
        if self.k != 0:
            x = self.gelu1(self.bn1(self.conv1(x))) + x # Residual connection
        else:
            x = self.gelu1(self.bn1(self.conv1(x)))
        
        x = self.gelu2(self.bn2(self.conv2(x))) + x
        x = self.conv3(x)
        x = self.glu(x)
        return x

class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def get_scores(self, estimates, candidates):
        # (B, C, T) * (B', C, T) -> (B, B')
        scores = torch.einsum("bct, oct->bo", estimates, candidates)
        return scores
    
    def forward(self, estimate, candidate):
        scores = self.get_scores(estimate, candidate)
        target = torch.arange(len(scores), device=estimate.device)
        return F.cross_entropy(scores, target)