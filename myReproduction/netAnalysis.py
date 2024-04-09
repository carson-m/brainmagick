from evaluation import SegmentLevel_Eval
import net
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os, os.path
import numpy as np
import mne
from utils import get_file
import scipy.io as sio
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class myDataset(Dataset):
    def __init__(self, data, ground_truth, segment_label, channel_layout, subject_num):
        self.brain_data = data
        self.channel_layout = channel_layout
        self.segment_label = segment_label
        self.subject_num = subject_num
        self.ground_truth = ground_truth
    
    def __len__(self):
        return len(self.brain_data)
    
    def __getitem__(self, idx):
        return self.brain_data[idx], self.channel_layout[idx], self.subject_num[idx], self.ground_truth[idx], self.segment_label[idx]
    
class datasetGenerator:
    def __init__(self, train_set_proportion, validation_set_proportion, event_times, tmin, tmax):
        self.train_set_proportion = train_set_proportion
        self.validation_set_proportion = validation_set_proportion
        assert train_set_proportion + validation_set_proportion < 1
        self.event_times = event_times
        self.tmin = tmin
        self.tmax = tmax
    
    def get_masks(self):
        total_size = len(self.event_times)
        split_train = int(self.train_set_proportion * total_size)
        train_mask = np.empty(total_size, dtype=bool)
        train_mask[:split_train] = True
        train_mask[split_train:] = False
        
        validation_set_size = int(self.validation_set_proportion * total_size)
        validation_set_mask = np.full(total_size - split_train, False, dtype=bool)
        True_choose = np.random.choice(len(validation_set_mask), validation_set_size, replace=False)
        validation_set_mask[True_choose] = True
        
        validation_mask = ~train_mask
        validation_mask[split_train:] = validation_set_mask
        
        test_mask = ~np.logical_or(train_mask, validation_mask)
        
        # Ensure that the validation set (test set) and the training set do not have overlapping events
        critical_time = self.event_times[split_train]
        critical_time = min(critical_time - self.tmax, critical_time + self.tmin)
        conflict = train_mask & (self.event_times > critical_time)
        train_mask[conflict] = False
        
        return train_mask, validation_mask, test_mask
    
    def load_data(self, brain_file_list, audio_pth):
        train_mask, validation_mask, test_mask = self.get_masks()
        train_size_each = np.sum(train_mask)
        validation_size_each = np.sum(validation_mask)
        test_size_each = np.sum(test_mask)
        print(f"For each subject: Training {train_size_each}, Validation {validation_size_each}, Test {test_size_each}")
        
        train_brain_segments = torch.tensor([])
        train_audio_embeddings = torch.tensor([])
        train_channel_layout = torch.tensor([])
        train_subject_num = np.array([]) # The subject number of each sample
        train_segment_label = np.array([]) # The segment number of each sample
    
        validation_brain_segments = torch.tensor([])
        validation_audio_embeddings = torch.tensor([])
        validation_channel_layout = torch.tensor([])
        validation_subject_num = np.array([]) # The subject number of each sample
        validation_segment_label = np.array([]) # The segment number of each sample
        
        test_brain_segments = torch.tensor([])
        test_audio_embeddings = torch.tensor([])
        test_channel_layout = torch.tensor([])
        test_subject_num = np.array([]) # The subject number of each sample
        test_segment_label = np.array([]) # The segment number of each sample
        
        # Load the audio data from file
        audio_embeddings = np.load(audio_pth)['audio_embeddings']
        segment_label = np.arange(audio_embeddings.shape[0]).astype(int)
        len_audio = audio_embeddings.shape[0]
        audio_embeddings = torch.tensor(audio_embeddings)
        print("Loaded audio embeddings")
        
        # Load brain data while generating the training, validation and test set
        for i in range(len(brain_file_list)):
            print("Loading Subject: ", i)
            data_file = brain_file_list[i]
            data = np.load(data_file, allow_pickle=True)
            
            # brain_segments = data['brain_segments']
            brain_segments = data['arr_0']
            num_samples = brain_segments.shape[0]
            assert num_samples == len_audio, 'The number of audio embeddings and brain segments do not match'
            mne_info = data['mne_info'].item()
            layout_tmp = mne.find_layout(mne_info)
            channel_layout = torch.tensor(layout_tmp.pos[:, :2])[None]
            channel_layout = channel_layout.repeat(num_samples, 1, 1)
            subject_num = i
            
            # Get data for the test set
            test_brain_segments = torch.cat((test_brain_segments, torch.tensor(brain_segments[test_mask])), dim = 0)
            test_audio_embeddings = torch.cat((test_audio_embeddings, audio_embeddings[test_mask]), dim = 0)
            test_segment_label = np.append(test_segment_label, segment_label[:test_size_each])
            test_subject_num = np.append(test_subject_num, np.repeat(subject_num, np.sum(test_mask)))
            test_channel_layout = torch.cat((test_channel_layout, channel_layout[test_mask]), dim = 0)
            
            train_subject_num = np.append(train_subject_num, np.repeat(subject_num, np.sum(train_mask)))
            train_brain_segments = torch.cat((train_brain_segments, torch.tensor(brain_segments[train_mask])), dim = 0)
            train_audio_embeddings = torch.cat((train_audio_embeddings, audio_embeddings[train_mask]), dim = 0)
            train_segment_label = np.append(train_segment_label, segment_label[:train_size_each])
            train_channel_layout = torch.cat((train_channel_layout, channel_layout[train_mask]), dim = 0)
            
            validation_subject_num = np.append(validation_subject_num, np.repeat(subject_num, np.sum(validation_mask)))
            validation_brain_segments = torch.cat((validation_brain_segments, torch.tensor(brain_segments[validation_mask])), dim = 0)
            validation_audio_embeddings = torch.cat((validation_audio_embeddings, audio_embeddings[validation_mask]), dim = 0)
            validation_channel_layout = torch.cat((validation_channel_layout, channel_layout[validation_mask]), dim = 0)
            validation_segment_label = np.append(validation_segment_label, segment_label[:validation_size_each])

        train_subject_num = torch.tensor(train_subject_num)
        validation_subject_num = torch.tensor(validation_subject_num)
        test_subject_num = torch.tensor(test_subject_num)
        train_segment_label = torch.tensor(train_segment_label)
        validation_segment_label = torch.tensor(validation_segment_label)
        test_segment_label = torch.tensor(test_segment_label)
        audio_embeddings_out = {'train': audio_embeddings[train_mask], 'validation': audio_embeddings[validation_mask], 'test': audio_embeddings[test_mask]}
        
        dataset = {'train': myDataset(train_brain_segments, train_audio_embeddings, train_segment_label, train_channel_layout, train_subject_num),
                   'validation': myDataset(validation_brain_segments, validation_audio_embeddings, validation_segment_label, validation_channel_layout, validation_subject_num),
                   'test': myDataset(test_brain_segments, test_audio_embeddings, test_segment_label, test_channel_layout, test_subject_num)}
        
        return dataset, audio_embeddings_out

class myNet(nn.Module):
    def __init__(self, n_subjects):
        super(myNet, self).__init__()
        self.spatial_attention = net.SpatialAttention(chout=270, n_freqs=32, r_drop=0.2, margin=0.1).cuda()
        self.conv1 = nn.Conv1d(in_channels=270, out_channels=270, kernel_size=1, stride=1, padding=0, dtype=torch.double).cuda()
        self.subject_layer = net.SubjectLayers(n_channels=270, n_subjects=n_subjects).cuda()
        self.convseq = nn.ModuleList([])
        for k in range(5):
            self.convseq.append(net.ConvSequence(k, dilation_period=5, groups=1, dtype=torch.double).cuda())
        self.conv2 = nn.Conv1d(in_channels=320, out_channels=640, kernel_size=1, stride=1, padding=0, dtype=torch.double).cuda()
        self.gelu = nn.GELU()
        self.conv3 = nn.Conv1d(in_channels=640, out_channels=1024, kernel_size=1, stride=1, padding=0, dtype=torch.double).cuda()
    
    def forward(self, x, mne_info, subjects):
        x = self.spatial_attention(x, mne_info)
        x = self.conv1(x)
        x = self.subject_layer(x, subjects)
        for i in range(5):
            x = self.convseq[i](x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        return x

pth_path = './results/trail_10/mynet.pth'



mynet = torch.load(pth_path).cuda()
participant_layer = mynet.module.subject_layer
print(participant_layer.weights.data)
np.save('participant_layer', participant_layer.weights.data.detach().cpu().numpy())
# mynet.eval()
# data = np.load('../../Data/brennanProcessed/brain/S0.npz', allow_pickle=True)
# mne_info = data['mne_info'].item()
# layout_tmp = mne.find_layout(mne_info)
# channel_layout = torch.tensor(layout_tmp.pos[:, :2])[None]
# channel_layout = torch.tensor(channel_layout).cuda()
# brain_data = data['arr_0']
# brain_data = torch.tensor(brain_data[0][None]).cuda()

# # print(mynet)
# embedding = mynet.module.spatial_attention.embedding
# heads = mynet.module.spatial_attention.heads
# pos_get = mynet.module.spatial_attention.position_obtainer

# B, C, T = brain_data.shape
# positions = pos_get.get_positions(brain_data, channel_layout)
# emb = embedding(positions).to(brain_data)
# score_offset = torch.zeros(B, C, device=brain_data.device)
        
# heads = heads.to(emb)
# heads = heads[None].expand(B, -1, -1) # (B, chout, pos_dim)
# # (B, C, pos_dim) * (B, chout, pos_dim) -> (B, chout, C)
# scores = torch.einsum("bcd, bod -> boc", emb, heads)
# print(scores.shape)
# scores_save = scores.detach().cpu().numpy()
# np.save('sa_scores', scores_save)