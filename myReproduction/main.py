import dataHandler
import net
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from torch.utils.data import Dataset, DataLoader
import os, os.path
import numpy as np

class myDataset(Dataset):
    def __init__(self, data, ground_truth, mne_info, subject_num):
        self.brain_data = data
        self.mne_info = mne_info
        self.subject_num = subject_num
        self.ground_truth = ground_truth
    
    def __len__(self):
        return len(self.brain_data)
    
    def __getitem__(self, idx):
        return self.brain_data[idx], self.mne_info[idx], self.subject_num[idx], self.ground_truth[idx]

class myNet(nn.Module):
    def __init__(self, n_subjects):
        super(myNet, self).__init__()
        self.spatial_attention = net.SpatialAttention(chout=270, n_freqs=32, r_drop=0.2, margin=0.1)
        self.subject_layer = net.SubjectLayers(n_channels=270, n_subjects=n_subjects)
        self.conv1 = nn.Conv1d(in_channels=270, out_channels=270, kernel_size=1, stride=1, padding=0)
        self.convseq = list([])
        for k in range(5):
            self.convseq.append(net.ConvSequence(k, dilation_period=5, groups=1))
        self.conv2 = nn.Conv1d(in_channels=320, out_channels=640, kernel_size=1, stride=1, padding=0)
        self.gelu = nn.GELU()
        self.conv3 = nn.Conv1d(in_channels=640, out_channels=1024, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x, mne_info, subjects):
        x = self.spatial_attention(x, mne_info)
        x = self.subject_layer(x, subjects)
        x = self.conv1(x)
        for i in range(5):
            x = self.convseq[i](x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        return x

def __main__():
    # Params
    USE_CUDA = True
    dataPth = None
    num_workers = 1
    
    # Variables
    num_subject = 0 # number of subjects
    data_file_list = []
    
    train_brain_segments = torch.tensor([], device = device)
    train_audio_embeddings = torch.tensor([], device = device)
    train_mne_info = np.array([])
    train_subject_num = np.array([]) # The subject number of each epoch
    
    test_brain_segments = torch.tensor([], device = device)
    test_audio_embeddings = torch.tensor([], device = device)
    test_mne_info = np.array([])
    test_subject_num = np.array([]) # The subject number of each epoch
    
    total_epochs = 0 # number of epochs
    
    
    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    maximum_thread = multiprocessing.cpu_count()
    print('Available threads:', maximum_thread)
    print('Num workers:', num_workers)
    
    
    # Load the data from file
    for __,__,filenames in os.walk(dataPth): 
    # print(root)
    # print(dirname)
    # print(filenames)
        for filename in filenames:  
            print(filename)
            if os.path.splitext(filename)[-1]=='.npz':
                num_subject += 1
                data_file_list.append(filename)
    
    train_set_proportion = 0.8
    for data_file in data_file_list:
        data = np.load(dataPth + '/' + data_file)
        num_epoch = data['brain_segments'].shape[0]
        train_epoch = int(num_epoch * train_set_proportion)
        subject_num_tmp = np.repeat(data['subject_num'], num_epoch)
        print('Loading Subject ', subject_num_tmp[0])
        train_subject_num = np.append(train_subject_num, subject_num_tmp[:train_epoch])
        train_brain_segments = torch.cat((train_brain_segments, torch.tensor(data['brain_segments'][:train_epoch],device = device)), dim = 0)
        train_audio_embeddings = torch.cat(train_audio_embeddings, torch.tensor(data['audio_embeddings'][:train_epoch], device = device), dim = 0)
        mne_info_tmp = np.repeat(data['mne_info'], num_epoch)
        train_mne_info = np.append(train_mne_info, mne_info_tmp[:train_epoch])
        
        test_subject_num = np.append(test_subject_num, subject_num_tmp[train_epoch:])
        test_brain_segments = torch.cat((test_brain_segments, torch.tensor(data['brain_segments'][train_epoch:],device = device)), dim = 0)
        test_audio_embeddings = torch.cat(test_audio_embeddings, torch.tensor(data['audio_embeddings'][train_epoch:], device = device), dim = 0)
        test_mne_info = np.append(test_mne_info, mne_info_tmp[train_epoch:])

        total_epochs += num_epoch
    print('Total number of subjects:', num_subject)
    print('Total number of epochs:', total_epochs)
    print('Num of training epochs:', train_brain_segments.shape[0])
    print('Num of testing epochs:', test_brain_segments.shape[0])
    print('Creating Training Dataset')
    train_dataset = myDataset(train_brain_segments, train_audio_embeddings, train_mne_info, train_subject_num)
    print('Creating Testing Dataset')
    test_dataset = myDataset(test_brain_segments, test_audio_embeddings, test_mne_info, test_subject_num)
    print('Creating Training DataLoader')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)