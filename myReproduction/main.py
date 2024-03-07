import dataHandler
import net_ref_new
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from torch.utils.data import Dataset, DataLoader
import os, os.path
import numpy as np

class myDataset(Dataset):
    def __init__(self, data, ground_truth, mne_info, subject_num, device = 'cpu'):
        self.brain_data = torch.tensor(data, device = device)
        self.mne_info = mne_info
        self.subject_num = subject_num
        self.ground_truth = torch.tensor(ground_truth, device = device)
    
    def __len__(self):
        return len(self.brain_data)
    
    def __getitem__(self, idx):
        return self.brain_data[idx], self.mne_info[idx], self.subject_num[idx], self.ground_truth[idx]
    
class Residual_Dilated_Conv(nn.Module):
    def __init__(self, chin, chout):
        super(Residual_Dilated_Conv, self).__init__()
        self.conv1 = nn.Conv1d()

class myNet(nn.Module):
    def __init__(self, num_subjects):
        super(myNet, self).__init__()
        self.num_subjects = num_subjects
        self.channel_merger = net_ref_new.ChannelMerger(chout = 270, n_subjects = self.num_subjects)
        self.subject_layer = net_ref_new.SubjectLayer(chin = 270, chout = 270, n_subjects = self.num_subjects)
    
    def forward(self, brain_data):
        brain_data = self.channel_merger(brain_data)
        brain_data = self.subject_layer(brain_data)
        return brain_data

def __main__():
    # Params
    USE_CUDA = True
    dataPth = None
    num_workers = 1
    
    # Variables
    num_subject = 0 # number of subjects
    data_file_list = []
    brain_segments = torch.tensor([], device = device)
    audio_embeddings = torch.tensor([], device = device)
    mne_info = np.array([])
    subject_num = np.array([]) # The subject number of each epoch
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
    
    for data_file in data_file_list:
        data = np.load(dataPth + '/' + data_file)
        num_epoch = data['brain_segments'].shape[0]
        subject_num_tmp = np.repeat(data['subject_num'], num_epoch)
        print('Loading Subject ', subject_num_tmp[0])
        subject_num = np.append(subject_num, subject_num_tmp)
        brain_segments = torch.cat((brain_segments, torch.tensor(data['brain_segments'],device = device)), dim = 0)
        audio_embeddings = torch.cat(audio_embeddings, torch.tensor(data['audio_embeddings'], device = device), dim = 0)
        mne_info_tmp = np.repeat(data['mne_info'], num_epoch)
        mne_info = np.append(mne_info, mne_info_tmp)

    total_epochs = brain_segments.shape[0]
    print('Total number of epochs:', total_epochs)
    
    training_epochs = int(total_epochs * 0.8)
    testing_epochs = total_epochs - training_epochs
    print('Creating Training Dataset with %d epochs' % (training_epochs))
    training_dataset = myDataset(brain_segments[:training_epochs], audio_embeddings[:training_epochs], mne_info[:training_epochs], subject_num[:training_epochs], device)
    print('Creating Testing Dataset with %d epochs' % (testing_epochs))
    testing_dataset = myDataset(brain_segments[training_epochs:], audio_embeddings[training_epochs:], mne_info[training_epochs:], subject_num[training_epochs:], device)
    print('Created datasets') #**May have to apply a random split to the dataset
    
    