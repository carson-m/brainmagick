import dataHandler
import net
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os, os.path
import numpy as np
import multiprocessing
import mne

class myDataset(Dataset):
    def __init__(self, data, ground_truth, mne_info, subject_num):
        self.brain_data = data
        self.mne_info = mne_info
        self.subject_num = subject_num
        self.ground_truth = ground_truth
    
    def __len__(self):
        return len(self.brain_data)
    
    def __getitem__(self, idx):
        channel_layout = mne.channels.find_layout(self.mne_info[idx]).pos[:, :2]
        return self.brain_data[idx], channel_layout, self.subject_num[idx], self.ground_truth[idx]

class myNet(nn.Module):
    def __init__(self, n_subjects):
        super(myNet, self).__init__()
        self.spatial_attention = net.SpatialAttention(chout=270, n_freqs=32, r_drop=0.2, margin=0.1)
        self.subject_layer = net.SubjectLayers(n_channels=270, n_subjects=n_subjects)
        self.conv1 = nn.Conv1d(in_channels=270, out_channels=270, kernel_size=1, stride=1, padding=0, dtype=torch.double)
        self.convseq = list([])
        for k in range(5):
            self.convseq.append(net.ConvSequence(k, dilation_period=5, groups=1, dtype=torch.double))
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

def train_net(device, train_loader, test_loader, train_set_size, test_set_size, n_subjects, lr, num_epochs):
    mynet = myNet(n_subjects)
    mynet.to(device).double()
    criterion = net.ClipLoss().to(device).double()
    optimizer = optim.Adam(mynet.parameters(), lr=lr)
    train_loss_array = []
    test_loss_array = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        test_loss = 0.0
        mynet.train() # Set the model to training mode
        for data, channel_layout, subjects, ground_truth in train_loader:
            data = data.to(device)
            ground_truth = ground_truth.to(device)
            channel_layout = channel_layout.to(device)
            subjects = subjects.to(device).long()
            optimizer.zero_grad()
            output = mynet(data, channel_layout, subjects)
            loss = criterion(output, ground_truth)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.shape[0]
        
        mynet.eval() # Set the model to evaluation mode
        for data, channel_layout, subjects, ground_truth in test_loader:
            data = data.to(device)
            ground_truth = ground_truth.to(device)
            channel_layout = channel_layout.to(device)
            subjects = subjects.to(device).long()
            output = mynet(data, channel_layout, subjects)
            loss = criterion(output, ground_truth)
            test_loss += loss.item() * data.shape[0]
        
        train_loss = train_loss / train_set_size
        test_loss = test_loss / test_set_size
        train_loss_array.append(train_loss)
        test_loss_array.append(test_loss)
        print('Epoch:', epoch, 'Train Loss:', train_loss, 'Test Loss:', test_loss)
    return mynet, train_loss_array, test_loss_array

def __main__():
    # Params
    USE_CUDA = True
    dataPth = './'
    num_workers = 1
    lr = 3e-4
    num_epochs = 40
    
    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    maximum_thread = multiprocessing.cpu_count()
    print('Available threads:', maximum_thread)
    print('Num workers:', num_workers)
    
    # Variables
    num_subject = 0 # number of subjects
    data_file_list = []
    
    train_brain_segments = torch.tensor([], device = device)
    train_audio_embeddings = torch.tensor([], device = device)
    train_channel_layout = np.array([])
    train_subject_num = np.array([]) # The subject number of each sample
    
    test_brain_segments = torch.tensor([], device = device)
    test_audio_embeddings = torch.tensor([], device = device)
    test_mne_info = np.array([])
    test_subject_num = np.array([]) # The subject number of each sample
    
    total_samples = 0 # number of samples
    
    
    # Load the data from file
    for __,__,filenames in os.walk(dataPth): 
    # print(root)
    # print(dirname)
    # print(filenames)
        for filename in filenames:  
            if os.path.splitext(filename)[-1]=='.npz':
                num_subject += 1
                data_file_list.append(filename)
    
    train_set_proportion = 0.8
    for data_file in data_file_list:
        data = np.load(dataPth + '/' + data_file, allow_pickle=True)
        num_samples = data['brain_segments'].shape[0]
        train_epoch = int(num_samples * train_set_proportion)
        subject_num_tmp = np.repeat(data['subject_num'], num_samples)
        print('Loading Subject ', subject_num_tmp[0])
        train_subject_num = np.append(train_subject_num, subject_num_tmp[:train_epoch])
        brain_segments = data['brain_segments']
        audio_embeddings = data['audio_embeddings']
        train_brain_segments = torch.cat((train_brain_segments, torch.tensor(brain_segments[:train_epoch],device = device)), dim = 0)
        train_audio_embeddings = torch.cat((train_audio_embeddings, torch.tensor(audio_embeddings[:train_epoch], device = device)), dim = 0)
        mne_info_tmp = np.repeat(data['mne_info'], num_samples)
        train_channel_layout = np.append(train_channel_layout, mne_info_tmp[:train_epoch])
        
        test_subject_num = np.append(test_subject_num, subject_num_tmp[train_epoch:])
        test_brain_segments = torch.cat((test_brain_segments, torch.tensor(brain_segments[train_epoch:],device = device)), dim = 0)
        test_audio_embeddings = torch.cat((test_audio_embeddings, torch.tensor(audio_embeddings[train_epoch:], device = device)), dim = 0)
        test_mne_info = np.append(test_mne_info, mne_info_tmp[train_epoch:])

        total_samples += num_samples
    print('Total number of subjects:', num_subject)
    print('Total number of samples:', total_samples)
    print('Num of training samples:', train_brain_segments.shape[0])
    print('Num of testing samples:', test_brain_segments.shape[0])
    print('Creating Training Dataset')
    train_dataset = myDataset(train_brain_segments, train_audio_embeddings, train_channel_layout, train_subject_num)
    print('Creating Testing Dataset')
    test_dataset = myDataset(test_brain_segments, test_audio_embeddings, test_mne_info, test_subject_num)
    print('Creating Training DataLoader')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    mynet, train_loss_array, test_loss_array = train_net(device, train_loader, test_loader, train_brain_segments.shape[0], test_brain_segments.shape[0], num_subject, lr=lr, num_epochs=num_epochs)
    
    # Save the model
    torch.save(mynet.state_dict(), 'mynet.pth')
    print('Model saved')
    # Save the loss arrays
    np.save('loss.npy', train_loss_array, test_loss_array)
    print('Loss saved')
    
if __name__ == '__main__':
    __main__()