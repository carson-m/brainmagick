# import dataHandler
from evaluation import SegmentLevel_Eval
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
from utils import get_file

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

def train_net(device, train_loader, test_loader, audio_embeddings, train_set_size, test_set_size, n_subjects, lr, num_epochs):
    mynet = myNet(n_subjects)
    mynet.to(device).double()
    criterion = net.ClipLoss().to(device).double()
    optimizer = optim.Adam(mynet.parameters(), lr=lr)
    eval = SegmentLevel_Eval(audio_embeddings)
    train_loss_array = []
    test_loss_array = []
    top10_acc_train_array = []
    top10_acc_test_array = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        test_loss = 0.0
        mynet.train() # Set the model to training mode
        for data, channel_layout, subjects, ground_truth, segment_label in train_loader:
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
            top10_acc_train = eval.get_correct(output, segment_label, 10)
        
        mynet.eval() # Set the model to evaluation mode
        for data, channel_layout, subjects, ground_truth, segment_label in test_loader:
            data = data.to(device)
            ground_truth = ground_truth.to(device)
            channel_layout = channel_layout.to(device)
            subjects = subjects.to(device).long()
            output = mynet(data, channel_layout, subjects)
            loss = criterion(output, ground_truth)
            test_loss += loss.item() * data.shape[0]
            top10_acc_test = eval.get_correct(output, segment_label, 10)
        
        train_loss = train_loss / train_set_size
        test_loss = test_loss / test_set_size
        train_loss_array.append(train_loss)
        test_loss_array.append(test_loss)
        top10_acc_train_array.append(top10_acc_train)
        top10_acc_test_array.append(top10_acc_test)
        print('Epoch:', epoch, 'Train Loss:', train_loss, 'Test Loss:', test_loss, 'Top-10 Train Acc:', top10_acc_train, 'Top-10 Test Acc:', top10_acc_test)
    return mynet, train_loss_array, test_loss_array, top10_acc_train_array, top10_acc_test_array

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
        
        # Load brain data while generating the training, validation and test set
        for i in len(brain_file_list):
            data_file = brain_file_list[i]
            data = np.load(data_file, allow_pickle=True)
            
            num_samples = data['brain_segments'].shape[0]
            assert num_samples == len_audio, 'The number of audio embeddings and brain segments do not match'
            brain_segments = data['brain_segments']
            mne_info = data['mne_info'].item()
            layout_tmp = mne.find_layout(mne_info)
            channel_layout = torch.tensor(layout_tmp.pos[:, :2])[None]
            channel_layout = channel_layout.repeat(num_samples, 1, 1)
            subject_num = i
            
            # Get data for the test set
            test_brain_segments = torch.cat((test_brain_segments, torch.tensor(brain_segments[test_mask])), dim = 0)
            test_audio_embeddings = torch.cat((test_audio_embeddings, audio_embeddings[test_mask]), dim = 0)
            test_segment_label = np.append(test_segment_label, segment_label[test_mask])
            test_subject_num = np.append(test_subject_num, np.repeat(subject_num, np.sum(test_mask)))
            test_channel_layout = torch.cat((test_channel_layout, channel_layout[test_mask]), dim = 0)
            
            train_subject_num = np.append(train_subject_num, np.repeat(subject_num, np.sum(train_mask)))
            train_brain_segments = torch.cat((train_brain_segments, torch.tensor(brain_segments[train_mask])), dim = 0)
            train_audio_embeddings = torch.cat((train_audio_embeddings, audio_embeddings[train_mask]), dim = 0)
            train_segment_label = np.append(train_segment_label, segment_label[train_mask])
            train_channel_layout = torch.cat((train_channel_layout, channel_layout[train_mask]), dim = 0)
            
            validation_subject_num = np.append(validation_subject_num, np.repeat(subject_num, np.sum(validation_mask)))
            validation_brain_segments = torch.cat((validation_brain_segments, torch.tensor(brain_segments[validation_mask])), dim = 0)
            validation_audio_embeddings = torch.cat((validation_audio_embeddings, audio_embeddings[validation_mask]), dim = 0)
            validation_channel_layout = torch.cat((validation_channel_layout, channel_layout[validation_mask]), dim = 0)
            validation_segment_label = np.append(validation_segment_label, segment_label[validation_mask])

        train_subject_num = torch.tensor(train_subject_num)
        validation_subject_num = torch.tensor(validation_subject_num)
        test_subject_num = torch.tensor(test_subject_num)
        train_segment_label = torch.tensor(train_segment_label)
        validation_segment_label = torch.tensor(validation_segment_label)
        test_segment_label = torch.tensor(test_segment_label)
        
        dataset = {'train': myDataset(train_brain_segments, train_audio_embeddings, train_segment_label, train_channel_layout, train_subject_num),
                   'validation': myDataset(validation_brain_segments, validation_audio_embeddings, validation_segment_label, validation_channel_layout, validation_subject_num),
                   'test': myDataset(test_brain_segments, test_audio_embeddings, test_segment_label, test_channel_layout, test_subject_num)}
        
        return dataset, audio_embeddings
        

def __main__():
    # Params
    USE_CUDA = True
    dataPth = '../../Data/brennanProcessed/'
    num_workers = 1
    lr = 3e-4
    num_epochs = 40
    
    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    maximum_thread = multiprocessing.cpu_count()
    print('Available threads:', maximum_thread)
    print('Num workers:', num_workers)
    brain_pth = dataPth + 'brain/'
    audio_pth = dataPth + 'audio/'
    
    # Variables
    num_subject = 0 # number of subjects
    data_file_list = []
    
    train_brain_segments = torch.tensor([], device = device)
    train_audio_embeddings = torch.tensor([], device = device)
    train_channel_layout = torch.tensor([], device = device)
    train_subject_num = np.array([]) # The subject number of each sample
    train_segment_label = np.array([]) # The segment number of each sample
    
    test_brain_segments = torch.tensor([], device = device)
    test_audio_embeddings = torch.tensor([], device = device)
    test_channel_layout = torch.tensor([], device = device)
    test_subject_num = np.array([]) # The subject number of each sample
    test_segment_label = np.array([]) # The segment number of each sample
    
    total_samples = 0 # number of samples
    
    # Load the audio data from file
    audio_embeddings = np.load(audio_pth + get_file(audio_pth, '.npz'))['audio_embeddings']
    segment_label = np.arange(audio_embeddings.shape[0]).astype(int)
    
    # Load the brain data from file
    for __,__,filenames in os.walk(brain_pth): 
    # print(root)
    # print(dirname)
    # print(filenames)
        for filename in filenames:  
            if os.path.splitext(filename)[-1]=='.npz':
                num_subject += 1
                data_file_list.append(filename)
    
    train_set_proportion = 0.8
    subject_num = 0
    for data_file in data_file_list:
        data = np.load(brain_pth + data_file, allow_pickle=True)
        print(data['mne_info'])
        num_samples = data['brain_segments'].shape[0]
        train_epoch = int(num_samples * train_set_proportion)
        subject_num_tmp = np.repeat(subject_num, num_samples)
        print('Loading Subject ', subject_num_tmp[0])
        brain_segments = data['brain_segments']
        mne_info = data['mne_info']
        layout_tmp = mne.channels.find_layout(mne_info)
        channel_layout = torch.tensor(layout_tmp.pos[:, :2], device = device)[None]
        channel_layout = channel_layout.repeat(num_samples, 1, 1)
        
        train_subject_num = np.append(train_subject_num, subject_num_tmp[:train_epoch])
        train_brain_segments = torch.cat((train_brain_segments, torch.tensor(brain_segments[:train_epoch],device = device)), dim = 0)
        train_audio_embeddings = torch.cat((train_audio_embeddings, torch.tensor(audio_embeddings[:train_epoch], device = device)), dim = 0)
        train_segment_label = np.append(train_segment_label, segment_label[:train_epoch])
        train_channel_layout = torch.cat((train_channel_layout, channel_layout[:train_epoch]), dim = 0)
        
        test_subject_num = np.append(test_subject_num, subject_num_tmp[train_epoch:])
        test_brain_segments = torch.cat((test_brain_segments, torch.tensor(brain_segments[train_epoch:],device = device)), dim = 0)
        test_audio_embeddings = torch.cat((test_audio_embeddings, torch.tensor(audio_embeddings[train_epoch:], device = device)), dim = 0)
        test_channel_layout = torch.cat((test_channel_layout, channel_layout[train_epoch:]), dim = 0)
        test_segment_label = np.append(test_segment_label, segment_label[train_epoch:])

        total_samples += num_samples
        subject_num += 1
    
    print('Total number of subjects:', num_subject)
    print('Total number of samples:', total_samples)
    print('Num of training samples:', train_brain_segments.shape[0])
    print('Num of testing samples:', test_brain_segments.shape[0])
    print('Creating Training Dataset')
    train_dataset = myDataset(train_brain_segments, train_audio_embeddings, train_segment_label, train_channel_layout, train_subject_num)
    print('Creating Testing Dataset')
    test_dataset = myDataset(test_brain_segments, test_audio_embeddings, test_segment_label, test_channel_layout, test_subject_num)
    print('Creating Training DataLoader')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    mynet, train_loss_array, test_loss_array, top10_train_acc, top10_test_acc = train_net(device, train_loader, test_loader, audio_embeddings, train_brain_segments.shape[0], test_brain_segments.shape[0], num_subject, lr=lr, num_epochs=num_epochs)
    
    # Save the model
    torch.save(mynet.state_dict(), 'mynet.pth')
    print('Model saved')
    # Save the loss arrays
    np.save('result.npy', train_loss_array, test_loss_array, top10_train_acc, top10_test_acc)
    print('Result saved')
    
if __name__ == '__main__':
    __main__()