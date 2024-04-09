# import dataHandler
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
    def __init__(self, n_subjects, device):
        super(myNet, self).__init__()
        self.spatial_attention = net.SpatialAttention(chout=270, n_freqs=32, r_drop=0.2, margin=0.1).to(device)
        self.subject_layer = net.SubjectLayers(n_channels=270, n_subjects=n_subjects).to(device)
        self.conv1 = nn.Conv1d(in_channels=270, out_channels=270, kernel_size=1, stride=1, padding=0, dtype=torch.double, device=device)
        self.convseq = nn.ModuleList()
        for k in range(5):
            self.convseq.append(net.ConvSequence(k, dilation_period=5, groups=1, dtype=torch.double).to(device))
        self.conv2 = nn.Conv1d(in_channels=320, out_channels=640, kernel_size=1, stride=1, padding=0, device=device)
        self.gelu = nn.GELU()
        self.conv3 = nn.Conv1d(in_channels=640, out_channels=1024, kernel_size=1, stride=1, padding=0, device=device)
    
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

def test_net(device, test_loader, audio_embeddings, n_subjects, model_to_load):
    mynet = myNet(n_subjects, device)
    mynet.to(device).double()
    mynet.load_state_dict(model_to_load)
    total = 0
    test_loss = 0
    top10_acc_test = 0
    mynet.eval() # Set the model to evaluation mode
    criterion = net.ClipLoss().to(device).double()
    eval = SegmentLevel_Eval(audio_embeddings)
    for data, channel_layout, subjects, ground_truth, segment_label in test_loader:
        # print('testing')
        
        total += data.shape[0]
        data = data.to(device)
        ground_truth = ground_truth.to(device)
        channel_layout = channel_layout.to(device)
        subjects = subjects.to(device).long()
        segment_label = segment_label.to(device)
        
        output = mynet(data, channel_layout, subjects)
        loss = criterion(output, ground_truth)
        test_loss += loss.item() * data.shape[0]
        top10_acc_test += eval.get_correct(output.detach(), segment_label.detach(), 10)
    top10_acc_test = top10_acc_test / total
    print('Top-10 Test Acc:', top10_acc_test)
    return top10_acc_test

def train_net(device, train_loader, validation_loader, test_loader, audio_embeddings, train_set_size, validation_set_size, n_subjects, lr, num_epochs, load_model=False, model_to_load=None):
    
    best_model = None # Temp for the best performing model on the validation set
    best_model_accuracy = 0 # Temp for the best accuracy on the validation set

    mynet = myNet(n_subjects, device)
    mynet.to(device).double()
    if load_model and model_to_load is not None:
        mynet.load_state_dict(torch.load(model_to_load))
    criterion = net.ClipLoss().to(device).double()
    optimizer = optim.Adam(mynet.parameters(), lr=lr)
    eval = SegmentLevel_Eval(audio_embeddings)
    train_loss_array = []
    validation_loss_array = []
    top10_acc_train_array = []
    top10_acc_validation_array = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        validation_loss = 0.0
        top10_acc_validation = 0.0
        top10_acc_train = 0.0
        total = 0
        mynet.train() # Set the model to training mode
        for data, channel_layout, subjects, ground_truth, segment_label in train_loader:
            # print(data.shape)
            # print('training')
            total += data.shape[0]
            data = data.to(device)
            ground_truth = ground_truth.to(device)
            channel_layout = channel_layout.to(device)
            subjects = subjects.to(device).long()
            segment_label = segment_label.to(device)
            
            optimizer.zero_grad()
            output = mynet(data, channel_layout, subjects)
            loss = criterion(output, ground_truth)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.shape[0]
            top10_acc_train += eval.get_correct(output.detach(), segment_label.detach(), 10)
        top10_acc_train = top10_acc_train / total
        
        total = 0
        mynet.eval() # Set the model to evaluation mode
        for data, channel_layout, subjects, ground_truth, segment_label in validation_loader:
            # print('testing')
            total += data.shape[0]
            data = data.to(device)
            ground_truth = ground_truth.to(device)
            channel_layout = channel_layout.to(device)
            subjects = subjects.to(device).long()
            segment_label = segment_label.to(device)
            output = mynet(data, channel_layout, subjects)
            loss = criterion(output, ground_truth)
            validation_loss += loss.item() * data.shape[0]
            top10_acc_validation += eval.get_correct(output.detach(), segment_label.detach(), 10)
        top10_acc_validation = top10_acc_validation / total
        
        train_loss = train_loss / train_set_size
        validation_loss = validation_loss / validation_set_size
        train_loss_array.append(train_loss)
        validation_loss_array.append(validation_loss)
        
        top10_acc_train_array.append(top10_acc_train)
        top10_acc_validation_array.append(top10_acc_validation)
        
        print('Epoch:', epoch, 'Train Loss:', train_loss, 'Validation Loss:', validation_loss, 'Top-10 Train Acc:', top10_acc_train, 'Top-10 Validation Acc:', top10_acc_validation)
        
        if top10_acc_validation > best_model_accuracy:
            best_model_accuracy = top10_acc_validation
            best_model = mynet.state_dict()
            total = 0
            test_loss = 0
            top10_acc_test = 0
            mynet.eval() # Set the model to evaluation mode
            for data, channel_layout, subjects, ground_truth, segment_label in test_loader:
            # print('testing')
        
                total += data.shape[0]
                data = data.to(device)
                ground_truth = ground_truth.to(device)
                channel_layout = channel_layout.to(device)
                subjects = subjects.to(device).long()
                segment_label = segment_label.to(device)
        
                output = mynet(data, channel_layout, subjects)
                loss = criterion(output, ground_truth)
                test_loss += loss.item() * data.shape[0]
                top10_acc_test += eval.get_correct(output.detach(), segment_label.detach(), 10)
            top10_acc_test = top10_acc_test / total
            print('Test Loss:', test_loss, 'Top-10 Test Acc:', top10_acc_test)
    
    return best_model, train_loss_array, validation_loss_array, top10_acc_train_array, top10_acc_validation_array

def __main__():
    mp.set_start_method('spawn')
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
    # Params
    trail = 1
    USE_CUDA = True
    dataPth = '../../Data/brennanProcessed/'
    num_workers = 2
    batch_size = 16
    shuffle = True
    lr = 3e-4
    num_epochs = 60
    test_set_proportion = 0.1
    cross_validation = 5
    do_cross_validation = False
    save_location = './results/'
    
    save_location = save_location + 'trail_' + str(trail) + '/'
    
    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    maximum_thread = mp.cpu_count()
    print('Available threads:', maximum_thread)
    print('Num workers:', num_workers)
    brain_pth = dataPth + 'brain/'
    audio_pth = dataPth + 'audio/'
    
    # Variables
    num_subject = 0 # number of subjects
    data_file_list = []
    
    total_samples = 0 # number of samples
    
    # Load the audio data from file
    audio_embeddings = np.load(audio_pth + get_file(audio_pth, '.npz'))['audio_embeddings']
    segment_label = np.arange(audio_embeddings.shape[0]).astype(int)
    
    # Create a mask for the test set
    test_set_mask = np.full(len(audio_embeddings), False, dtype=bool)
    True_choose = np.random.choice(len(audio_embeddings), int(len(audio_embeddings) * test_set_proportion), replace=False)
    test_set_mask[True_choose] = True
    
    # Get audio embeddings and labels for the test set
    test_audio_embeddings_source = audio_embeddings[test_set_mask]
    test_segment_label_source = segment_label[test_set_mask]
    
    # Remove the test set from the audio embeddings and labels
    audio_embeddings = audio_embeddings[~test_set_mask]
    segment_label = segment_label[~test_set_mask]
    
    # Load the brain data from file
    for __,__,filenames in os.walk(brain_pth): 
    # print(root)
    # print(dirname)
    # print(filenames)
        for filename in filenames:  
            if os.path.splitext(filename)[-1]=='.npz':
                num_subject += 1
                data_file_list.append(filename)
                break
    
    
    train_set_proportion = 1 - 1/cross_validation
    subject_num = 0
    for i in range(cross_validation):
        print('Cross Validation:', i)
        
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
        
        for data_file in data_file_list:
            data = np.load(brain_pth + data_file, allow_pickle=True)
            
            num_samples = data['brain_segments'].shape[0]
            brain_segments = data['brain_segments']
            mne_info = data['mne_info'].item()
            layout_tmp = mne.find_layout(mne_info)
            channel_layout = torch.tensor(layout_tmp.pos[:, :2])[None]
            channel_layout = channel_layout.repeat(num_samples, 1, 1)
            
            # Get data for the test set
            test_brain_segments = torch.cat((test_brain_segments, torch.tensor(data['brain_segments'][test_set_mask])), dim = 0)
            test_audio_embeddings = torch.cat((test_audio_embeddings, torch.tensor(test_audio_embeddings_source)), dim = 0)
            test_segment_label = np.append(test_segment_label, test_segment_label_source)
            test_subject_num = np.append(test_subject_num, np.repeat(subject_num, np.sum(test_set_mask)))
            test_channel_layout = torch.cat((test_channel_layout, channel_layout[test_set_mask]), dim = 0)
            
            # Remove the test set from the brain segments, channel layout
            brain_segments = brain_segments[~test_set_mask]
            channel_layout = channel_layout[~test_set_mask]
            
            num_samples = brain_segments.shape[0]
            train_epoch = int(num_samples * train_set_proportion)
            validation_epoch = num_samples - train_epoch
            subject_num_tmp = np.repeat(subject_num, num_samples)
            print('Loading Subject ', subject_num_tmp[0])
            
            
            epochs_tmp = np.arange(num_samples)
            validation_epoch_mask = (epochs_tmp >= i * validation_epoch) & (epochs_tmp < (i + 1) * validation_epoch)
            train_epoch_mask = ~validation_epoch_mask
            
            train_subject_num = np.append(train_subject_num, subject_num_tmp[train_epoch_mask])
            train_brain_segments = torch.cat((train_brain_segments, torch.tensor(brain_segments[train_epoch_mask])), dim = 0)
            train_audio_embeddings = torch.cat((train_audio_embeddings, torch.tensor(audio_embeddings[train_epoch_mask])), dim = 0)
            train_segment_label = np.append(train_segment_label, segment_label[train_epoch_mask])
            train_channel_layout = torch.cat((train_channel_layout, channel_layout[train_epoch_mask]), dim = 0)
            
            validation_subject_num = np.append(validation_subject_num, subject_num_tmp[validation_epoch_mask])
            validation_brain_segments = torch.cat((validation_brain_segments, torch.tensor(brain_segments[validation_epoch_mask])), dim = 0)
            validation_audio_embeddings = torch.cat((validation_audio_embeddings, torch.tensor(audio_embeddings[validation_epoch_mask])), dim = 0)
            validation_channel_layout = torch.cat((validation_channel_layout, channel_layout[validation_epoch_mask]), dim = 0)
            validation_segment_label = np.append(validation_segment_label, segment_label[validation_epoch_mask])

            total_samples += num_samples
            subject_num += 1
        train_subject_num = torch.tensor(train_subject_num)
        validation_subject_num = torch.tensor(validation_subject_num)
        train_segment_label = torch.tensor(train_segment_label)
        validation_segment_label = torch.tensor(validation_segment_label)
        audio_embeddings = torch.tensor(audio_embeddings)
        
        # Test the model
        print('Testing the model')
        test_subject_num = torch.tensor(test_subject_num)
        test_segment_label = torch.tensor(test_segment_label)
        test_dataset = myDataset(test_brain_segments, test_audio_embeddings, test_segment_label, test_channel_layout, test_subject_num)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        # top10_test_acc = test_net(device, test_loader, audio_embeddings, num_subject, mynet)
        
        print('Total number of subjects:', num_subject)
        print('Total number of samples:', total_samples)
        print('Num of training samples:', train_brain_segments.shape[0])
        print('Num of validation samples:', validation_brain_segments.shape[0])
        print('Creating Training Dataset')
        train_dataset = myDataset(train_brain_segments, train_audio_embeddings, train_segment_label, train_channel_layout, train_subject_num)
        print('Creating Validation Dataset')
        validation_dataset = myDataset(validation_brain_segments, validation_audio_embeddings, validation_segment_label, validation_channel_layout, validation_subject_num)
        print('Creating Training DataLoader')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        mynet, train_loss_array, validation_loss_array, top10_train_acc, top10_validation_acc = train_net(device, train_loader, validation_loader, test_loader, audio_embeddings, train_brain_segments.shape[0], validation_brain_segments.shape[0], num_subject, lr=lr, num_epochs=num_epochs)
        
        # Save the model
        save_location_tmp = save_location + 'cross_validation_' + str(i) + '/'
        if not os.path.exists(save_location_tmp):
            os.makedirs(save_location_tmp)
        torch.save(mynet, save_location_tmp + 'mynet.pth')
        print('Model saved')
        # Save the loss arrays
        # np.save('result.npy', train_loss_array, test_loss_array, top10_train_acc, top10_test_acc)
        sio.savemat(save_location_tmp + 'result.mat', {'train_loss': train_loss_array, 'validation_loss': validation_loss_array,\
            'top10_train_acc': top10_train_acc, 'top10_validation_acc': top10_validation_acc, 'top10_test_acc': top10_test_acc})
        print('Result saved')
        if not do_cross_validation:
            break
    
if __name__ == '__main__':
    __main__()