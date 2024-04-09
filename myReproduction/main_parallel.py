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
    
    def get_masks_random(self):
        max_overlap = self.tmax - self.tmin
        total_size = len(self.event_times)
        test_proportion = 1 - self.train_set_proportion - self.validation_set_proportion
        test_size = int(test_proportion * total_size)
        test_mask = np.full(total_size, False, dtype=bool)
        test_mask[np.random.choice(total_size, test_size, replace=False)] = True
        safe_mask = ~test_mask
        
        for time in self.event_times[test_mask]:
            conflict = (self.event_times > time - max_overlap) & (self.event_times < time + max_overlap)
            safe_mask[conflict] = False
        
        validation_size = int(self.validation_set_proportion * total_size)
        print(np.sum(safe_mask))
        validation_choose = np.full(np.sum(safe_mask), False, dtype=bool)
        validation_choose[np.random.choice(np.sum(safe_mask), validation_size, replace=False)] = True
        validation_mask = np.full(total_size, False, dtype=bool)
        validation_mask[safe_mask] = validation_choose
        
        for time in self.event_times[validation_mask]:
            conflict = (self.event_times > time - max_overlap) & (self.event_times < time + max_overlap)
            safe_mask[conflict] = False
        
        train_mask = safe_mask
        
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

def test_net(test_loader, audio_embeddings, model_to_load):
    mynet = model_to_load
    total = 0
    mynet.eval() # Set the model to evaluation mode
    criterion = net.ClipLoss().double().cuda()
    eval = SegmentLevel_Eval(audio_embeddings)
    test_loss = 0
    top10_acc_test = 0
    for data, channel_layout, subjects, ground_truth, segment_label in test_loader:
        # print('testing')
        
        total += data.shape[0]
        data = data.cuda(non_blocking=True)
        ground_truth = ground_truth.cuda(non_blocking=True)
        channel_layout = channel_layout.cuda(non_blocking=True)
        subjects = subjects.long().cuda(non_blocking=True)
        segment_label = segment_label.cuda(non_blocking=True)
        
        output = mynet(data, channel_layout, subjects)
        loss = criterion(output, ground_truth)
        test_loss += loss.item() * data.shape[0]
        top10_acc_test += eval.get_correct(output.detach(), segment_label.detach(), 10)
    top10_acc_test = top10_acc_test / total
    test_loss = test_loss / total
    print(f'Test Loss: {test_loss:.4f}, Top-10 Test Acc: {top10_acc_test:.4f}')
    return test_loss, top10_acc_test

def train_net(gpus, train_loader, validation_loader, test_loader, audio_embeddings, n_subjects, lr, num_epochs, patience, load_model=False, model_to_load=None):
    mynet = myNet(n_subjects)
    best_model = None # Temp for the best performing model on the validation set
    best_model_loss = 10 # Temp for the best loss on the validation set

    if load_model and model_to_load is not None:
        mynet.load_state_dict(torch.load(model_to_load))
    mynet = nn.DataParallel(mynet, device_ids=gpus).cuda()
    criterion = net.ClipLoss().double().cuda()
    optimizer = optim.Adam(mynet.parameters(), lr=lr, weight_decay=1.3e-5)
    eval_train = SegmentLevel_Eval(audio_embeddings['train'])
    eval_validation = SegmentLevel_Eval(audio_embeddings['validation'])
    eval_test = SegmentLevel_Eval(audio_embeddings['test'])
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
            
            data = data.cuda(non_blocking=True)
            ground_truth = ground_truth.cuda(non_blocking=True)
            channel_layout = channel_layout.cuda(non_blocking=True)
            subjects = subjects.long().cuda(non_blocking=True)
            segment_label = segment_label.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            output = mynet(data, channel_layout, subjects)
            loss = criterion(output, ground_truth)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.shape[0]
            top10_acc_train += eval_train.get_correct(output.detach(), segment_label.detach(), 10)
        top10_acc_train = top10_acc_train / total
        train_loss = train_loss / total
        
        total = 0
        mynet.eval() # Set the model to evaluation mode
        for data, channel_layout, subjects, ground_truth, segment_label in validation_loader:
            # print('testing')
            total += data.shape[0]
            
            data = data.cuda(non_blocking=True)
            ground_truth = ground_truth.cuda(non_blocking=True)
            channel_layout = channel_layout.cuda(non_blocking=True)
            subjects = subjects.long().cuda(non_blocking=True)
            segment_label = segment_label.cuda(non_blocking=True)
            
            output = mynet(data, channel_layout, subjects)
            loss = criterion(output, ground_truth)
            validation_loss += loss.item() * data.shape[0]
            top10_acc_validation += eval_validation.get_correct(output.detach(), segment_label.detach(), 10)
        top10_acc_validation = top10_acc_validation / total
        validation_loss = validation_loss / total
        
        train_loss_array.append(train_loss)
        validation_loss_array.append(validation_loss)
        
        top10_acc_train_array.append(top10_acc_train)
        top10_acc_validation_array.append(top10_acc_validation)
        
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, Top-10 Train Acc: {top10_acc_train:.4f}, Top-10 Validation Acc: {top10_acc_validation:.4f}')
        
        if validation_loss < best_model_loss:
            best_model_loss = validation_loss
            best_model = mynet
            total = 0
            test_loss = 0
            top10_acc_test = 0
            mynet.eval() # Set the model to evaluation mode
            for data, channel_layout, subjects, ground_truth, segment_label in test_loader:
            # print('testing')
        
                total += data.shape[0]
                data = data.cuda(non_blocking = True)
                ground_truth = ground_truth.cuda(non_blocking = True)
                channel_layout = channel_layout.cuda(non_blocking = True)
                subjects = subjects.long().cuda(non_blocking = True)
                segment_label = segment_label.cuda(non_blocking = True)
        
                output = mynet(data, channel_layout, subjects)
                loss = criterion(output, ground_truth)
                test_loss += loss.item() * data.shape[0]
                top10_acc_test += eval_test.get_correct(output.detach(), segment_label.detach(), 10)
            top10_acc_test = top10_acc_test / total
            test_loss = test_loss / total
            print(f'Test Loss: {test_loss:.4f}, Top-10 Test Acc: {top10_acc_test:.4f}')
            
    return best_model, train_loss_array, validation_loss_array, top10_acc_train_array, top10_acc_validation_array

def __main__():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
    mp.set_start_method('spawn')
    gpus = [0,1]
    
    # Params
    trail = 16
    dataPth = '../../Data/brennanProcessed/'
    num_workers = 4
    lr = 2.5e-4
    num_epochs = 12
    patience = 15
    batch_size = 256
    training_set_proportion = 0.7
    validation_set_proportion = 0.2
    tmin = -0.5
    tmax = 2.5
    save_location = './results/'
    
    save_location = save_location + 'trail_' + str(trail) + '/'
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    
    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"

    maximum_thread = mp.cpu_count()
    print('Available threads:', maximum_thread)
    print('Num workers:', num_workers)
    brain_pth = dataPth + 'brain/'
    audio_pth = dataPth + 'audio/S0.npz'
    time_pth = dataPth
    
    # Load Data
    time_file = get_file(time_pth, '.npz')
    assert time_file is not None, 'No time file found'
    word_times = np.load(time_pth + time_file)['word_times']
    brain_file_list = [brain_pth + f for f in os.listdir(brain_pth) if os.path.isfile(brain_pth + f)]
    # brain_file_list = brain_file_list[:2]# !!!!! For Testing Code Only
    print("Loading data")
    dataGen = datasetGenerator(training_set_proportion, validation_set_proportion, word_times, tmin, tmax)
    datasets, audio_embeddings = dataGen.load_data(brain_file_list, audio_pth)
    num_subjects = len(brain_file_list)
    print("Total number of loaded subjects: ", num_subjects)
    
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(datasets['validation'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    mynet, train_loss_array, validation_loss_array, top10_train_acc, top10_validation_acc = train_net(gpus, train_loader, validation_loader, test_loader, audio_embeddings, num_subjects, lr=lr, num_epochs=num_epochs, patience=patience)

    # Test the model
    print('Testing the model')
    test_loss, top10_test_acc = test_net(test_loader, audio_embeddings['test'], mynet)

    # Save the model
    torch.save(mynet, save_location + 'mynet.pth')
    print('Model saved')
    # Save the loss arrays
    # np.save('result.npy', train_loss_array, test_loss_array, top10_train_acc, top10_test_acc)
    sio.savemat(save_location + 'result.mat', {'train_loss': train_loss_array, 'validation_loss': validation_loss_array, 'top10_train_acc': top10_train_acc, 'top10_validation_acc': top10_validation_acc, 'test_loss': test_loss, 'top10_test_acc': top10_test_acc})
    print('Result saved')
    
if __name__ == '__main__':
    __main__()