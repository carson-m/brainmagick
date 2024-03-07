# Testfield for dataloader
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class myDataset(Dataset):
    def __init__(self, data, additional, lables):
        super(myDataset, self).__init__()
        self.data = data
        self.lables = lables
        self.additional = additional
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.additional[idx], self.lables[idx]
    
def __main__():
    data = np.random.rand(10)
    additional = np.random.rand(10)
    lables = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    dataset = myDataset(data, additional, lables)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)
    for i, (data, additional, lable) in enumerate(dataloader):
        print(data, additional, lable)
        print(type(data))
    
if __name__ == '__main__':
    __main__()