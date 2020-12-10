# Script to load the mnist code, and return train- and testloader.
# Redid loader https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
import torch
import torchvision
import torchvision.transforms as transforms
import numbers

import matplotlib.pyplot as plt
import numpy as np
import math
import time
import h5py
def load_norms(file):
    data = {}
    with h5py.File(file, 'r') as f:
        for key in list(f.keys()):
            data[key] = np.array(f[key])
    norm = data['normalization']
    return norm

class dset_wAFO(torch.utils.data.Dataset):
    """
    I have a torch dataset class for loading of the pictures.
    This have support for adding artificial foreign objects.
    """
    def __init__(self, in_file, nx,ny,transform=None):
        super(dset_wAFO, self).__init__()
        self.file_path = in_file#h5py.File(in_file, 'r')
        self.dataset=None
        with h5py.File(self.file_path, 'r') as file:
            self.n_images, self.num_channels, self.nx, self.ny = file['images'].shape
        self.transform=transform

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
        
        pic = torch.tensor(self.dataset['images'][index]).float()
        label = self.dataset['labels'][index]
        label = torch.tensor(label).type(torch.LongTensor)

        if self.transform:
            for t in self.transform:
                if format(t)[:6] != 'AddAFO' :
                    pic = t(pic)
                else:
                    if (0.2 > np.random.random()) and (label ==0):
                        pic = t(pic)
                        label = torch.tensor(1).type(torch.LongTensor)
        return pic,label#pic.astype('float32'), label

    def __len__(self):
        return self.n_images

    def num_fo(self):
        c=0
        for i in range(0,self.n_images):
            c+=int(self.file['images'][i,-1])
        return c

class RandFlipRot(object):
    """Random flip and rotate image.
    """
    def __init__(self, size, p=0.5):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.p = p

    def __call__(self, img):
        if self.p < np.random.random():
            img = img.permute(0,2,1)
        if self.p < np.random.random():
            img = torch.flip(img,[1])
        if self.p < np.random.random():
            img = torch.flip(img,[2])
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
    
class ZeroChannel(object):
    """Zeroeth out channel - 1 channel.
    """
    def __init__(self, size, ch=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ch = ch-1
    def __call__(self, img):
        img[self.ch] = img[self.ch]*0.0
        return img
    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

def kfold_dataloaders(dset_train, dset_valid, kfold=5, batch_size=64, num_workers=0):
    """Function to return two lists of dataloaders for a k-fold train validation split.
       As input needs two torch datasets, which can be the same, but can also be made
       using different transformations."""
    dataset_size = len(dset_train)
    kpart = math.floor(dataset_size / kfold)
    # Create list
    indicies = np.arange(dataset_size)
    indicies_list = []
    for i in range(kfold):
        if i < kfold-1:
            indicies_list.append(indicies[i*kpart: (i+1)*kpart])
        else:
            indicies_list.append(indicies[i*kpart:])
    # Now make 5 sets each with one category out:
    ind_sets = []
    for i in range(kfold):
        val = indicies_list[i]
        tra = [indicies_list[j] for j in range(kfold) if j != i]
        tra = np.concatenate(tra)
        ind_sets.append([tra,val])
    train_loaders = []
    valid_loaders = []

    c=0
    for train_indices, valid_indices in ind_sets:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)

        train_loader = torch.utils.data.DataLoader(dset_train,batch_size=batch_size, shuffle=False, sampler = train_sampler, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(dset_valid,batch_size=batch_size, shuffle=False, sampler = valid_sampler, num_workers=num_workers)

        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
        c+=1
    return train_loaders, valid_loaders

def create_indices(dataset, kfold):
    dset = dset_wAFO(dataset,32,32, transform=None)
    dataset_size = len(dset)
    kpart = math.floor(dataset_size / kfold)
    # Create list
    indicies = np.arange(dataset_size)
    if kfold == 1: return [(indicies,indicies)]

    indicies_list = []
    for i in range(kfold):
        if i < kfold-1:
            indicies_list.append(indicies[i*kpart: (i+1)*kpart])
        else:
            indicies_list.append(indicies[i*kpart:])
    # Now make 5 sets each with one category out:
    ind_sets = []
    for i in range(kfold):
        val = indicies_list[i]
        tra = [indicies_list[j] for j in range(kfold) if j != i]
        tra = np.concatenate(tra)
        ind_sets.append([tra,val])
    return ind_sets

def k_dataload(dataset, indices, tforms, fo, batch_size=64, num_workers=0):
    dset = dset_wAFO(dataset,32,32, transform=tforms)
    subsetsampler = torch.utils.data.SubsetRandomSampler(indices)
    loader  = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, sampler = subsetsampler, num_workers=num_workers)
    return loader
