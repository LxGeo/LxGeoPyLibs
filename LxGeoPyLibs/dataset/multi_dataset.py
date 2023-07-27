# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:03:31 2022

@author: cherif
"""

from torch.utils.data import Dataset
import bisect
import itertools


class MultiDatasets(Dataset):
    
    def __init__(self, dataset_iterable):
        
        self.datasets = list(dataset_iterable)
        
        self.datasets_lengths = [len(d) for d in self.datasets]
        self.cumulative_lengths = [0] + list(itertools.accumulate(self.datasets_lengths))
        
        self.total_length = sum(self.datasets_lengths)
    
    
    def __len__(self):
        return self.total_length
    
    def __getitem__old__(self, idx, rem_datasets=None):
        
        if rem_datasets:
            first_dataset_len = len(rem_datasets[0])
            if idx < first_dataset_len:
                return rem_datasets[0][idx]
            else:
                return self.__getitem__(idx-first_dataset_len, rem_datasets[1:])
        else:
            idx = (idx+len(self))%len(self)
            return self.__getitem__(idx, self.datasets)
       
    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)

        if idx >= self.cumulative_lengths[-1]:
            raise IndexError("Index out of range")

        # Use binary search to find the appropriate dataset
        dataset_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        dataset_offset = self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][idx - dataset_offset]


