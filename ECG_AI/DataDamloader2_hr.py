import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import json


class DataDamloader2_hr(Dataset):
    def __init__(
        self,
        hr_array,
        desire_len
    ):
        self.hr_array = hr_array
        self.desire_len = desire_len


    def __len__(self):
        return len(self.hr_array)

    def get_labels(self):
        return [ arr[1] for arr in self.hr_array]

    def __getitem__(self, index):

        hr_arr = self.hr_array[index][0]
        hr_arr = self._adjust_len(hr_arr)
        label = self.hr_array[index][1]

        return 0,torch.tensor(hr_arr).double().unsqueeze(0).float(), torch.tensor(label).long()


        return hr_arr, label

    def _adjust_len(self,arr):
        if len(arr)>self.desire_len:
            arr = arr[:self.desire_len]
        elif len(arr)<self.desire_len:
            zero_n = self.desire_len-len(arr)
            arr = np.pad(arr,(0,zero_n))
        
        return arr



