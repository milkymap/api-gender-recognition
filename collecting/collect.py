import numpy as np 
import itertools as it 
import fucntools as ft 

import torch 
from torch.utils.data import Dataset 

class Collector(Dataset):
    def __init__(self, source_data, target_data):
        self.source_data = source_data
        self.target_data = target_data
    
    def __len__(self):
        return len(self.source_data)
    
    def __getitem__(self, index):
        current_source_data = self.source_data[index]  
        current_target_data = self.target_data[index] 

        tensorized_current_source_data = torch.from_numpy(current_source_data).float()
        tensorized_current_target_data = torch.tensor(current_target_data).long()

        return tensorized_current_source_data, tensorized_current_target_data