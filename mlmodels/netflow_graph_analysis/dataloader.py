import pandas as pd
import torch
import torch.utils.data import Dataset, DataLoader


class NetworkDataset(Dataset):
    def __init__(self, data_root):
        self.samples=[]

        with open(data_root,'r') as flow_file:
            for row in flow_file.read().splitlines():
                self.samples.append((time, link1,link2))
    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        return self.samples[idx]
