import os
import pandas as pd
import torch

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
from torchvision.io import read_image
import numpy as np

class DatasetImageLabel(Dataset):

    def __init__(self, csv_file, root_dir, device):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.device = device
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor() # transfer image from 0-255 to 0-1, and numpy to tensors
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index][0])
        image = read_image(img_path)
        image = image.to(self.device)
        y_label = torch.tensor( int(self.annotations.iloc[index][1]) ) # float or int
        image = self.transforms(image)

        return (image, y_label)


class DatasetVAE(Dataset):

    def __init__(self, csv_file, csv_label_file, root_dir, device):

        self.x_set, self.y_set = pd.DataFrame(), pd.DataFrame()
        for file1, file2 in zip(csv_label_file,csv_file):
            temp_y = pd.read_csv(file1, header=None)
            self.y_set = pd.concat( [self.y_set, temp_y] )
            temp_x = pd.read_csv(file2, header=None)
            self.x_set = pd.concat( [self.x_set, temp_x] )

        self.len = self.__len__()
        self.root_dir = root_dir
        self.device = device

        # maxsingal = self.x_set.values.max()
        # minsingal = self.x_set.values.min()

        x = self.CustomeNormalize(self.x_set)
        x_reshaped = x.values.reshape(self.len, 1, 104, 104)
        self.X = torch.from_numpy(x_reshaped)

    def __len__(self):
        return len(self.y_set)

    def CustomeNormalize(self, inputs, min_value=-1.0, max_value=1.0):
        scaled_inputs = (inputs - min_value) / (max_value - min_value)
        return scaled_inputs

    def __getitem__(self, idx):
        'Generates one sample of data'

        return self.X[idx], self.y_set.iloc[idx][0]


