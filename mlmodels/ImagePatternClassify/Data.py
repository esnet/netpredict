import os
import pandas as pd
import torch

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
from torchvision.io import read_image

class DatasetImageLabel(Dataset):

    def __init__(self, csv_file, root_dir, device):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.device = device
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128,128)),
            transforms.ToTensor() # transfer image from 0-255 to 0-1, and numpy to tensors
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index][0])
        image = read_image(img_path)
        image = image.to(self.device)
        y_label = torch.tensor( self.annotations.iloc[index][1] ) # float or int
        image = self.transforms(image)

        return (image, y_label)

