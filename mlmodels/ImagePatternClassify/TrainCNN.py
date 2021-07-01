import argparse
import numpy as np
import datetime
import json
from scipy.stats import special_ortho_group
import pandas as pd
import CNN
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from Data import DatasetImageLabel
from torchvision.io import read_image
# for creating validation set
from sklearn.model_selection import train_test_split

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = transforms.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def train_CNNmodel(model, dataset, device='cpu',
                patience=7, n_epochs=2, batch_size=2, verbose=False):

    model.to(device)
    df = pd.DataFrame(columns=['train_loss', 'valid_loss'])

    # divide data to training and test randomly for every epoch
    # train_x -> torch.Size([4500, 5, 5, 25])
    # train_y -> torch.Size([4500, 8])
    vali_num = int(0.5 * len(dataset))
    train_num = len(dataset) - vali_num
    train_dataset, vali_dataset = random_split(dataset, [train_num, vali_num])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=vali_dataset, batch_size=batch_size, shuffle=True)

    # define optimizer Adam, SGD with mommentum
    # optimizer = Adam(model.parameters(), lr=0.0148) #learning rate
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0, nesterov=False)

    # Evaluation tool MSE
    criterion = nn.CrossEntropyLoss()

    # Schedule learning rate
    # scheduler = MultiStepLR( optimizer, milestones = [25, 40], gamma = 0.1)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience) # "min" mode or "max" mode

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []

    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch, (x, y) in enumerate(train_loader, 0):

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # get the inputs;
            output_train, target_train = model( x.float().to(device) ), y.float().to(device)
            # calculate the loss
            loss_train = criterion(output_train.reshape(batch_size), target_train)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss_train.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss_train.item())

        #####################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for (x, y) in valid_loader:

            # forward pass: compute predicted outputs by passing inputs to the model
            output_val, target_val = model( x.float().to(device) ), y.float().to(device)
            # calculate the loss
            loss_val = torch.sqrt( criterion(output_val.reshape(vali_num), target_val) )
            # record validation loss
            valid_losses.append(loss_val.item())

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        df.loc[epoch] = [train_loss, valid_loss]

        # print training loss, validation loss, and learning rate
        epoch_len = len(str(n_epochs))
        curr_lr = optimizer.param_groups[0]['lr']

        print(f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
              + f"train_loss: {train_loss:4.2f} deg, valid_loss: {valid_loss:4.2f} deg, lr: {curr_lr:4.5f}" )

        # adjust lr
        # scheduler.step() # for MultiStepLR
        scheduler.step( valid_loss) # for ReduceLROnPlateau

        # # clear lists to track next epoch
        train_losses = []
        valid_losses = []

    # # load the last checkpoint with the best model
    # model.load_state_dict(torch.load('checkpoint.pt'))

    return model, df

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('-e', '--n_epochs', type=int, default=10, help='number of episodes')
    p.add_argument('--force_cpu', action='store_true', help='Forces CPU usage')
    args = p.parse_args()
    device = 'cpu' if args.force_cpu else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # original image size [1,320, 320]
    imgSample = read_image(str(Path('Data') / '????????.png'))
    show(imgSample)

    # defining the model
    model = CNN.Net() #, args.net_config
    print(model)

    dataset = DatasetImageLabel(csv_file = "ImageLabel.csv",
                                root_dir = "Data",
                                device = device)

    # open a file
    # with open("directory/fileName.xxx", "xx (e.g., w+)") as ...,
    # "w+": open a file for reading and writing. If file doesn't exit, create a file.
    # with open(fout.replace('pth', 'json'), 'w+') as f:
    #     json.dump(dataset.config, f, indent=4)

    print('Start Training...')
    start = datetime.datetime.now()

    net_trained, df = train_CNNmodel(
        model, dataset, device,
        n_epochs=args.n_epochs, verbose=args.verbose)

    end = datetime.datetime.now()