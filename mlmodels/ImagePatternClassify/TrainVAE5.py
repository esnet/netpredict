
import argparse
import numpy as np

import pandas as pd
import VAE_4

import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from Data import DatasetVAE
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.ion()
import matplotlib
import datetime
from matplotlib.ticker import FormatStrFormatter
# from torchsummary import summary

def show(imgs):

    img = transforms.ToPILImage()(imgs.to('cpu'))
    plt.imshow(np.asarray(img))
    # plt.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def vae_loss(recon_x, x, mu, logvar):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    # recon_loss = F.binary_cross_entropy(recon_x.float().view(-1, 104*104), x.view(-1, 104*104), reduction='sum')
    recon_loss = F.mse_loss(recon_x.float(), x )
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kldivergence /= 104*104
    return recon_loss + variational_beta * kldivergence

def train_CNNmodel(model, dataset, device='cpu',
                patience=7, n_epochs=3, batch_size=32, verbose=False):

    model.to(device)
    df = pd.DataFrame(columns=['train_loss', 'valid_loss'])

    # divide data to training and test randomly for every epoch
    vali_num = int(0.1 * len(dataset))
    train_num = len(dataset) - vali_num
    train_dataset, vali_dataset = random_split(dataset, [train_num, vali_num])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=vali_dataset, batch_size=batch_size, shuffle=True)

    # define optimizer Adam, SGD with mommentum
    optimizer = Adam(model.parameters(), lr=0.01 ) #learning rate
    # optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0, nesterov=False)

    # Schedule learning rate
    scheduler = MultiStepLR( optimizer, milestones = [25, 40], gamma = 0.1)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience) # "min" mode or "max" mode

    # Track the training loss as the model trains
    # train_losses = []
    # Track the validation loss as the model trains
    # valid_losses = []

    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        train_losses, train_num = 0, 0
        for batch, (x, y) in enumerate(train_loader, 0):

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # get the inputs;
            image_batch_recon, latent_mu, latent_logvar, _ = model( x.float().to(device) )
            # calculate the loss
            loss_train = vae_loss(image_batch_recon, x.float(), latent_mu, latent_logvar)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss_train.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            # train_losses.append(loss_train.item())
            train_losses += loss_train.item()
            train_num += 1

        #####################
        # validate the model #
        ######################
        # model.eval()  # prep model for evaluation
        # valid_acc, valid_num = 0, 0
        # for (x, y) in valid_loader:
        #
        #     # forward pass: compute predicted outputs by passing inputs to the model
        #     output_val, target_val = model( x.to(device) ), y.to(device)
        #     _, pred_val = torch.max(output_val, 1)
        #     # calculate the loss
        #     loss_val = criterion(output_val, target_val)
        #     # record validation loss
        #     # valid_losses.append(loss_val.item())
        #     valid_acc += torch.sum(pred_val == target_val)
        #     valid_num += target_val.size(0)

        # calculate average loss over an epoch
        train_epoch_loss = train_losses / train_num
        # valid_epoch_acc = valid_acc.double() / valid_num
        # train_loss = np.average(train_losses)
        # valid_loss = np.average(valid_losses)
        # df.loc[epoch] = [train_loss, valid_loss]

        # print training loss, validation loss, and learning rate
        epoch_len = len(str(n_epochs))
        curr_lr = optimizer.param_groups[0]['lr']

        print(f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
              + f"train_loss: {train_epoch_loss:4.5f}, lr: {curr_lr:4.5f}" )

        # adjust lr
        scheduler.step() # for MultiStepLR
        # scheduler.step( valid_loss) # for ReduceLROnPlateau

        # # clear lists to track next epoch
        # train_losses = []
        # valid_losses = []

    return model, valid_loader, df

def DeNormalize(Inputs, min_value=-1, max_value=1):
    Recover_inputs = min_value + (max_value - min_value)*Inputs
    return Recover_inputs

def plot_colormap(vmin, vmax, colormap, normalize):
    scalarmappaple = plt.cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(np.arange(vmin,vmax))
    plt.colorbar(scalarmappaple)

    plt.tight_layout()
    plt.show()

def plot_latent(autoencoder, x_set, y_set, num_batches=100):
    font = {'family': 'Times New Roman',
            'weight': 'normal',  # bold
            'size': 12}
    matplotlib.rc('font', **font)
    # vmin, vmax = 0, 7
    # normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
    plt.rcParams['savefig.dpi'] = 200  # Image Pixel
    plt.rcParams['figure.dpi'] = 200  # Resolution ratio
    plt.rcParams['figure.figsize'] = (5.0, 3.0)  # Set figure_size

    fig, ax = plt.subplots()

    mu, sigma = autoencoder.encoder(x_set.float().to(device))  # x_set.float(), x_set.float().to(device)
    mu_cpu = mu.to('cpu').detach().numpy()
    plt.scatter(mu_cpu[:, 0] * 10 ** 4, mu_cpu[:, 1] * 10 ** 4, c=y_set)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    # axs[1].xaxis.set_major_formatter('{x:.5f}')
    cbar = plt.colorbar()
    cbar.set_ticks(list(range(8)))
    cbar.set_ticklabels(
        ["Night", "Monday_day", "Tuesday_day", "Wednesday_day", "Thursday_day", "Friday_day", "Saturday_day",
         "Sunday_day"])

    plt.xlabel("$\mu_1$ $(10^{-4})$")
    plt.ylabel("$\mu_2$ $(10^{-4})$")

    # plt.xlim([-10, 30])
    # plt.ylim([5, 50])

    plt.tight_layout()
    plt.show(block=True)


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('-e', '--n_epochs', type=int, default = 3, help='number of episodes')
    p.add_argument('--force_cpu', action='store_true', help='Forces CPU usage')
    args = p.parse_args()
    device = 'cpu' if args.force_cpu else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # defining the model
    model = VAE_4.VariationalAutoencoder()
    print(model)

    # Load the training data
    variational_beta = 1
    dataset_2019 = DatasetVAE(csv_file = ["featurizers/2019/inputAE.csv"],
                         csv_label_file = ["featurizers/2019/AELabel.csv"],
                         root_dir = ["featurizers/2019"],
                         device = device)

    # Load the validation data
    dataset_2018 = DatasetVAE(csv_file = ["featurizers/2018/inputAE.csv"],
                         csv_label_file = ["featurizers/2018/AELabel.csv"],
                         root_dir = ["featurizers/2018"],
                         device = device)

    # Load the testing data
    dataset_2020 = DatasetVAE(csv_file = ["featurizers/2020/inputAE.csv"],
                         csv_label_file = ["featurizers/2020/AELabel.csv"],
                         root_dir = ["featurizers/2020"],
                         device = device)


    # Load a pre-trained model
    # parameters = torch.load("VAE_model.pt")
    # model.load_state_dict(parameters)
    # net_trained = model

    # Model training process
    print('Start Training...')
    start = datetime.datetime.now()

    net_trained, valid_loader, df = train_CNNmodel(
        model, dataset_2019, device,
        n_epochs=args.n_epochs, verbose=args.verbose)

    end = datetime.datetime.now()
    print( "Time comsumption of model training process: {}s".format( (end-start).total_seconds() ) )

    # Visulizing the model structure
    # summary(net_trained, (1, 104, 104))

    # Visualizing the preformance of the trained VAE model (i.e., net_trained) on training, validation, and test datasets.
    plot_latent(net_trained, dataset_2018.X, dataset_2018.y_set)
    plot_latent(net_trained, dataset_2019.X, dataset_2019.y_set)
    plot_latent(net_trained, dataset_2020.X, dataset_2020.y_set)

    # Save the trained model to current directory
    # torch.save(net_trained.state_dict(), "VAE_model.pt")
    # torch.save(model, "VAE_model.pt")
