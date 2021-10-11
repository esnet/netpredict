import torch
import torch.nn as nn
import torch.nn.functional as F

latent_dims = 2
capacity = 32

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=5, stride=3, padding=0)  # out: c x 25 x 25
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=0)  # out: c x 11 x 11
        self.conv3 = nn.Conv2d(in_channels=c * 2, out_channels=c * 4, kernel_size=4, stride=2, padding=0) # out: c x 7 x 7
        self.conv4 = nn.Conv2d(in_channels=c * 4, out_channels=c * 4, kernel_size=3, stride=1, padding=0) # out: c x 5 x 5
        self.conv5 = nn.Conv2d(in_channels=c * 4, out_channels=c * 4, kernel_size=3, stride=1, padding=0)  # out: c x 3 x 3

        self.fc_mu = nn.Linear(in_features=c * 4 * 3 * 3, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c * 4 * 3 * 3, out_features=latent_dims)

    def forward(self, x):
        x = F.relu( self.conv1(x) )
        x = F.relu( self.conv2(x) )
        x = F.relu( self.conv3(x) )
        x = F.relu( self.conv4(x) )
        x = F.relu( self.conv5(x) )

        x = torch.flatten(x, 1) #x.view(x.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c * 4 * 3 * 3)

        self.conv5 = nn.ConvTranspose2d(in_channels=c * 4, out_channels=c * 4, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.ConvTranspose2d(in_channels=c * 4, out_channels=c * 4, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.ConvTranspose2d(in_channels=c * 4, out_channels=c * 2, kernel_size=4, stride=2, padding=0)
        self.conv2 = nn.ConvTranspose2d(in_channels=c * 2, out_channels=c, kernel_size=4, stride=2, padding=0)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=5, stride=3, padding=0)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), capacity * 4, 3, 3)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        feature = x
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))

        x = torch.sigmoid( self.conv1(x) )  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x, feature


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon, Exract_feature = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar, Exract_feature

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
