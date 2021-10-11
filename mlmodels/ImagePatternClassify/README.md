# DynamicDeepFlow

The objective of this study is to identify the unseen network traffic patterns based on the flow change correlation among 52 research sites and universities in ESnet. To meet this objective, we proposed an innovative unsupervised deep learning model, DynamicDeepFlow (DDF), that can recognize new flow patterns in real-time. The proposed DDF leverages a combination of deep learning model, variational autoencoder (VAE), and shallow learning model, k-means++ to recognize real traffic change patterns. 

## Train the VAE

we pre-trained the deep learning model VAE with the flow change correlation in years 2018 and 2019. The VAE can automatically extract complex features from input data without human intervention. This avoids manual feature extraction and the risk of dropping useful information in the data. The extracted features are then fed into the shallow learning model k-means++. The TrainVAE5.py file is used to train a VAE model for network traffic information extraction. _N_ indicate the number of epoches used in the training process. After training process, you can get three figures about the extracted mean and sigma features. 

```
TrainVAE5.py --n_epochs=N
```
Below please find the model structure information. The overall architecture of the VAE consists of five convolutional layers, five transposed convolutional layers, and three fully-connected layers.

VariationalAutoencoder(
  (encoder): Encoder(
    (conv1): Conv2d(1, 32, kernel_size=(5, 5), stride=(3, 3))
    (conv2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    (conv3): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2))
    (conv4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
    (conv5): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
    (fc_mu): Linear(in_features=1152, out_features=2, bias=True)
    (fc_logvar): Linear(in_features=1152, out_features=2, bias=True)
  )
  (decoder): Decoder(
    (fc): Linear(in_features=2, out_features=1152, bias=True)
    (conv5): ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
    (conv4): ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
    (conv3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2))
    (conv2): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2))
    (conv1): ConvTranspose2d(32, 1, kernel_size=(5, 5), stride=(3, 3))
  )
)

## Plot the Results
