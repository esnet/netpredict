# DynamicDeepFlow

The objective of this study is to identify the unseen network traffic patterns based on the flow change correlation among 52 research sites and universities in ESnet. To meet this objective, we proposed an innovative unsupervised deep learning model, DynamicDeepFlow (DDF), that can recognize new flow patterns in real-time. The proposed DDF leverages a combination of deep learning model, variational autoencoder (VAE), and shallow learning model, k-means++ to recognize real traffic change patterns. 

## Training the VAE

we pre-trained the deep learning model VAE with the flow change correlation in years 2018 and 2019. The VAE can automatically extract complex features from input data without human intervention. This avoids manual feature extraction and the risk of dropping useful information in the data. The extracted features are then fed into the shallow learning model k-means++. The TrainVAE5.py file is used to train a VAE model for network traffic information extraction. _N_ indicate the number of epoches used in the training process. After training process, you can get three figures about the extracted mean and sigma features. 

```
TrainVAE5.py --n_epochs=N
```
