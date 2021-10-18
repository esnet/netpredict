
import argparse
import numpy as np

import pandas as pd
import VAE_4

import torch
from Data import DatasetVAE
import matplotlib.pyplot as plt
plt.ion()
import matplotlib

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

def plotElbow(autoencoder, x_set_2018, y_set_2018, x_set_2019, y_set_2019, x_set_2020, y_set_2020):
    _, mu_2018, sigma_2018, latent_2018 = autoencoder(x_set_2018.float())
    _, mu_2019, sigma_2019, latent_2019 = autoencoder(x_set_2019.float())
    _, mu_2020, sigma_2020, latent_2020 = autoencoder(x_set_2020.float())

    mu_2018_cpu = mu_2018.to('cpu').detach().numpy()
    sigma_2018_cpu = sigma_2018.to('cpu').detach().numpy()
    muSigma_2018 = np.concatenate([mu_2018_cpu, sigma_2018_cpu], axis=1)
    latent_2018_cpu = latent_2018.to('cpu').detach().numpy().reshape(len(mu_2018), -1)

    mu_2019_cpu = mu_2019.to('cpu').detach().numpy()
    sigma_2019_cpu = sigma_2019.to('cpu').detach().numpy()
    latent_2019_cpu = latent_2019.to('cpu').detach().numpy().reshape(len(mu_2019), -1)
    muSigma_2019 = np.concatenate([mu_2019_cpu, sigma_2019_cpu], axis=1)

    mu_2020_cpu = mu_2020.to('cpu').detach().numpy()
    sigma_2020_cpu = sigma_2020.to('cpu').detach().numpy()
    latent_2020_cpu = latent_2020.to('cpu').detach().numpy().reshape(len(mu_2020), -1)
    muSigma_2020 = np.concatenate([mu_2020_cpu, sigma_2020_cpu], axis=1)

    n_runs = 5
    n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    inertia_2018 = np.zeros((len(n_clusters), n_runs))
    inertia_2019 = np.zeros((len(n_clusters), n_runs))
    inertia_2020 = np.zeros((len(n_clusters), n_runs))

    for run in range(n_runs):

        for idx, cluster in enumerate(n_clusters):
            km_2018 = KMeans(n_clusters=cluster).fit(muSigma_2018)
            km_2019 = KMeans(n_clusters=cluster).fit(muSigma_2019)
            km_2020 = KMeans(n_clusters=cluster).fit(muSigma_2020)

            inertia_2018[idx][run] = km_2018.inertia_
            inertia_2019[idx][run] = km_2019.inertia_
            inertia_2020[idx][run] = km_2020.inertia_

    fig1 = plt.figure()
    font = {'family': 'Times New Roman',
            'weight': 'normal',  # bold
            'size': 10}
    matplotlib.rc('font', **font)

    plt.rcParams['savefig.dpi'] = 200  # Image Pixel
    plt.rcParams['figure.dpi'] = 200  # Resolution ratio
    plt.rcParams['figure.figsize'] = (5.0, 3.0)  # Set figure_size

    plt.errorbar(n_clusters, inertia_2018.mean(axis=1), inertia_2018.std(axis=1), label='2018')
    plt.errorbar(n_clusters, inertia_2019.mean(axis=1), inertia_2019.std(axis=1), label='2019')
    plt.errorbar(n_clusters, inertia_2020.mean(axis=1), inertia_2020.std(axis=1), label='2020')

    plt.legend(loc='best')
    plt.xlabel('Number of cluster')
    plt.ylabel('SEE')  # Sum of squared distances of samples to their closest cluster center

    # this is another inset axes over the main axes
    ax2 = plt.axes([0.4, 0.45, .25, .25])

    plt.errorbar(n_clusters[2:6], inertia_2018[2:6].mean(axis=1), inertia_2018[2:6].std(axis=1), label='2018')
    plt.errorbar(n_clusters[2:6], inertia_2019[2:6].mean(axis=1), inertia_2019[2:6].std(axis=1), label='2019')
    plt.errorbar(n_clusters[2:6], inertia_2020[2:6].mean(axis=1), inertia_2020[2:6].std(axis=1), label='2020')

    ax2.set_xticks(np.arange(3, 7))
    ax2.set_xticklabels(('3', '4', '5', '6'))
    plt.xlabel('Number of cluster')
    plt.ylabel('SSE')  # Sum of squared distances of samples to their closest cluster center

    plt.show(block=True)

def kmeans(autoencoder, x_set_2018, y_set_2018, x_set_2019, y_set_2019, x_set_2020, y_set_2020):

    # Extract learned features (mu and sigma) from the trained autoencoder
    _, mu_2018, sigma_2018, latent_2018 = autoencoder(x_set_2018.float())
    _, mu_2019, sigma_2019, latent_2019 = autoencoder(x_set_2019.float())
    _, mu_2020, sigma_2020, latent_2020 = autoencoder(x_set_2020.float())

    # Load the data to cpu or GPU. You may change ".to('cpu')" to .to(device) if you are using GPU.
    mu_2018_cpu = mu_2018.to('cpu').detach().numpy()
    sigma_2018_cpu = sigma_2018.to('cpu').detach().numpy()
    muSigma_2018 = np.concatenate([mu_2018_cpu, sigma_2018_cpu], axis=1)
    latent_2018_cpu = latent_2018.to('cpu').detach().numpy().reshape(len(mu_2018),-1)

    mu_2019_cpu = mu_2019.to('cpu').detach().numpy()
    sigma_2019_cpu = sigma_2019.to('cpu').detach().numpy()
    latent_2019_cpu = latent_2019.to('cpu').detach().numpy().reshape(len(mu_2019),-1)
    muSigma_2019 = np.concatenate([mu_2019_cpu, sigma_2019_cpu], axis=1)

    mu_2020_cpu = mu_2020.to('cpu').detach().numpy()
    sigma_2020_cpu = sigma_2020.to('cpu').detach().numpy()
    latent_2020_cpu = latent_2020.to('cpu').detach().numpy().reshape(len(mu_2020),-1)
    muSigma_2020 = np.concatenate([mu_2020_cpu, sigma_2020_cpu], axis=1)

    n_runs = 1
    n_clusters = [5]
    inertia_2018 = np.zeros((len(n_clusters), n_runs))
    inertia_2019 = np.zeros((len(n_clusters), n_runs))
    inertia_2020 = np.zeros((len(n_clusters), n_runs))

    # Kmeans++ training process for data 2018, 2019, and 2020
    for run in range(n_runs):

        for idx, cluster in enumerate(n_clusters):
            km_2018 = KMeans(n_clusters=cluster).fit(muSigma_2018)
            km_2019 = KMeans(n_clusters=cluster).fit(muSigma_2019)
            km_2020 = KMeans(n_clusters=cluster).fit(muSigma_2020)

            inertia_2018[idx][run] = km_2018.inertia_
            inertia_2019[idx][run] = km_2019.inertia_
            inertia_2020[idx][run] = km_2020.inertia_

    font = {'family' : 'Times New Roman',
        'weight' : 'normal', # bold
        'size'   : 10}
    matplotlib.rc('font', **font)
    plt.rcParams['savefig.dpi'] = 200 # Image Pixel
    plt.rcParams['figure.dpi'] = 200 # Resolution ratio
    plt.rcParams['figure.figsize'] = (5.0, 3.0) # Set figure_size

    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    n_clusters = 5
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # We want to have the same colors for the same cluster from the MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per closest one.
    means_cluster_centers_2018 = km_2018.cluster_centers_
    order = pairwise_distances_argmin(km_2018.cluster_centers_, km_2019.cluster_centers_)
    means_cluster_centers_2019 = km_2019.cluster_centers_[order]
    order = pairwise_distances_argmin(km_2018.cluster_centers_, km_2020.cluster_centers_)
    means_cluster_centers_2020 = km_2020.cluster_centers_[order]

    means_labels_2018 = pairwise_distances_argmin(muSigma_2018, means_cluster_centers_2018)
    means_labels_2019 = pairwise_distances_argmin(muSigma_2019, means_cluster_centers_2019)
    means_labels_2020 = pairwise_distances_argmin(muSigma_2020, means_cluster_centers_2020)

    clusters = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']
    # KMeans_2018
    count = 0
    ax1 = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(n_clusters), colors):
        my_members = means_labels_2018 == k
        cluster_center = means_cluster_centers_2018[k]
        ax1.plot(muSigma_2018[my_members, 0], muSigma_2018[my_members, 1], "w", markerfacecolor=col, marker='o', markersize=6, label=clusters[count])
        ax1.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=8)
        count += 1

    ax1.legend(loc= 'best')
    ax1.set_title('2018')
    ax1.set_xlim([0.01, 0.083])
    ax1.set_ylim([0.03, 0.11])
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')

    # KMeans_2019
    count = 0
    ax2 = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(n_clusters), colors):
        my_members = means_labels_2019 == k
        cluster_center = means_cluster_centers_2019[k]
        ax2.plot(muSigma_2019[my_members, 0], muSigma_2019[my_members, 1], "w", markerfacecolor=col, marker='o', markersize=6, label=clusters[count])
        ax2.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=8)
        count += 1

    ax2.legend(loc= 'best')
    ax2.set_title('2019')
    # ax.set_xticks(())
    ax2.set_yticks(())
    ax2.set_xlim([0.01, 0.083])
    ax2.set_ylim([0.03, 0.11])
    ax2.set_xlabel('Dimension 1')

    # KMeans_2020
    df_2020 = pd.DataFrame(columns=['index', 'x', "y"])
    count = 0
    ax3 = fig.add_subplot(1, 3, 3)
    for k, col in zip(range(n_clusters), colors):
        my_members = means_labels_2020 == k
        cluster_center = means_cluster_centers_2020[k]
        print( "Number of samples in the cluster {}: {}".format (count+1, len(muSigma_2020[my_members, 0])) )
        if len(muSigma_2020[my_members, 0]) != 0:
          ax3.plot(muSigma_2020[my_members, 0], muSigma_2020[my_members, 1], "w", markerfacecolor=col, marker='o',
                  markersize=6, label=clusters[count])
          ax3.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=8)
        count += 1

    df_2020['index'] = np.array(range(len(mu_2020)))[my_members]
    df_2020['x'] = muSigma_2020[my_members, 0]
    df_2020['y'] = muSigma_2020[my_members, 1]

    ax3.legend(loc='best')
    ax3.set_title('2020')
    ax3.set_yticks(())
    ax3.set_xlim([0.01, 0.083])
    ax3.set_ylim([0.03, 0.11])
    ax3.set_xlabel('Dimension 1')
    
    plt.show(block=True)
    
    return df_2020

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('-e', '--n_epochs', type=int, default=1, help='number of episodes')
    p.add_argument('--force_cpu', action='store_true', help='Forces CPU usage')
    args = p.parse_args()
    device = 'cpu' if args.force_cpu else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained VAE model
    model = VAE_4.VariationalAutoencoder() #, args.net_config
    print(model)
    model.load_state_dict(torch.load("VAE_model.pt"))
    net_trained = model

    # Load data
    dataset_2018 = DatasetVAE(csv_file = ["featurizers/2018/inputAE.csv"],
                             csv_label_file = ["featurizers/2018/AELabel.csv"],
                             root_dir = ["featurizers/2018"],
                             device = device)

    dataset_2019 = DatasetVAE(csv_file = ["featurizers/2019/inputAE.csv"],
                        csv_label_file = ["featurizers/2019/AELabel.csv"],
                        root_dir = ["featurizers/2019"],
                        device = device)

    dataset_2020 = DatasetVAE(csv_file = ["featurizers/2020/inputAE.csv"],
                        csv_label_file = ["featurizers/2020/AELabel.csv"],
                        root_dir = ["featurizers/2020"],
                        device = device)

    # # Elbow plot
    plotElbow(net_trained, dataset_2018.X, dataset_2018.y_set, dataset_2019.X, dataset_2019.y_set, dataset_2020.X, dataset_2020.y_set)

    # Visualize results by K-means
    data_2020 = kmeans(net_trained, dataset_2018.X, dataset_2018.y_set, dataset_2019.X, dataset_2019.y_set,
                       dataset_2020.X, dataset_2020.y_set)
