import argparse
import numpy as np

import pandas as pd
import VAE_4

import torch
from Data import DatasetVAE
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
import matplotlib


def plot_joint(autoencoder, x_set, y_set, num_batches=100):
    font = {'family': 'Times New Roman',
            'weight': 'normal',  # bold
            'size': 18}
    matplotlib.rc('font', **font)
    # vmin, vmax = 0, 7
    # normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
    plt.rcParams['savefig.dpi'] = 200  # Image Pixel
    plt.rcParams['figure.dpi'] = 200  # Resolution ratio
    plt.rcParams['figure.figsize'] = (5.0, 3.0)  # Set figure_size

    df_data = pd.DataFrame(columns=["mu", "sigma", "Category"])
    df_summary = pd.DataFrame(columns=["mu", "sigma", "Category"])

    mu, sigma = autoencoder.encoder(x_set.float())
    mu_cpu = mu.to('cpu').detach().numpy()
    sigma_cpu = sigma.to('cpu').detach().numpy()

    df_data["mu"] = np.mean(mu_cpu, 1) * 10 ** 4
    df_data["sigma"] = abs(np.mean(sigma_cpu, 1)) * 10 ** 4
    df_data["Category"] = y_set
    daylist = ["Night", "Monday_day", "Tuesday_day", "Wednesday_day", "Thursday_day", "Friday_day", "Saturday_day",
               "Sunday_day"]
    for i in range(8):
        df_summary.loc[i] = df_data[df_data["Category"] == i].mean()

    df_data_revised = df_data
    for i in range(8):
        df_data_revised = df_data_revised.replace(i, daylist[int(df_summary.iloc[i][2])])

    # bivariate relational or distribution plot with the marginal distributions of the mu and sigma
    h = sns.jointplot(data=df_data_revised, x="mu", y="sigma", hue="Category", xlim=(-200, 150), ylim=(-10, 130))
    h.ax_joint.set_xlabel("Average $\mu$ $(10^{-4})$")
    h.ax_joint.set_ylabel("Average $\sigma$ $(10^{-4})$")

    # plt.xlim([-10, 80])
    # plt.ylim([-10, 40])
    h.ax_joint.legend_._visible = False
    h.fig.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, borderaxespad=0)
    # plt.tight_layout()
    plt.show()

def plotYearCompare(autoencoder, x_set_2018, y_set_2018, x_set_2019, y_set_2019, x_set_2020, y_set_2020,
                    num_batches=100):
    mu_2018, sigma_2018 = autoencoder.encoder(x_set_2018.float())
    mu_2019, sigma_2019 = autoencoder.encoder(x_set_2019.float())
    mu_2020, sigma_2020 = autoencoder.encoder(x_set_2020.float())

    mu_2018_cpu = mu_2018.to('cpu').detach().numpy() * 10 ** 4
    sigma_2018_cpu = sigma_2018.to('cpu').detach().numpy() * 10 ** 4
    mu_2019_cpu = mu_2019.to('cpu').detach().numpy() * 10 ** 4
    sigma_2019_cpu = sigma_2019.to('cpu').detach().numpy() * 10 ** 4
    mu_2020_cpu = mu_2020.to('cpu').detach().numpy() * 10 ** 4
    sigma_2020_cpu = sigma_2020.to('cpu').detach().numpy() * 10 ** 4

    def combine(res, nums1, nums2, nums3):
        res = np.concatenate((res, nums1), axis=0)
        res = np.concatenate((res, nums2), axis=0)
        res = np.concatenate((res, nums3), axis=0)
        return res

    mu_1, mu_2, sigma1, sigma2 = np.array([]), np.array([]), np.array([]), np.array([])
    mu_1 = combine(mu_1, mu_2018_cpu[:, 0], mu_2019_cpu[:, 0], mu_2020_cpu[:, 0])
    mu_2 = combine(mu_2, mu_2018_cpu[:, 1], mu_2019_cpu[:, 1], mu_2020_cpu[:, 1])
    sigma1 = combine(sigma1, sigma_2018_cpu[:, 0], sigma_2019_cpu[:, 0], sigma_2020_cpu[:, 0])
    sigma2 = combine(sigma2, sigma_2018_cpu[:, 1], sigma_2019_cpu[:, 1], sigma_2020_cpu[:, 1])

    year = ["2018" for _ in range(len(mu_2018_cpu))] + ["2019" for _ in range(len(mu_2019_cpu))] + ["2020" for _ in
                                                                                                    range(len(
                                                                                                        mu_2020_cpu))]

    df_data = pd.DataFrame(columns=["$\mu_1$", "$\mu_2$", "$\sigma_1$", "$\sigma_2$", "Years"])
    df_data["$\mu_1$"], df_data["$\mu_2$"], df_data["$\sigma_1$"], df_data["$\sigma_2$"], df_data[
        "Years"] = mu_1, mu_2, sigma1, sigma2, year

    g = sns.pairplot(df_data, hue="Years", palette="Set2", diag_kind="kde", height=2.5)
    labels = ["$\mu_1$ $(10^{-4})$", "$\mu_2$ $(10^{-4})$", "$\sigma_1$ $(10^{-4})$", "$\sigma_2$ $(10^{-4})$"]

    for i in range(4):
        g.axes[i, 0].yaxis.set_label_text(labels[i])

    for j in range(4):
        g.axes[3, j].xaxis.set_label_text(labels[j])

    plt.show()

def plot_normaldistribution(autoencoder, x_set_2018, y_set_2018, x_set_2019, y_set_2019, x_set_2020, y_set_2020, num_batches=100):
  font = {'family' : 'Times New Roman',
      'weight' : 'normal', # bold
      'size'   : 18}
  matplotlib.rc('font', **font)
  # vmin, vmax = 0, 7
  # normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
  plt.rcParams['savefig.dpi'] = 150 # Image Pixel
  plt.rcParams['figure.dpi'] = 150 # Resolution ratio
  plt.rcParams['figure.figsize'] = (5.0, 3.0) # Set figure_size

  mu_2018, sigma_2018 = autoencoder.encoder(x_set_2018.float())
  mu_2019, sigma_2019 = autoencoder.encoder(x_set_2019.float())
  mu_2020, sigma_2020 = autoencoder.encoder(x_set_2020.float())

  mu_2018_cpu = mu_2018.to('cpu').detach().numpy()
  sigma_2018_cpu = sigma_2018.to('cpu').detach().numpy()
  mu_2019_cpu = mu_2019.to('cpu').detach().numpy()
  sigma_2019_cpu = sigma_2019.to('cpu').detach().numpy()
  mu_2020_cpu = mu_2020.to('cpu').detach().numpy()
  sigma_2020_cpu = sigma_2020.to('cpu').detach().numpy()

  df_data = pd.DataFrame(columns=['mu', 'sigma', "Category", "Year"])
  df_summary = pd.DataFrame(columns=['mu', 'sigma', "Category","Year"])
  df_XY = pd.DataFrame(columns=["value", "Years"])

  df_data["mu"] = np.concatenate((np.mean(mu_2018_cpu, 1), np.mean(mu_2019_cpu, 1), np.mean(mu_2020_cpu, 1)), axis=0)
  df_data["sigma"] = np.concatenate( (abs(np.mean(sigma_2018_cpu, 1)), abs(np.mean(sigma_2019_cpu, 1)), abs(np.mean(sigma_2020_cpu, 1)) ), axis=0)
  df_data["Category"] = np.concatenate((y_set_2018, y_set_2019, y_set_2020), axis=0)
  df_data["Year"] = ["2018" for _ in range(len(y_set_2018)) ] + ["2019" for _ in range(len(y_set_2019)) ] + ["2020" for _ in range(len(y_set_2020)) ]

  def plotWeekdayWeekend( day):
    df_summary = pd.DataFrame(columns=['mu', 'sigma', "Category", "Year"])
    df_summary.loc[0] = df_data[ (df_data['Year'] == "2018") & (df_data['Category'] == day) ].mean()
    df_summary.loc[1] = df_data[(df_data['Year'] == "2019") & (df_data['Category'] == day)].mean()
    df_summary.loc[2] = df_data[(df_data['Year'] == "2020") & (df_data['Category'] == day)].mean()
    df_summary["Year"] = np.array(("2018", "2019", "2020"))

    df_XY = pd.DataFrame(columns=["value", "Years", "Category"])
    values, labels, categories = np.array([]), np.array([]), np.array([])
    for i in range(3):
        normal_x = np.random.normal(loc=df_summary.iloc[i][0], scale=df_summary.iloc[i][1], size=100)*10**4
        values = np.concatenate((values, normal_x), axis=0)
        label = [df_summary.iloc[i][3] for _ in range(100)]
        labels = np.concatenate((labels, label), axis=0)
        category = [df_summary.iloc[i][2] for _ in range(100)]
        categories = np.concatenate((categories, category), axis=0)

    df_XY["value"] = values
    df_XY["Years"] = labels
    df_XY["Category"] = categories

    return df_XY

  for day in range(8):
    temp = plotWeekdayWeekend(day)
    sns.displot(temp, x="value", hue="Years", palette=['#2ca02c', '#ff7f0e', '#1f77b4'], alpha=0.2, kind="kde", fill=True)

    plt.ylabel("pdf")
    plt.xlabel("Extracted features $(10^{-4})$")

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

    # Visualizing the learned features (e.g., mu1, mu2, sigma1, and sigma2) by convolutional VAE for 2018, 2019, 2020
    plot_joint(net_trained, dataset_2018.X, dataset_2018.y_set)
    plot_joint(net_trained, dataset_2019.X, dataset_2019.y_set)
    plot_joint(net_trained, dataset_2020.X, dataset_2020.y_set)

    # Visualizing the relationship between mu1, mu2, sigma1, and sigma2 for 2018, 2019, 2020
    plotYearCompare(net_trained, dataset_2018.X, dataset_2018.y_set, dataset_2019.X, dataset_2019.y_set, dataset_2020.X, dataset_2020.y_set)

    # Visualizing the probability density function (pdf) of mu1, mu2, sigma1, and sigma2 for 2018, 2019, 2020
    plot_normaldistribution(net_trained, dataset_2018.X, dataset_2018.y_set, dataset_2019.X, dataset_2019.y_set, dataset_2020.X, dataset_2020.y_set)