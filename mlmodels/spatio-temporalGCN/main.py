import dgl
import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from load_data import *
from utils import *
from model import *
from sensors2graph import *
import torch.nn as nn
import argparse
import scipy.sparse as sp

graph_link_ids="" #txt
link_weights="" #csv
link_data="" #h5

n_window_size=144
save_path=""
predicted_len=20



def trainSTGCN():
    device = torch.device("cuda") if torch.cuda.is_available() and not args.disablecuda else torch.device("cpu")

    with open(graph_link_ids) as f:
    link_ids = f.read().strip().split(',')

    weights_df= pd.read_csv(link_weights, dtype={'from': 'str', 'to': 'str'})

    adj_mx = get_adjacency_matrix(weights_df, link_ids)
    sp_mx = sp.coo_matrix(adj_mx)
    G = dgl.from_scipy(sp_mx)

    df = pd.read_hdf(link_data)
    num_samples, num_nodes = df.shape
    tsdata = df.to_numpy()

