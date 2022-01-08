# This is an implementation to DCRNN by Yu et al.

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
import networkx as nx
import matplotlib.pyplot as plt

########################################
#data files:
# Create the adjacency matrix as a Graph in DGL:
## graph_sensors_ids.txt (for Id of sensors) === esnet_links_id.txt
## distances_la_so12.csv for distance between sensors  === link_capacity.csv
# actual speed data
# metr_la.h5 shape [x,y], where x shows number of timesteps and y is the number of sensors- GNN will use
# this histroical speed to predict future speed

########################################

graph_link_ids="../../datasets/snmp_esnet/esnet_links_ids.txt" #txt
link_bw_capacity="../../datasets/snmp_esnet/link_capacity.csv" #csv
link_data="" #h5

n_window_size=144
save_path=""
predicted_len=20


"""
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
"""
def main():
    #prepare and load the data set
    with open(graph_link_ids) as f:
        link_ids = f.read().strip().split(',')

    print(link_ids)
    print("Number of nodes:", len(link_ids))


    #MK:change the link details file

    bwcap_df=pd.read_csv(link_bw_capacity, dtype={'from': 'str', 'to': 'str'})
    #print(bwcap_df)
    adj_mx = get_adjacency_matrix(bwcap_df, link_ids)
    sp_mx = sp.coo_matrix(adj_mx)
    G = dgl.from_scipy(sp_mx)
    #print(G)

    #nx.draw(G.to_networkx(), with_labels=True)
    #plt.show()

main()