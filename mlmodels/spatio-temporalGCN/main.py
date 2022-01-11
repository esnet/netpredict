# This is an implementation to DCRNN by Yu et al.
# original code is here: https://github.com/dmlc/dgl/blob/master/examples/pytorch/stgcn_wave/main.py

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

graph_link_ids="../../datasets/snmp_esnet/esnet_links_ids2019-2020.txt"#graph_sensor_id.txt"#" #txt
link_bw_capacity="../../datasets/snmp_esnet/esnet_link_graph2019-2020.csv"#distances_la_2012.csv"#" #csv
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
    i=0
    #G=nx.Graph()

    with open(graph_link_ids) as f:
        link_ids = f.read().strip().split(',')
    print(link_ids)
    print("Number of nodes:", len(link_ids))
    #G.add_nodes_from(link_ids)

    #MK:change the link details file
    
    
    bwcap_df=pd.read_csv(link_bw_capacity, dtype={'from': 'str', 'to': 'str'})
    #print(bwcap_df['from'][3])
    print(len(bwcap_df))
    #sz=len(bwcap_df)
    adj_mx = get_adjacency_matrix(bwcap_df, link_ids)
    print(adj_mx)
    sp_mx = sp.coo_matrix(adj_mx)
    G = dgl.from_scipy(sp_mx)
    """
    for i in bwcap_df.index:
        #print(bwcap_df['to'][i])
        #print(sz[2])
        G.add_edge(bwcap_df['from'][i],bwcap_df['to'][i])
    mapping={0:"SACR",1:"SUNN",2:"NEWY",3:"JGI",4:"PANTEX",
    5:"BOIS",6:"CHIC",7:"WASH",8:"LLNL",9:"LSVN",10:"KANS",
    11:"STAR",12:"DENV",13:"AMST",14:"CERN-513",16:"GA",17:"EQX-ASH",
    18:"NETL-MGN",19:"NASH",20:"NETL-PGH",
    21:"ALBQ",22:"SNLA",23:"SLAC",24:"ELPA",25:"ATLA",26:"EQX-CHI",27:"NERSC",
    28:"BOST",29:"ORNL",30:"AOFA",31:"LOND",32:"HOUS",33:"ANL",34:"CERN-773",35:"SNLL",36:"SRS",37:"PNWG",38:"FNAL"}
    """
    print(G)
    #for e in G.edges:
    #    print(e)
    
    #G = nx.relabel_nodes(G, mapping) 
    # this represents the adjacency matrix not the actual graph!
    nx.draw_circular(G.to_networkx(), with_labels=True)
    #nx.draw(G, with_labels=True)
    
    plt.show()

main()