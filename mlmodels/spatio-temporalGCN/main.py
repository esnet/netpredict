# This is an implementation to DCRNN by Yu et al.
# original code is here: https://github.com/dmlc/dgl/blob/master/examples/pytorch/stgcn_wave/main.py

#dataset generation:
# The data is located under the datasets/snmp_esnet folder
#for the h5 file, Please run the file featurizers/csv_to_hdf.py. This will generate a time series HDF5 file for this code to run
#
#

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
link_data="../../datasets/snmp_esnet/snmp_2019_data.hdf5" #h5



LR=0.001
BATCH_SIZE=50
EPOCHS=50
NUM_LAYERS=2
WINDOW_LENGTH=144
SAVE_MODEL="stgcnwavemodel.pt"
PRED_LEN=48
CHANNELS=[1, 16, 32, 64, 32, 128]
control_str='TNTSTNTST'
#parser.add_argument('--control_str', type=str, default='TNTSTNTST', help='model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer')
#parser.add_argument('--channels', type=int, nargs='+', default=[1, 16, 32, 64, 32, 128], help='model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer')



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


    device = torch.device("cuda") if torch.cuda.is_available() and not args.disablecuda else torch.device("cpu")


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
    
    #plt.show()


    timedatadf = pd.read_hdf(link_data)
    num_samples, num_nodes = timedatadf.shape
    print("HDF data")
    tsdata = timedatadf.to_numpy()
    #print(tsdata)


    n_his = WINDOW_LENGTH

    save_path = SAVE_MODEL

    n_pred = PRED_LEN
    n_route = num_nodes
    blocks = CHANNELS
    # blocks = [1, 16, 32, 64, 32, 128]
    drop_prob = 0
    num_layers = NUM_LAYERS

    batch_size = BATCH_SIZE
    epochs = EPOCHS
    lr = LR
    
    W = adj_mx
    len_val = round(num_samples * 0.1)
    len_train = round(num_samples * 0.7)
    train = timedatadf[: len_train]
    val = timedatadf[len_train: len_train + len_val]
    test = timedatadf[len_train + len_val:]

    print("SIZE OF DATA")
    print(len(val))
    print(len(test))

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    x_train, y_train = data_transform(train, n_his, n_pred, device)
    x_val, y_val = data_transform(val, n_his, n_pred, device)
    x_test, y_test = data_transform(test, n_his, n_pred, device)

    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    val_iter = torch.utils.data.DataLoader(val_data, batch_size)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size)



    loss = nn.MSELoss()
    G = G.to(device)
    model = STGCN_WAVE(blocks, n_his, n_route, G, drop_prob, num_layers, device, control_str).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    print(model)
    print(model.parameters)
    print(sum(p.numel() for p in model.parameters()))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    min_val_loss = np.inf
    train_losses, valid_losses = [], []

    for epoch in range(1, epochs + 1):
        print("Epoch NO:", epoch)
        l_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            #print("infor")
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = evaluate_model(model, loss, val_iter)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
        train_loss = l_sum / n
        train_losses.append(train_loss)
        valid_losses.append(val_loss)

        print("epoch", epoch, ", train loss:", train_loss, ", validation loss:", val_loss)

    
    best_model = STGCN_WAVE(blocks, n_his, n_route, G, drop_prob, num_layers, device).to(device)
    best_model.load_state_dict(torch.load(save_path))


    l = evaluate_model(best_model, loss, test_iter)
    MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
    
    with open("results.txt","w") as f:
        for i in range (epochs):
            f.write("Epoch {}:\n".format(i+1))
            f.write("\ttrain loss: {}, valid loss: {}\n ".format(train_losses[i], valid_losses[i]))
        f.write("MAE: {}\n".format(MAE))
        f.write("MAPE: {}\n".format(MAPE))
        f.write("RMSE: {}\n".format(RMSE))
        
main()