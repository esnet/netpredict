from functools import partial
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
from model import GraphRNN
from dcrnn import DiffConv
from gaan import GatedGAT
from utils import NormalizationLayer, masked_mae_loss, get_learning_rate
import pandas as pd
from sklearn.preprocessing import StandardScaler
from load_data import *
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
from sensors2graph import *




batch_cnt = [0]

graph_link_ids="../../datasets/snmp_esnet/esnet_links_ids2019-2020.txt"#graph_sensor_id.txt"#" #txt
link_bw_capacity="../../datasets/snmp_esnet/esnet_link_graph2019-2020.csv"#distances_la_2012.csv"#" #csv
link_data="../../datasets/snmp_esnet/snmp_2019_data.hdf5" #h5



LR=0.001
BATCH_SIZE=64
EPOCHS=50
NUM_LAYERS=9
WINDOW_LENGTH=144
SAVE_MODEL="stgcnwavemodel.pt"
PRED_LEN=5
CHANNELS=[1, 16, 32, 64, 32, 128]
control_str='TNTSTNTST'
MIN_LR=2e-6,
MAX_GRAD_NORM=5.0,
DIFFSTEPS=2,
DECAYSTEPS=2000,

def train(model, graph, dataloader, optimizer, scheduler, normalizer, loss_fn, device):
    total_loss = []
    graph = graph.to(device)
    model.train()
    batch_size = BATCH_SIZE
    for i, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        # Padding: Since the diffusion graph is precmputed we need to pad the batch so that
        # each batch have same batch size
        if x.shape[0] != batch_size:
            x_buff = torch.zeros(
                batch_size, x.shape[1], x.shape[2], x.shape[3])
            y_buff = torch.zeros(
                batch_size, x.shape[1], x.shape[2], x.shape[3])
            x_buff[:x.shape[0], :, :, :] = x
            x_buff[x.shape[0]:, :, :,
                   :] = x[-1].repeat(batch_size-x.shape[0], 1, 1, 1)
            y_buff[:x.shape[0], :, :, :] = y
            y_buff[x.shape[0]:, :, :,
                   :] = y[-1].repeat(batch_size-x.shape[0], 1, 1, 1)
            x = x_buff
            y = y_buff
        # Permute the dimension for shaping
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)

        x_norm = normalizer.normalize(x).reshape(
            x.shape[0], -1, x.shape[3]).float().to(device)
        y_norm = normalizer.normalize(y).reshape(
            x.shape[0], -1, x.shape[3]).float().to(device)
        y = y.reshape(y.shape[0], -1, y.shape[3]).float().to(device)

        batch_graph = dgl.batch([graph]*batch_size)
        output = model(batch_graph, x_norm, y_norm, batch_cnt[0], device)
        # Denormalization for loss compute
        y_pred = normalizer.denormalize(output)
        loss = loss_fn(y_pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        if get_learning_rate(optimizer) > MIN_LR:
            scheduler.step()
        total_loss.append(float(loss))
        batch_cnt[0] += 1
        print("Batch: ", i)
    return np.mean(total_loss)


def eval(model, graph, dataloader, normalizer, loss_fn, device):
    total_loss = []
    graph = graph.to(device)
    model.eval()
    batch_size = BATCH_SIZE
    for i, (x, y) in enumerate(dataloader):
        # Padding: Since the diffusion graph is precmputed we need to pad the batch so that
        # each batch have same batch size
        if x.shape[0] != batch_size:
            x_buff = torch.zeros(
                batch_size, x.shape[1], x.shape[2], x.shape[3])
            y_buff = torch.zeros(
                batch_size, x.shape[1], x.shape[2], x.shape[3])
            x_buff[:x.shape[0], :, :, :] = x
            x_buff[x.shape[0]:, :, :,
                   :] = x[-1].repeat(batch_size-x.shape[0], 1, 1, 1)
            y_buff[:x.shape[0], :, :, :] = y
            y_buff[x.shape[0]:, :, :,
                   :] = y[-1].repeat(batch_size-x.shape[0], 1, 1, 1)
            x = x_buff
            y = y_buff
        # Permute the order of dimension
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)

        x_norm = normalizer.normalize(x).reshape(
            x.shape[0], -1, x.shape[3]).float().to(device)
        y_norm = normalizer.normalize(y).reshape(
            x.shape[0], -1, x.shape[3]).float().to(device)
        y = y.reshape(x.shape[0], -1, x.shape[3]).to(device)

        batch_graph = dgl.batch([graph]*batch_size)
        output = model(batch_graph, x_norm, y_norm, i, device)
        y_pred = normalizer.denormalize(output)
        loss = loss_fn(y_pred, y)
        total_loss.append(float(loss))
    return np.mean(total_loss)



def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    #normalizer = NormalizationLayer(train_data.mean, train_data.std)

    #DCRNN code

    batch_g = dgl.batch([G]*BATCH_SIZE).to(device)
    out_gs, in_gs = DiffConv.attach_graph(batch_g, DIFFSTEPS)
    net = partial(DiffConv, k=DIFFSTEPS,in_graph_list=in_gs, out_graph_list=out_gs)


    dcrnn = GraphRNN(in_feats=2,
                     out_feats=64,
                     seq_len=12,
                     num_layers=2,
                     net=net,
                     decay_steps=DECAYSTEPS).to(device)

    optimizer = torch.optim.Adam(dcrnn.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_fn = masked_mae_loss

    for e in range(EPOCHS):
        train_loss = train(dcrnn, g, train_iter, optimizer, scheduler,
                           normalizer, loss_fn, device)
        valid_loss = eval(dcrnn, g, val_iter,
                          normalizer, loss_fn, device)
        test_loss = eval(dcrnn, g, test_iter,
                         normalizer, loss_fn, device)
        print("Epoch: {} Train Loss: {} Valid Loss: {} Test Loss: {}".format(e,
                                                                             train_loss,
                                                                             valid_loss,
                                                                             test_loss))


main()