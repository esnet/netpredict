#!/usr/bin/env python

import json
import csv
import yaml
import networkx as nx
import matplotlib.pyplot as plt

network_yaml="../example_topos/PRP_topo.yaml"

def main():
    #starting demo to mimic G2
    print("##################################")
    print("Welcome to our Demo!!!!!")
    print("This is our PRP topology where we have active data moving")
    #show topology and active animation
    NetworkDict={}
    with open(network_yaml, "r") as stream:
        try:
            NetworkDict=yaml.safe_load(stream)
            print(NetworkDict)
        except yaml.YAMLError as exc:
            print(exc)
    print(NetworkDict['links'])
    
    g = nx.DiGraph()
    for i in NetworkDict['links']:
        src=i['src']
        dst=i['dst']
        g.add_edge(src, dst, weight=1)

    pos = nx.circular_layout(g)
  
    nx.draw_networkx_nodes(g,pos,node_size=500)
    plt.title("PRP topology")
    plt.axis('off')
    plt.show()

    print("G2 started......")

    print("G2 collects 1 min data......")

    print("G2 calls netpredict to predict future flows....")

    ##### no pretrained ML model
    #record history and build it over time





main()