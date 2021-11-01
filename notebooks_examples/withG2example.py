#!/usr/bin/env python

import json
import csv
import yaml
import networkx as nx
import matplotlib.pyplot as plt
import time

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

    for i in NetworkDict['regions']:
        g.add_node(i['name'])


    for i in NetworkDict['links']:
        src=i['src']
        dst=i['dst']
        g.add_edge(src, dst, weight=2)

    pos = nx.circular_layout(g)
  
    nx.draw(g,pos,node_size=500, with_labels='True')
    plt.title("PRP topology")
    plt.axis('off')
    plt.show()


    print("G2 started......")

    print("G2 starts collecting 1 min interval data......")
    time.sleep(2)
    print("G2 calls netpredict to predict future flows....")
    time.sleep(2)
    link1,link2,link3 = call_netpredict_arima(10, send historical (last 10 steps, hist range))

    ##### no pretrained ML model
    #record history and build it over time

    #### pretrained




main()