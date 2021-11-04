#!/usr/bin/env python

import json
import csv
import yaml
import networkx as nx
import matplotlib.pyplot as plt
import time

network_yaml="../example_topos/PRP_topo.yaml"
g2_snapshots="../datasets/g2_outputs/esnet_2013_groupbypath.json"


def gen_flow_data_g2():
    g2_snapshot_json=open(g2_snapshots)

    g2_snapshot_data=json.load(g2_snapshot_json)

    #iterating through snapshots has time periods
    nosnapshots=0
    nolinks=0
    linkflowsdict={}

    for i in g2_snapshot_data['data']['snapshots']:
        nosnapshots=nosnapshots+1
        nolinks=0
        for j in g2_snapshot_data['data']['snapshots'][i]['topo']['topology']['links']:
            nolinks=nolinks+1
        
        linkflowdict['time']=nosnapshots
        linkflowdict['time']   

    for j in 
    #close file
    print("Number of snapshots found:",nosnapshots)
    g2_snapshot_json.close()



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
    #plt.show()


    print("G2 started......")

    print("G2 starts collecting 1 min interval data......")
    #time.sleep(2)
    print("G2 calls netpredict to predict future flows....")
    
    #time.sleep(2)
    gen_flow_data_g2()


    #netpredict_arima(hist, a,b,c)
    #link1,link2,link3 =call_netpredict_arima(10, send historical (last 10 steps, hist range))

    ##### no pretrained ML model
    #record history and build it over time

    #### pretrained




main()