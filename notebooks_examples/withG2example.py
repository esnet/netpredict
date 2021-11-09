#!/usr/bin/env python
import sys
sys.path.append('../')

import json
import csv
import yaml
import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from mlmodels.statsmodels.multioutput_regression import NP_LinearRegression 

network_yaml="../example_topos/PRP_topo.yaml"
g2_snapshots="../datasets/g2_outputs/esnet_2013_groupbypath.json"


def gen_flow_data_g2():
    g2_snapshot_json=open(g2_snapshots)

    g2_snapshot_data=json.load(g2_snapshot_json)

    #iterating through snapshots has time periods
    nosnapshots=0
    nolinks=0
    linknamelist=[]
    linkflowsdf=[]
    
    nosnapshots=g2_snapshot_data['data']['num_snapshots']
    
    #in the snapshot first parse topology to find number of unique links
    #then parse the snapshot again to match number of flows per link
    num=0
    # loop to read topology only
    for i in g2_snapshot_data['data']['snapshots']:
        nolinks=0 
        for j in i['topo']['topology']['links']:
            print(j['id'])
            linknamelist.append(j['id']) #records the id of the link
            nolinks=nolinks+1

        linkflowsdf = pd.DataFrame(columns = linknamelist)#creates column names with links
        tempDf = pd.DataFrame(index=[0],columns=linknamelist)

        for col in tempDf.columns:
            tempDf[col]=tempDf[col].fillna(0)


        #print("$$")
        #print(tempDf.head())
    #create new loop to loop through all snapshots
    for i in g2_snapshot_data['data']['snapshots']:
        #now add flow numbers per link
        for k in i['flows']['flowgroups']:
            print("reading flow id", k['id'])
            for l in k['links']:
                #print(l['id'])
                for m in linknamelist:
                    #print("####")
                    #print(m)
                    if l['id']==m:
                        #print("match")
                #        print(m)
                        tempDf[m][0]=tempDf[m][0]+1
                        #print(tempDf[m][0])

        linkflowsdf = pd.concat([linkflowsdf,tempDf])


        #linkflowdict['time']=nosnapshots
        #linkflowdict['time']   

    #for j in 
    #close file
    
    
    g2_snapshot_json.close()
    return linkflowsdf,nosnapshots,nolinks



def main():
    #starting demo to mimic G2
    print("##################################")
    print("Welcome to our Demo!!!!!")
    print("This is our PRP topology where we have active data moving")
    #show topology and active animation
    flowsperlinkdf=[]
    nosnapshotsm=0
    nolinksm=0

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
    #lets build of ML models on our historical data
    flowsperlinkdf,nosnapshotsm,nolinksm=gen_flow_data_g2()
    print("Number of snapshots found:",nosnapshotsm)
    print("Number of links found:",nolinksm)
    print(flowsperlinkdf)
    print("Building ML models for Predictions......")

    for col in flowsperlinkdf:
        for v in col:
            NP_LinearRegression(v)

    #call netpredict to build ML models per link,
    #  since the topology is constantly changing
    # we will build ML models per link and just 
    # read in latest data to predict 5 future steps


    #call netpredict for predicting 5 steps ahead

    #netpredict_arima(hist, a,b,c)
    #link1,link2,link3 =call_netpredict_arima(10, send historical (last 10 steps, hist range))

    ##### no pretrained ML model
    #record history and build it over time

    #### pretrained




main()