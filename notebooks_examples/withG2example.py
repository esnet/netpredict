#!/usr/bin/env python
import sys
sys.path.append('../')
import os
import glob
import json
import csv
import yaml
import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from mlmodels.statsmodels.multioutput_regression import NP_LinearRegression 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


network_yaml="../example_topos/PRP_topo.yaml"
g2_snapshots_dir="../datasets/g2_outputs/sc21_data/"


def gen_flow_data_g2():

    #load all snapshots to build a dataframe
    data_list = []
    nosnapshots=0
    nolinks=0
    linknamelist=[]
    linkflowsdf=[]

    list_of_files = filter(os.path.isfile,glob.glob(g2_snapshots_dir + '*') )
    list_of_files = sorted(list_of_files,key = os.path.getmtime)

    for file in list_of_files: #os.listdir(g2_snapshots_dir):

    #If file is a json, construct it's full path and open it, append all json data to list
        if 'json' in file:
            json_path = file #os.path.join(g2_snapshots_dir, file)
            #print(json_path)
            json_data_df = pd.read_json(json_path)
            print("Reading file....")
            print(json_path)
            nosnapshots=nosnapshots+1

            #print(json_data_df.head())

            #g2_snapshot_json=open(json_data)
            #json_object=json.loads(json_data_df)
            print("num of snapshots: ", json_data_df['data'][0])#g2_snapshot_data=json.loads(json_data)
            #for obj in json_data_df['data']:
            #    print(obj)    
            #print("###")
            #data_list.append(json_data)
    
            #print(data_list)
            #iterating through snapshots has time periods
            #g2_snapshot_data['data']['num_snapshots']
    
            #in the snapshot first parse topology to find number of unique links
            #then parse the snapshot again to match number of flows per link
            num=0
            # loop to read topology only
            for obj in json_data_df['data']['snapshots']:
                #print("OBJ")
                #print(obj['topo'])
                for i in obj['topo']['topology']['links']:
                    nolinks=0 
                    #print(i['id'])
                        #if linkname not in list add it
                        #if len(linknamelist)<=0:
                         #   linknamelist.append(j['id'])
                    if not i['id'] in linknamelist:
                            #print("found")
                        #else:
                        linknamelist.append(i['id']) 
                            #print(ln)
                                 #records the id of the link
                        #nolinks=nolinks+1
                print("links found in range ", len(linknamelist))
          
        
        
    linkflowsdf = pd.DataFrame(columns = linknamelist)#creates column names with links
    
    tempDf = pd.DataFrame(index=[0],columns=linknamelist)

    for col in tempDf.columns:
        tempDf[col]=tempDf[col].fillna(0)

    print(tempDf)

    print("filling values in the DF")

    #create new loop to loop through all snapshots
    for file in list_of_files: #for file in os.listdir(g2_snapshots_dir):

        #If file is a json, construct it's full path and open it, append all json data to list
        if 'json' in file:
            json_path = file #os.path.join(g2_snapshots_dir, file)
            #print(json_path)
            json_data_df = pd.read_json(json_path)
            for obj in json_data_df['data']['snapshots']:
                for i in obj['flows']['flowgroups']:
                    print("here")
                    #print("reading flow id", k['id'])
                    for l in i['links']:
                        print(l['id'])
                        colname=l['id']
                        if not colname in linkflowsdf:
                            print("not found link...")
                            #linkflowsdf.append(colname) 
                        else:
                            
                            tempDf[colname]=tempDf[colname]+1
                            print(tempDf[colname])

        linkflowsdf = pd.concat([linkflowsdf,tempDf])


        #linkflowdict['time']=nosnapshots
        #linkflowdict['time']   

    #for j in 
    #close file
    print("database formed....")
    print(linkflowsdf)
   
   
    #g2_snapshot_json.close()
    
    return linkflowsdf



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
    #flowsperlinkdf,nosnapshotsm,nolinksm=
    flowsperlinkdf=gen_flow_data_g2()
    #print("Number of snapshots found:",nosnapshotsm)
    #print("Number of links found:",nolinksm)
    #print(flowsperlinkdf)
    print("Building ML models for Predictions......")

    
    #for col in flowsperlinkdf:
    NP_LinearRegression(flowsperlinkdf)

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
