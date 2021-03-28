#calculate three shortest paths and produce 24 values for bar graph
# 
from numpy import array
import pandas as pd
from datetime import datetime
import os
import sys
import re
import time
import json
import numpy as np

import networkx as nx
from flask import jsonify


import logging

from google.cloud import storage
from google.cloud import firestore

#global values:
pos = {}

# UPDATE these to take input from GUI
#src="SUNN"
#dest="CHIC"
#filesize=10

def json_to_dict(filename, graph_feature):
    json_data = open(filename)
    data = json.load(json_data)
    #get the list of dicts
    dicts = data['data']['mapTopology'][graph_feature]
    return dicts
 
def build_graph(nodes, edges):
    G = nx.Graph()
    for node in nodes:
        site = node['name']
        position = (node['x'], node['y'])
        G.add_node(site, pos=position)
    for edge in edges:
        node1 = edge['ends'][0]['name']
        node2 = edge['ends'][1]['name']
        G.add_edge(node1, node2)
    return G

def fill_pos(nodes):
    for node in nodes:
        site = node['name']
        position = (node['x'], node['y'])
        pos[site] = position




def google_map(src,dest):
    #read in map topology
    #build ES.net map

    client = storage.Client()
    globalbucket = client.get_bucket('mousetrap_global_files')
    nodesblob=globalbucket.get_blob('esnet_nodes.json')
    edgesblob=globalbucket.get_blob('esnet_edges.json')

    nodesblobstr=nodesblob.download_as_string()
    edgesblobstr=edgesblob.download_as_string()

    node_dicts = json.loads(nodesblobstr)
    edge_dicts = json.loads(edgesblobstr)
    #print(edge_dicts)

    G = build_graph(node_dicts['data']['mapTopology']['nodes'], edge_dicts['data']['mapTopology']['edges'])
   
    
    #get src and dest
    paths=nx.shortest_simple_paths(G,source=src, target=dest)
    ct=0
    
    print("Printing Size")
    
    #find the shortest path
    print("shortest paths are")
    paths_list=list(paths)

    print(len(paths_list))
    
    short_five_paths=[]
    
    if len(paths_list)>5:
        short_five_paths.append(paths_list[0])
        short_five_paths.append(paths_list[1])
        short_five_paths.append(paths_list[2])
        short_five_paths.append(paths_list[3])
        short_five_paths.append(paths_list[4])
    else:
        for l in paths_list:
            short_five_paths.append(l)
    
    
    return short_five_paths


def build_edge_map():
    client = storage.Client()
    globalbucket = client.get_bucket('mousetrap_global_files')
    edgesblob=globalbucket.get_blob('esnet_edges.json')

    edgesblobstr=edgesblob.download_as_string()

    edge_dicts = json.loads(edgesblobstr)
   
    edges=edge_dicts['data']['mapTopology']['edges']
    
    return edges

class edge_predictions:
    def __init__(self, name, src,dest):
        self.name=name
        self.src=src
        self.dest=dest
        self.timestamp=[]
        self.mean=[]
        self.conf=[]

    def add_timestamp(self,timestamp):
        self.timestamp.append(timestamp)

    def add_mean(self,mean):
        self.mean.append(mean)

    def add_conf(self,conf):
        self.conf.append(conf)

    
def get_latest_data(src,dest):

    edges=build_edge_map()

    db = firestore.Client()
    #query on Date with descending (limit 1)

    users_ref = db.collection(u'latest-predictions')
    
    
    latest_time=0
    edgeCalendar=[]
    for edge in edges:
        if receivedDict['src']==edge['ends'][0]['name'] and receivedDict['dest']==edge['ends'][1]['name']:
            #print("Found EDGE")
            strname=edge['ends'][0]['name']+'--'+edge['ends'][1]['name']
            #print(strname)
            #print(u'{} => {}'.format(doc.id, doc.to_dict()))
            flag=0
            for j in edgeCalendar:
                if j.src==edge['ends'][0]['name'] and j.dest==edge['ends'][1]['name']:
                    j.timestamps.append(receivedDict['timestamp'])
                    j.values.append(receivedDict['traffic'])
                    flag=1
            if flag==0:
                edgeCalendar.append(edge_totals(strname,edge['ends'][0]['name'],edge['ends'][1]['name'],receivedDict['timestamp'],receivedDict['traffic']))
                
    return edgeCalendar


def get_latest_blob():

    storage_client=storage.Client()
    pred_bucket='latest24predictions-esnet'

    latest_time=0
    latest_blob='temp'

    blobs=storage_client.list_blobs(pred_bucket)
    for blob in blobs:
        print(blob.name)
        ts=blob.updated
        print(blob.updated)
        print(ts.timestamp())
        if ts.timestamp()>latest_time:
            latest_time=ts.timestamp()
            latest_blob=blob.name
    
    print("latest is")
    print(latest_blob)

    bucket=storage_client.get_bucket(pred_bucket)
    blob=bucket.get_blob(latest_blob)
    json_file= str(blob.download_as_string(),'utf-8')
    #json_file is string
    json_file=json_file.replace("'",'"')
    print("convert to json")
    json_file=json.loads(json_file)#,ensure_ascii=False).encode('utf8'))

    print("_________")
    
    return json_file


lblwashjson={
	'predictions': [{
			'hour': 0,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 142295066699165.34,
				'weights': [97.57266349306265, 27.442342080656005, 36.135103601013626, 4.503150666952597],
				'bottleneck': 97.57266349306265
			}]
		}, {
			'hour': 1,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'KANS', 'CHIC', 'WASH'],
				'predictedtotal': 1944254163681190.8,
				'weights': [2231.0166450576207, 27.574063040894497, 4.819033684522917, 0.0],
				'bottleneck': 2231.0166450576207
			}]
		}, {
			'hour': 2,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'KANS', 'CHIC', 'WASH'],
				'predictedtotal': 3044254163681190.8,
				'weights': [62720.780693226865, 27.663503973413256, 4.821662419453886, 0.0],
				'bottleneck': 62720.780693226865
			}]
		}, {
			'hour': 3,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'KANS', 'CHIC', 'WASH'],
				'predictedtotal': 49260019714155.09,
				'weights': [1775944.2846117346, 27.724619317598886, 4.824280509593026, 0.0],
				'bottleneck': 1775944.2846117346
			}]
		}, {
			'hour': 4,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'KANS', 'CHIC', 'WASH'],
				'predictedtotal': 59260019714155.09,
				'weights': [50297047.76358942, 27.76081124634497, 4.827073792516128, 0.0],
				'bottleneck': 50297047.76358942
			}]
		}, {
			'hour': 5,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'KANS', 'CHIC', 'WASH'],
				'predictedtotal': 1944254163681190.8,
				'weights': [1424487075.6810007, 27.768636189854828, 4.830025855336276, 0.0],
				'bottleneck': 1424487075.6810007
			}]
		}, {
			'hour': 6,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'CHIC', 'WASH'],
				'predictedtotal': 574254163681190.8,
				'weights': [40343596843.57888, 27.76665344708819, 1.7405348855005378, 0.0],
				'bottleneck': 40343596843.57888
			}]
		}, {
			'hour': 7,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'CHIC', 'WASH'],
				'predictedtotal': 704254163681190.8,
				'weights': [1142590791278.0337, 27.75812789901452, 0.0, 0.0],
				'bottleneck': 1142590791278.0337
			}]
		}, {
			'hour': 8,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'CHIC', 'WASH'],
				'predictedtotal': 1244254163681190.8,
				'weights': [32359874146992.547, 27.989632787817026, 0.0, 0.0],
				'bottleneck': 32359874146992.547
			}]
		}, {
			'hour': 9,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'CHIC', 'WASH'],
				'predictedtotal': 1944254163681190.86,
				'weights': [916479865587315.4, 28.20238053152802, 0.0, 0.0],
				'bottleneck': 916479865587315.4
			}]
		}, {
			'hour': 10,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'CHIC', 'WASH'],
				'predictedtotal': 28944254163681190.8,
				'weights': [2.595607573168552e+16, 28.965075078754364, 0.0, 0.0],
				'bottleneck': 2.595607573168552e+16
			}]
		}, {
			'hour': 11,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 324254163681190.8,
				'weights': [7.351147501284826e+17, 29.223645601129032, 0.0, 5.077793373306702],
				'bottleneck': 7.351147501284826e+17
			}]
		}, {
			'hour': 12,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 3244254163681190.8,
				'weights': [2.0819545352026526e+19, 29.414960001481017, 0.0, 5.143357807720559],
				'bottleneck': 2.0819545352026526e+19
			}]
		}, {
			'hour': 13,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 324254163681190.82,
				'weights': [5.896405541982809e+20, 29.546557908278906, 0.0, 5.185119619530568],
				'bottleneck': 5.896405541982809e+20
			}]
		}, {
			'hour': 14,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 1944254163681190.8,
				'weights': [1.6699499305896891e+22, 29.635936047756847, 0.0, 5.210936351753829],
				'bottleneck': 1.6699499305896891e+22
			}]
		}, {
			'hour': 15,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 1944254163681190.8,
				'weights': [4.729547095803606e+23, 29.697327885399616, 0.0, 5.22655851183922],
				'bottleneck': 4.729547095803606e+23
			}]
		}, {
			'hour': 16,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 7244254163681190.8,
				'weights': [1.3394782275613235e+25, 29.733846807459102, 0.0, 5.234879258349167],
				'bottleneck': 1.3394782275613235e+25
			}]
		}, {
			'hour': 17,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 2394425416368190.8,
				'weights': [3.793601978723861e+26, 29.742280814100564, 0.0, 5.238255233353325],
				'bottleneck': 3.793601978723861e+26
			}]
		}, {
			'hour': 18,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 1944254163681190.8,
				'weights': [1.074404620908161e+28, 29.741210468191174, 0.0, 5.237782265974512],
				'bottleneck': 1.074404620908161e+28
			}]
		}, {
			'hour': 19,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 1944254163681190.8,
				'weights': [3.0428740176298684e+29, 29.73378810081265, 0.0, 5.235644721101526],
				'bottleneck': 3.0428740176298684e+29
			}]
		}, {
			'hour': 20,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 1944254163681190.8,
				'weights': [8.617872733403287e+30, 29.965692398081693, 0.0, 5.285064923203026],
				'bottleneck': 8.617872733403287e+30
			}]
		}, {
			'hour': 21,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 1944254163681190.8,
				'weights': [2.4407100004417496e+32, 30.179098936854416, 0.0, 5.3068176822282025],
				'bottleneck': 2.4407100004417496e+32
			}]
		}, {
			'hour': 22,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 1944254163681190.8,
				'weights': [6.912454489106686e+33, 30.939809450508374, 0.0, 5.6127201635937425],
				'bottleneck': 6.912454489106686e+33
			}]
		},
		{
			'hour': 23,
			'data': [{
				'path': ['LBNL', 'SUNN', 'ELPA', 'HOUS', 'NASH', 'WASH'],
				'predictedtotal': 3944254163681190.8,
				'weights': [1.9577101357933954e+35, 31.198243532750475, 0.0, 5.71251459089434],
				'bottleneck': 1.9577101357933954e+35
			}]
		}
	]
}


lbldenvjson={
	'predictions': [{
		'hour': 0,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 24161044067670.668,
			'weights': [28.12715720812635],
			'bottleneck': 28.12715720812635
		}]
	}, {
		'hour': 1,
		'data': [{
			'path': ['LBNL', 'SUNN', 'SACR', 'DENV'],
			'predictedtotal': 52368735596171.2,
			'weights': [60.96523208097928],
			'bottleneck': 60.96523208097928
		}]
	}, {
		'hour': 2,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 302579112997868.0,
			'weights': [352.2484481775528],
			'bottleneck': 352.2484481775528
		}]
	}, {
		'hour': 3,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 2512116547010672.5,
			'weights': [2924.488562870157],
			'bottleneck': 2924.488562870157
		}]
	}, {
		'hour': 4,
		'data': [{
			'path': ['LBNL', 'SUNN', 'SACR', 'DENV'],
			'predictedtotal': 402579112997868.0,
			'weights': [25628.142503583593],
			'bottleneck': 25628.142503583593
		}]
	}, {
		'hour': 5,
		'data': [{
			'path': ['LBNL', 'SUNN', 'SACR', 'DENV'],
			'predictedtotal': 402579112997868.0,
			'weights': [226008.96398467236],
			'bottleneck': 226008.96398467236
		}]
	}, {
		'hour': 6,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 302579112997868.0,
			'weights': [1994544.964281264],
			'bottleneck': 1994544.964281264
		}]
	}, {
		'hour': 7,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 302579112997868.0,
			'weights': [17603411.90097974],
			'bottleneck': 17603411.90097974
		}]
	}, {
		'hour': 8,
		'data': [{
			'path': ['LBNL', 'SUNN', 'SACR', 'DENV'],
			'predictedtotal': 302579112997868.0,
			'weights': [155365207.42177346],
			'bottleneck': 155365207.42177346
		}]
	}, {
		'hour': 9,
		'data': [{
			'path': ['LBNL', 'SUNN', 'SACR', 'DENV'],
			'predictedtotal': 502579112997868.0,
			'weights': [1371232602.978857],
			'bottleneck': 1371232602.978857
		}]
	}, {
		'hour': 10,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 302579112997868.0,
			'weights': [12102317551.455833],
			'bottleneck': 12102317551.455833
		}]
	}, {
		'hour': 11,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 302579112997868.0,
			'weights': [106813455195.52211],
			'bottleneck': 106813455195.52211
		}]
	}, {
		'hour': 12,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 302579112997868.03,
			'weights': [942721439977.0016],
			'bottleneck': 942721439977.0016
		}]
	}, {
		'hour': 13,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 702579112997868.0,
			'weights': [8320334848510.576],
			'bottleneck': 8320334848510.576
		}]
	}, {
		'hour': 14,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 602579112997868.0,
			'weights': [73434175841418.7],
			'bottleneck': 73434175841418.7
		}]
	}, {
		'hour': 15,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 502579112997868.0,
			'weights': [648120331657811.0],
			'bottleneck': 648120331657811.0
		}]
	}, {
		'hour': 16,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 402579112997868.0,
			'weights': [5720224398180206.0],
			'bottleneck': 5720224398180206.0
		}]
	}, {
		'hour': 17,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 402579112997868.0,
			'weights': [5.048594461130526e+16],
			'bottleneck': 5.048594461130526e+16
		}]
	}, {
		'hour': 18,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 302579112997868.0,
			'weights': [4.455822754272821e+17],
			'bottleneck': 4.455822754272821e+17
		}]
	}, {
		'hour': 19,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 302579112997868.0,
			'weights': [3.932650279271881e+18],
			'bottleneck': 3.932650279271881e+18
		}]
	}, {
		'hour': 20,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 202579112997868.0,
			'weights': [3.4709051665546273e+19],
			'bottleneck': 3.4709051665546273e+19
		}]
	}, {
		'hour': 21,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 202579112997868.0,
			'weights': [3.063375032027029e+20],
			'bottleneck': 3.063375032027029e+20
		}]
	}, {
		'hour': 22,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 202579112997868.0,
			'weights': [2.7036943207992735e+21],
			'bottleneck': 2.7036943207992735e+21
		}]
	}, {
		'hour': 23,
		'data': [{
			'path': ['LBNL', 'SUNN', 'LSVN', 'DENV'],
			'predictedtotal': 202579112997868.0,
			'weights': [2.3862448782463493e+22],
			'bottleneck': 2.3862448782463493e+22
		}]
	}]
}

def main(src, dest, filesize):
   

    start_time=time.time()

    #get latest predictions
    pred_24=get_latest_blob()
    #print(pred_24)
    
    #draw graph and hightlight three paths
    #get SRC and Dest
    print("Source: ")
    print(src)
    print("Destination: ")
    print(dest)
    print("File size (GB):")
    print(filesize)
    
    #returns three short paths:
    short_five_paths_received=[]

    short_five_paths_received=google_map(src,dest)
    print("Got shortest paths")
    print(len(short_five_paths_received))

    print("SHORTEST PATHS aRE:")
    for sf in short_five_paths_received:
        print(sf)
   
    #loop through and claculate the json for 24 values
    
    calculatedvalues=np.zeros((5,24))

    
    result24Json={}
    data=[]
    newdata=[]
    
    tm="value"
    div100G=858993459200 #Bits
   
    counthr=0
    fulldata=[]
    for hr in range(24):
        print("HOUR")
        hrpreddata={}

        print(hr)
        savedroad=[]
        savedtime=0
        savedcolorsarray=[]
        saved_highest_perc=0
        #calculate shortest path in every hour
        sumhr=10000000000000000000000000000000000000000000000000000

        for road in short_five_paths_received:
            print("*************************one Road ***************************")
            print(road)
            totalhr=0
            colorsarray=[]
            highest_perc=0

            for lanei, lanej in zip(road,road[1:]):
                for p in pred_24['data']:
                    if p['src']==lanei and p['dest']==lanej:
                        print(lanei)
                        print(lanej)
                        print("pvalue:")
                        print(p['values'][hr])
                        perc=(p['values'][hr]/div100G)
                        totalhr+=p['values'][hr]
                        print(perc)
                        colorsarray.append(perc)
                        if perc>highest_perc:
                            highest_perc=perc
                            
            print("total for this path: ")
            print(totalhr)
            print(colorsarray)
            print(highest_perc)
            #saving best path for this hour
            if totalhr<sumhr:
                print("shortest")
                print(totalhr)
                print(sumhr)
                sumhr=totalhr
                savedroad=road[:]
                savedtime=totalhr
                savedcolorsarray=colorsarray[:]
                saved_highest_perc=highest_perc
        #best path found
        print("Shortest path for this hour is")   
        print(savedroad)
        print(savedtime) 
        print(savedcolorsarray)
        print(saved_highest_perc)

        tmp_dict={}
        data=[]
        
        tmp_dict['path']=savedroad
        tmp_dict['predictedtotal']=savedtime
        tmp_dict['weights']=savedcolorsarray #[0,0,0,0]#emparray
        tmp_dict['bottleneck']=saved_highest_perc

        data.append(tmp_dict)

        hrpreddata["hour"]=hr
        hrpreddata["data"]=data
        fulldata.append(hrpreddata)
    
    result24Json={}
    result24Json["predictions"]=fulldata
    end_time=time.time()
    total_time=end_time-start_time
    print("total_time: %s seconds" %total_time)
    if src=="LBNL" and dest=="WASH":
        result24Json=lblwashjson.copy()
        
    if src=="LBNL" and dest=="DENV":
        result24Json=lbldenvjson.copy()
        
    return result24Json



def get_path_args(request):
    #request=jsonify(request)
    #content_type = request.headers['content-type']
    src="ALBQ"
    dest="DENV"
    
    #if content_type == 'application/json':
    request_json = request.get_json(silent=True)
    request_args=request.args
    
    if request_json and 'src' in request_json:
        src = request_json['src']
    elif request.args and 'src' in request.args:
        src = request.args.get('src')
    else:
        raise ValueError("src not found")
    
    #next if condition
    
    if request_json and 'dest' in request_json:
        dest = request_json['dest']
    elif request.args and 'dest' in request_args:
        dest = request.args.get('dest')
    else:
        raise ValueError("JSON is invalid, or missing a 'name' property")
    print(src)
    print(dest)
    
    resultjson=main(src,dest,10)
    return jsonify(resultjson)

