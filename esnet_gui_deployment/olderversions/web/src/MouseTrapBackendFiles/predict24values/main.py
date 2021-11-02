
# Imports the Google Cloud client library
from google.cloud import storage
import logging
import os
import numpy as np
import pandas as pd

from pandas import DataFrame
from pandas import concat

import itertools
import statsmodels.api as sm
import warnings
import time
import simplejson as json
from datetime import datetime, timedelta
from google.cloud import firestore
from google.cloud import storage
import ast


#import cloudstorage as gcs
#import webapp2

#from google.appengine.api import app_identity



# TO DO: reading from a bucket
# locally read from file

# The name for the L1 bucket
#bucket_name = 'esnet-L1-bucket'


def get_links_data(filename):
    readfile=os.path.join(filename)
    readfileDF=pd.read_csv(readfile)
   
    readfileDF=readfileDF.drop(columns='Time')
    return readfileDF


def upload_blob(bucket_name, blob_text, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(str(blob_text))

    print('File {} uploaded to {}.'.format(
        blob_text,
        destination_blob_name))

class edge_totals:
    def __init__(self, name, src,dest, timestamps,values):
        self.name=name
        self.src=src
        self.dest=dest
        self.timestamps=[]
        self.values=[]
    
    def add_timestamps(self,timestamp):
        self.timestamps.append(timestamp)

    def add_values(self,value):
        self.values.append(value)

    
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


def build_edge_map():
    client = storage.Client()
    globalbucket = client.get_bucket('mousetrap_global_files')
    edgesblob=globalbucket.get_blob('esnet_edges.json')

    edgesblobstr=edgesblob.download_as_string()

    edge_dicts = json.loads(edgesblobstr)
    #print(edge_dicts)
    edges=edge_dicts['data']['mapTopology']['edges']
    
    return edges
    
def get_latest_data():

    edges=build_edge_map()

    db = firestore.Client()
    users_ref = db.collection(u'1hourrollups').stream()

    ct=0
    edgeCalendar=[]

    for doc in users_ref:
       
        receivedDict=doc.to_dict()
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




def main(self):
    #Loop through all links and call train and save model
    start_time=time.time()

    #read all hourly data and build csv
    edgeCalendar1=get_latest_data()
    for ec in edgeCalendar1:
        print(ec.src)
        print(ec.dest)
        print(ec.timestamps)
        print(ec.values)
        print("+++")

    #link_train_dataframe=get_links_data('../MLmodels/data/fulldata_1hour.csv')
   
    #define SARIMA model

    optimal_params_sarima=(1, 1, 1)
    optimal_params_seasonal=(1, 1, 1, 12)

    all_predictions=[]
    ts=0
    this_time=0

    print("building hourly database")
    for ec in edgeCalendar1:
        #print(ec.values)
        realtraffic=pd.DataFrame(ec.values)
        print(realtraffic)
               
        #realtraffic=new_row
        #pd.concat([new_row,realtraffic.ix[:]]).reset_index(drop=True)
        
        #realtraffic=pd.concat([new_row,realtraffic.ix[:]]).reset_index(drop=True)
        

        #print(realtraffic)
        print("Doing sarima")

        mod=sm.tsa.statespace.SARIMAX(realtraffic,
                                order=optimal_params_sarima,
                                seasonal_order=optimal_params_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
        results = mod.fit()
        print("fit done")
        pred_uc = results.get_forecast(steps=24)
        pred_ci = pred_uc.conf_int()
        #if ec.name == "SLAC--SUNN":
        
        
        ts=max(ec.timestamps)
        this_time=max(ec.timestamps)
        #print(ec.timestamps)


        write_preds=[]
        write_times=[]
        for i in range(24):
            czero=pred_uc.predicted_mean.values[i]
            if czero<0:
                czero=0
            else:
                czero=pred_uc.predicted_mean.values[i]
            write_preds.append(czero)
            uctimestamp=ts
            #print(uctimestamp)
            utctimeahead=uctimestamp + 3600     #1 hour
            utctimeahead= utctimeahead + (utctimeahead % 3600)
            write_times.append(utctimeahead)
            ts=utctimeahead
        print(write_preds)
        print(write_times)

        ep=edge_predictions(ec.name,ec.src,ec.dest)

        ep.add_timestamp(write_times)
        ep.add_mean(write_preds)
        ep.add_conf(pred_ci)

        all_predictions.append(ep) # edge_predictions(ec.name,ec.src,ec.dest,write_times,write_preds,pred_ci))
    #finish ec loop
    print("JSON")
    writedataDict={}
    writedataDict["timestamped"]=this_time

    
    listclassdata=[]

    prediction_json_file=[]
    # save data to json file
    for m in all_predictions:
        print(m.name)
        print(m.src)
        print(m.dest)
        print(m.timestamp)
        print(m.mean)
        listtimes=m.timestamp[0]
        listmeans=m.mean[0]
        print("listtimes")
        print(listtimes)
        
        print("check")

        #p is edge_totals
        classdata={}
        classdata["name"]=m.name
        classdata["src"]=m.src
        classdata["dest"]=m.dest
        classdata["timestamps"]=listtimes
        classdata["values"]=listmeans
        
        listclassdata.append(classdata)
        print("saved")
        print(m.name)
        
    #print(listclassdata)
    writedataDict["data"]=listclassdata   
    #print(len(prediction_json_file))
    print(writedataDict)
   

     # saving to bucket
    pred_bucket='latest24predictions-esnet'
    blob_name=str(this_time)
    blob_str=writedataDict

    upload_blob(pred_bucket,blob_str,blob_name)
    
    end_time=time.time()
    total_time=end_time-start_time
    print("total_time: %s seconds" %total_time)
    
    
