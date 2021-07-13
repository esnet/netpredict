import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import warnings
import itertools
from datetime import datetime



inputfile= "../datasets/snmp_2019_data.csv"


def weeklynetworkcorr():
    print('hello')
    df = pd.read_csv(inputfile)
    print(df)
    del df['time']
    print(df.head())

    #removing empty column
    df=df.iloc[:,1:]
    print(df.head())

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df) 
    df.loc[:,:] = scaled_values
    print(df.head())

    weekcounter = 1
    start=0
    stop=8738
    step=168

    while start<stop:
        weekdf=df.loc[start:start+step,:]
        corr =weekdf.corr().round(2)


        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["font.family"] = "sans-serif"
        plt.figure(figsize = (60,50))
        sns.heatmap(corr,cmap='coolwarm',annot=True, annot_kws={"size":10})
        #plt.title("Year-2019 January-Week1", fontsize=80)
        plt.savefig(str(weekcounter)+".png")
        weekcounter=weekcounter+1
        start=start+step+1
        plt.close(str(weekcounter)+".png")







weeklynetworkcorr()