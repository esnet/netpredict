import os
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

def weeklynetworkcorr(inputfile, path):
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

    # Label
    df_label = pd.DataFrame(columns=['picName', 'Lable'])
    df_corr = pd.DataFrame()

    # Daytime is from sunrise (this varies, but we can say approximately 6am) to sunset (we can say approximately 6pm). Night-time is from sunset to sunrise.
    daycounter, count = 1, 0

    if inputfile_2018:
        start, stop = 12, 12 + 24*364
    elif inputfile_2019:
        start, stop = 9, 12 + 24*363
    step = 12

    while start < stop:

        night_df = df.loc[ start:start+step-1, : ] # df.loc[star:end,:], inclusive
        day_df = df.loc[ start+step:start+2*step-1, : ]

        corr_night = night_df.corr().round(2)
        corr_day = day_df.corr().round(2)

        # Combine columns and form a vector for ML
        corr_night_arry = pd.concat([corr_night.iloc[1:,0], corr_night.iloc[2:, 1]])
        for i in range( 2, len(corr_night) ):
            corr_night_arry = pd.concat([corr_night_arry, corr_night.iloc[(i+1):, i]])

        corr_day_arry = pd.concat([corr_day.iloc[1:,0], corr_day.iloc[2:, 1]])
        for i in range( 2, len(corr_day) ):
            corr_day_arry = pd.concat([corr_day_arry, corr_day.iloc[(i+1):, i]])

        # Remove NaN
        corr_night_arry = corr_night_arry.fillna(corr_night_arry.mean())
        corr_day_arry = corr_day_arry.fillna(corr_day_arry.mean())

        df_corr[count] = corr_night_arry
        df_corr[count+1] = corr_day_arry
        plt.figure(figsize=(30, 30))
        sns.heatmap(corr_night, cmap='coolwarm', yticklabels=False, xticklabels=False, cbar=False)
        night_pic_name = str(daycounter) + "night" + ".jpg"
        plt.savefig( path + night_pic_name )
        plt.close( 'all' )
        #
        plt.figure(figsize = (30,30))
        sns.heatmap(corr_day, cmap='coolwarm', yticklabels=False, xticklabels=False, cbar=False)
        day_pic_name = str(daycounter) + "day" + ".jpg"
        plt.savefig( path + day_pic_name )
        plt.close( 'all' )

        df_label.loc[count] = [night_pic_name, 1]
        df_label.loc[count + 1] = [day_pic_name, 0]

        daycounter += 1
        count += 2
        start = start + 2*step

    df_label.to_csv( path_or_buf = path + 'ImageLabel.csv', header=False, index=False)
    df_corr.to_csv(path_or_buf=path + 'Correlation.csv', header=False, index=False)



if __name__ == '__main__':

    inputfile_2018, inputfile_2019 = None, None
    inputfile_2019 = "../datasets/snmp_2019_data.csv"
    path_2019 = '../featurizers/2019/'
    # inputfile_2018= "../datasets/snmp_2018_1hourinterval.csv"
    # path_2018 = '../featurizers/2018/'

    if inputfile_2019:
        weeklynetworkcorr(inputfile_2019, path_2019)
    else:
        weeklynetworkcorr(inputfile_2018, path_2018)