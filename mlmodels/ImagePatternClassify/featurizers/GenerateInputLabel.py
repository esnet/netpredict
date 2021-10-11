
import pandas as pd
import numpy as np

def weeklynetworkcorr(inputfile, path):
    print('hello')
    df = pd.read_csv(inputfile)
    print(df)
    del df['time']
    print(df.head())

    # Removing first index column
    df=df.iloc[:,1:]
    print(df.head())

    # Replace NaN with column mean
    df = df.fillna(df.mean())
    # Replace NaN with row mean
    df = df.T.fillna(df.mean(axis=1)).T

    # Initialize datafream for data and the corresponding Labels
    df_label = pd.DataFrame(columns=['Lable'])
    df_corr_AE = pd.DataFrame()

    # Daytime is from sunrise (this varies, but we can say approximately 6am) to sunset (we can say approximately 6pm). Night-time is from sunset to sunrise.
    # step refers to the resolution, Each sample involves 12 hours traffic data. You may decrease this value to 1 for a higher resolution, where each sample involves 1 hour traffic data.
    count, step = 0, 12

    if inputfile_2018:
        daycounter, start, stop = 1, 12, 12 + 24 * 364 # Start from Monday
    elif inputfile_2019:
        daycounter, start, stop = 2, 9, 12 + 24 * 363 # Start from Tuesday
    else:
        daycounter, start, stop = 3, 9, 12 + 24 * 363  # Start from Wednesday

    while start < stop:

        # Extract nighttime and daytime for one day. The nighttime corresponds to 18: 00 - 5:00 and daytime corresponds to 6: 00 - 17:00, respectively.
        night_df = df.loc[ start:start+step-1, : ] # df.loc[star:end,:], inclusive
        day_df = df.loc[ start+step:start+2*step-1, : ]

        # Pearson correlation
        corr_night = night_df.corr().round(2)
        corr_day = day_df.corr().round(2)
        df_corr_AE = pd.concat([df_corr_AE, corr_night, corr_day])

        # Create Label 0 maps night, 1,2,..7 map Monday, Thuesday, ... Sunday
        df_label.loc[count] = 0  # night
        df_label.loc[count + 1] = daycounter % 7 if daycounter % 7 else 7

        daycounter += 1
        count += 2
        start = start + 2*step

    # Fill empty column with zero or it's mean
    # df_corr_AE = df_corr_AE.fillna(0)
    df_corr_AE = df_corr_AE.fillna(df_corr_AE.mean())

    # Check if there is any NaN or Inf in data
    np.where(np.asanyarray(np.isnan(df_corr_AE)))
    np.isinf(df_corr_AE).values.sum()

    # Save the input and labels
    df_corr_AE.to_csv( path_or_buf = path + 'inputAE.csv', header=False, index=False )
    df_label.to_csv( path_or_buf = path + 'AELabel.csv', header=False, index=False )

if __name__ == '__main__':

    # Define directory
    inputfile_2018, inputfile_2019, inputfile_2020 = None, None, None
    inputfile_2020 = "../datasets/snmp_2020_data.csv"
    path_2020 = '../featurizers/2020/Abnormal'
    inputfile_2019 = "../datasets/snmp_2019_data.csv"
    path_2019 = '../featurizers/2019/'
    inputfile_2018= "../datasets/snmp_2018_data.csv"
    path_2018 = '../featurizers/2018/'

    weeklynetworkcorr(inputfile_2020, path_2020)
    weeklynetworkcorr(inputfile_2019, path_2019)
    weeklynetworkcorr(inputfile_2018, path_2018)