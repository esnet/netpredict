import numpy as np
import pandas as pd
import tables

#filename = '/tmp/test.hdf5'
filename = '../datasets/snmp_esnet/snmp_2019_data.hdf5'
inputfile= "../datasets/snmp_esnet/snmp_2019.csv"


df = pd.read_csv(inputfile)
df=df.replace(np.nan, 0)

print(df)
df=df.iloc[:,1:]

# Save to HDF5

df.to_hdf(filename, 'data', mode='w', format='table')
del df    # allow df to be garbage collected
print("hdf created")

# Append more data
#df2 = pd.DataFrame(np.arange(10).reshape((4,2))*10, columns=['C1', 'C2'])
#df2.to_hdf(filename, 'data', append=True)

print(pd.read_hdf(filename, 'data'))
