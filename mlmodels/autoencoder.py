#autoencoder.py
from __future__ import print_function

from pandas import read_csv, DataFrame
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#----



#raw_data={'Average rtt C2S', 'Average rtt S2C','target'}
#df=pd.DataFrame(raw_data, columns = ['Sent','Received','Lost','Duplicated','Reordered'])

label_file = os.path.join("newworkflow_experiments/full_new_experiments.csv")

df=pd.read_csv(label_file)


# load dataset into Pandas DataFrame
df = pd.read_csv(label_file, names=['Average rtt C2S', 'Average rtt S2C',
	'max seg size1','min seg size1','win max1','win min1','win zero1','cwin max1',
	'cwin min1','initial cwin1','rtx RTO1','rtx FR1','reordering1','net dup1','max seg size',
	'in seg size','win max','win min','win zero','cwin max','cwin min','initial cwin','rtx RTO','rtx FR',
	'eordering','net dup','target'])

print("*******")
#print df

#extracting features
features = ['Average rtt C2S', 'Average rtt S2C','max seg size1','min seg size1','win max1',
'win min1','win zero1','cwin max1','cwin min1','initial cwin1','rtx RTO1','rtx FR1','reordering1','net dup1',
'max seg size','in seg size','win max','win min','win zero','cwin max','cwin min','initial cwin','rtx RTO','rtx FR',
'eordering','net dup']


# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
print(y)
# Standardizing the features
x = StandardScaler().fit_transform(x)

print("1")

targets = ['Normal', 'Loss1%', 'Loss3%', 'Loss5%', 'Dup1%', 'Dup3%','Dup5%','Reord10%','Reord30%','Reord50%']

  

colors = ['r', 'g', 'b', 'black', 'lime', 'yellow', 'cyan', 'coral', 'gray', 'indigo']


 
# SCALE EACH FEATURE INTO [0, 1] RANGE
sX = minmax_scale(x, axis = 0)
ncol = sX.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(sX, y, train_size = 0.1, random_state = seed(2017))
 
### AN EXAMPLE OF SIMPLE AUTOENCODER ###
# InputLayer (None, 10)
#      Dense (None, 5)
#      Dense (None, 10)
print("training")
print(X_train)

print("test")
print(X_test)

input_dim = Input(shape = (ncol, ))
# # DEFINE THE DIMENSION OF ENCODER ASSUMED 3
encoding_dim = 2
# # DEFINE THE ENCODER LAYER
encoded = Dense(encoding_dim, activation = 'relu')(input_dim)
# # DEFINE THE DECODER LAYER
decoded = Dense(ncol, activation = 'sigmoid')(encoded)
# # COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
autoencoder = Model(input = input_dim, output = decoded)
# # CONFIGURE AND TRAIN THE AUTOENCODER
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.fit(X_train, X_train, nb_epoch = 50, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))

# # THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
encoder = Model(inputs = input_dim, outputs = encoded)
encoded_input = Input(shape = (encoding_dim, ))
encoded_out = encoder.predict(X_test)
encoded_out[0:2]

# cellvalue='abc'

pd.set_option('display.max_columns', None)

log = open("resultsc.csv", "w")
for x in encoded_out:
	cellvalue = np.array2string(x, precision=4, separator=',',suppress_small=True)
 	cellvalue = cellvalue.replace("[", "")
 	cellvalue = cellvalue.replace("]", "")
 	print(cellvalue, file = log)

print(encoded_out) #print into a csv file

# #for x in encoded_out:
# 	#print x

print(encoded_out.size)
#array([[ 0.        ,  1.26510417,  1.62803197],
#       [ 2.32508397,  0.99735016,  2.06461048]], dtype=float32)
######

# input_dim = Input(shape = (ncol, ))
# # DEFINE THE DIMENSION OF ENCODER ASSUMED 3
# encoding_dim = 2
# # DEFINE THE ENCODER LAYERS
# encoded1 = Dense(20, activation = 'relu')(input_dim)
# encoded2 = Dense(10, activation = 'relu')(encoded1)
# encoded3 = Dense(5, activation = 'relu')(encoded2)
# encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)
# # DEFINE THE DECODER LAYERS
# decoded1 = Dense(5, activation = 'relu')(encoded4)
# decoded2 = Dense(10, activation = 'relu')(decoded1)
# decoded3 = Dense(20, activation = 'relu')(decoded2)
# decoded4 = Dense(ncol, activation = 'sigmoid')(decoded3)
# # COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
# autoencoder = Model(input = input_dim, output = decoded4)
# # CONFIGURE AND TRAIN THE AUTOENCODER
# autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
# autoencoder.fit(X_train, X_train, nb_epoch = 100, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))
# # THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
# encoder = Model(input = input_dim, output = encoded4)
# encoded_input = Input(shape = (encoding_dim, ))
# encoded_out = encoder.predict(X_test)
# encoded_out[0:2]

# log = open("results2.csv", "w")
# for x in encoded_out:
# 	cellvalue = np.array2string(x, precision=4, separator=',',suppress_small=True)
# 	cellvalue = cellvalue.replace("[", "")
# 	cellvalue = cellvalue.replace("]", "")
# 	print(cellvalue, file = log)

# print(encoded_out) #print into a csv file

# # #for x in encoded_out:
# # 	#print x

# print(encoded_out.size)
# #array([[ 0.        ,  1.26510417,  1.62803197],
#       [ 2.32508397,  0.99735016,  2.06461048]], dtype=float32)

