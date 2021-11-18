# collection of simple multi Output regression models that can be called by any example

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

test_samplex=[0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303]
test_sampley=[-0.29257894, -0.2820059]

# linear regression for multioutput regression



def NP_LinearRegression(x):
    # create datasets

    
    #split incoming data into X and y 
    #x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size=2, random_state=0)
    # define model
    X, y = make_regression(n_samples=10, n_features=11, n_informative=5, n_targets=5, random_state=1, noise=0.5)
    # define model
    # fit model
    model = LinearRegression()

    # fit model
    model.fit(X, y)

    print("Predictions...")
    predDf = pd.DataFrame(index=[0],columns=x)
    for col in predDf.columns:
        predDf[col]=predDf[col].fillna(0)
    #print(predDf)
    # make a prediction
    for colname in x:
        ys=x[colname]
        
        y_hats2 = model.predict([ys])
        print("For link id: ",colname)
        print(y_hats2)

        #predDf=predDf.append(predDf.loc[colname], ignore_index=True)

        #predDf[colname].append(y_hats2)

        #predDf = pd.concat([predDf,y_hats2])

#NP_LinearRegression(test_samplex, test_sampley)