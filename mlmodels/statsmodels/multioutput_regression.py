# collection of simple multi Output regression models that can be called by any example

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

test_samplex=[0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303]
test_sampley=[-0.29257894, -0.2820059]

# linear regression for multioutput regression



def NP_LinearRegression(x):
    # create datasets

    print("HHHH")
    print(x)
    #split incoming data into X and y 
    #x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size=2, random_state=0)
    # define model
    X, y = make_regression(n_samples=10, n_features=1, n_informative=5, n_targets=5, random_state=1, noise=0.5)
    # define model
    # fit model
    print("1HHHH")
    print(x)
    print("ffff")
    print(y)
    model = LinearRegression()

    # fit model
    model.fit(X, y)

    print("2HHHH")

    # make a prediction
    row = [-0.00290545]
    yhat = model.predict([row])
    # summarize prediction
    print(yhat[0])


#NP_LinearRegression(test_samplex, test_sampley)