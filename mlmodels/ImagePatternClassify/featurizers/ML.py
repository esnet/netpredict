
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data_path = r"C:\Users\sheng\Desktop\Postdoc berkeley\DAPHNE\Code\ImagePatternClassify\featurizers\2019"
os.chdir(data_path)

# Read training data and divide it into training and test with ratio 0.7:0.3
Model_inputs = pd.read_csv('Correlation.csv', header=None)
y_true = pd.read_csv('ImageLabel.csv', header=None)
X_train, X_test, y_train, y_test = train_test_split(Model_inputs.T, y_true[1], test_size=0.3, random_state=0)


def logisticRegression():

    logreg = LogisticRegression(penalty="elasticnet", solver= "saga", l1_ratio=0.8, tol=0.01 )
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression on test set: {:.2f}%'.format(logreg.score(X_test, y_test)*100))

    return y_pred

def SupportVectorMachine():

    clf = svm.SVC(kernel='linear')  # Linear Kernel
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('Accuracy of support vector machine on test set: {:.2f}%'.format(clf.score(X_test, y_test) * 100))

    return y_pred

def K_nearest_neighbor():

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print('Accuracy of K-nearest neighbor on test set: {:.2f}%'.format(knn.score(X_test, y_test) * 100))

    return y_pred

def Random_Forest():

    RF = RandomForestClassifier(n_estimators=100)
    RF.fit(X_train, y_train)

    y_pred = RF.predict(X_test)
    print('Accuracy of Random Forest on test set: {:.2f}%'.format(RF.score(X_test, y_test) * 100))

    return y_pred

def plotConfutionMatrix( y_test, y_pred, modelName):

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    # ax.margins(x=2, y=2)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")

    plt.title('Confusion matrix'+ " " + modelName, y=0.8) #,
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return

y_pred_LR = logisticRegression()
y_pred_SVM = SupportVectorMachine()
y_pred_KNN = K_nearest_neighbor()
y_pred_RF = Random_Forest()

plotConfutionMatrix(y_test, y_pred_LR, "logistic_Regression")
plotConfutionMatrix(y_test, y_pred_SVM, "Support_Vector_Machine")
plotConfutionMatrix(y_test, y_pred_KNN, "K-Nearest Neighbor")
plotConfutionMatrix(y_test, y_pred_RF, "Random Forest")