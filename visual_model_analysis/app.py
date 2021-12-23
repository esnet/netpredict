from flask import Flask, request, session, g, redirect,\
             url_for, abort, render_template, flash, jsonify
import os
import flask
import os.path as path
from server.Test_dataset import Dataset
from server.dgl_karate_club import KarateModel


from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)
App = Flask(__name__)

class mainModel(object):
    def __init__(self):
        print("Initialized Prototype")
        self.Dataset = Dataset.load_test_data()

@App.route('/')
def index():
    # send matrix data to index
    return flask.render_template("views/index.html",
                         Misclassification= modelData.Dataset.misclassification,\
                         Class_Data_B= modelData.Dataset.ground_truth_classification_data_B, 
                         Class_Data_A=modelData.Dataset.ground_truth_classification_data_A,\

                         statistics=modelData.Dataset.statistics,
                         sentenceFilter=modelData.Dataset.sentenceFilter,
                         calculateAgreement=modelData.Dataset.calculateAgreement,
                         Model_A_Name = modelData.Dataset.model_A["Model_Type"],
                         Model_B_Name = modelData.Dataset.model_B["Model_Type"],
                         Test_input_size = len(modelData.Dataset.model_A["Input Test Data"]["Predicted Outcome"]))

if __name__ == "__main__":
    App.secret_key = 'super secret key'
    App.config['SESSION_TYPE'] = 'filesystem'
    modelData = mainModel()

    port = 8080
    # Open a web browser pointing at the app.
    os.system("open http://localhost:{0}".format(port))

    #AppSet up the development server on port 8000.
    App.debug = True
    App.run(port=port)




