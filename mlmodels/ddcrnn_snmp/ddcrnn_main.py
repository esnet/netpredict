"""
DDCRNN model developed in conjuction Lawrence Berkeley National Laboratory and Argonne National Laboratory.
Adapted from original paper: "This is a PyTorch implementation of Diffusion Convolutional Recurrent Neural Network in the following paper:
Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting, ICLR 2018."

Network Paper Link: https://arxiv.org/abs/2008.12767

Reference: Published IEEE BigData 2020

Authors
Tanwi Mallick
Bashir Mohammed
Mariam Kiran
Prasanna Balaprakash


Run: python3 ddcrnn_main.py --config_filename=configs/ddcrnn_config.yaml 

"""

#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import yaml
import pandas as pd
import glob
import os
import sys

#sys.path.append("/Users/mkiran/SWProjects/calibers/daphne/mouseTrap/MLmodels/DDCRNN/DDCRNN/")
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from lib.utils import load_graph_data

from modellib.ddcrnn_supervisor import DCRNNSupervisor

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main(args):

    #print("here")
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        #print(supervisor_config)
        horizon = supervisor_config.get('model').get('horizon')
        print(horizon)

        supervisor = DCRNNSupervisor(supervisor_config)
        print(supervisor)
        # Train
        data_tag = supervisor_config.get('data').get('dataset_dir')
        print(data_tag)
        supervisor.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)

