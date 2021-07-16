"""

DDCRNN model developed in conjuction with Argonne National Laboratory and Lawrence Berkeley National Laboratory. 
Paper Link: https://arxiv.org/abs/2008.12767 Reference: Published IEEE BigData 2020

Authors
Tanwi Mallick
Bashir Mohammed
Mariam Kiran
Prasanna Balaprakash

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import yaml
import pandas as pd
import glob
import numpy as np
from lib.utils import load_graph_data
from lib.utils import generate_seq2seq_data
from lib.utils import train_val_test_split
from lib.utils import StandardScaler

from model.ddcrnn_supervisor import DCRNNSupervisor
from lib import metrics

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random
SEED = 3001
os.environ['PYTHONHASHSEED']=str(SEED) 
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)

        # Data preprocessing 
        traffic_df_filename = supervisor_config['data']['hdf_filename'] 
        df_data = pd.read_hdf(traffic_df_filename)
        supervisor_config['model']['num_nodes'] = num_nodes = df_data.shape[1]
        validation_ratio = supervisor_config.get('data').get('validation_ratio')
        test_ratio = supervisor_config.get('data').get('test_ratio')
        df_train, df_val, df_test = train_val_test_split(df_data, val_ratio=validation_ratio, test_ratio=test_ratio)

        batch_size = supervisor_config.get('data').get('batch_size')
        val_batch_size = supervisor_config.get('data').get('val_batch_size')
        test_batch_size = supervisor_config.get('data').get('test_batch_size')
        horizon = supervisor_config.get('model').get('horizon')
        seq_len = supervisor_config.get('model').get('seq_len')
        scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
        
        data_train = generate_seq2seq_data(df_train, batch_size, seq_len, horizon, num_nodes, 'train', scaler)
        data_val = generate_seq2seq_data(df_val, val_batch_size, seq_len, horizon, num_nodes, 'val', scaler)
        data_train.update(data_val)
        #data_train['scaler'] = scaler
    
        data_test = generate_seq2seq_data(df_test, test_batch_size, seq_len, horizon, num_nodes, 'test', scaler)
        #data_test['scaler'] = scaler


        tf_config = tf.compat.v1.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(data_train, supervisor_config)
            
            data_tag = supervisor_config.get('data').get('dataset_dir')
            folder = data_tag + '/model/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            # Train
            supervisor.train(sess=sess)

            # Test
            yaml_files = glob.glob('%s/model/*/*.yaml'%data_tag, recursive=True)
            yaml_files.sort(key=os.path.getmtime)
            config_filename = yaml_files[-1] #'config_%d.yaml' % config_id
            
            with open(config_filename) as f:
                config = yaml.safe_load(f)
            # Load model and predict
            supervisor.load(sess, config['train']['model_filename'])
            y_preds = supervisor.predict(sess, data_test)
            
            # Evaluate
            n_test_samples = data_test['y_test'].shape[0]
            folder = data_tag + '/results/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            for horizon_i in range(data_test['y_test'].shape[1]):
                y_pred = scaler.inverse_transform(y_preds[:, horizon_i, :, 0])
                eval_dfs = df_test[seq_len + horizon_i: seq_len + horizon_i + n_test_samples]
                df = pd.DataFrame(y_pred, index=eval_dfs.index, columns=eval_dfs.columns)
                filename = os.path.join('%s/results/'%data_tag, 'dcrnn_speed_prediction_%s.h5' %str(horizon_i+1))
                df.to_hdf(filename, 'results')
                
                mae = metrics.masked_mae_np(df.values, eval_dfs.values, null_val=0)
                mape = metrics.masked_mape_np(df.values, eval_dfs.values, null_val=0)
                rmse = metrics.masked_rmse_np(df.values, eval_dfs.values, null_val=0)
                print(
                    "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                        horizon_i + 1, mae, mape, rmse
                    ))

            print('Predictions saved as %s/results/dcrnn_prediction_[1-12].h5...' %data_tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
