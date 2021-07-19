install instructions:
Requirements
conda create -n ddcrnn_env python=3.7
conda activate ddcrnn_env

conda install tensorflow=2.4.0
conda install pandas=1.2.5
conda install pyaml=20.4.0
conda install numpy=1.19.5
conda install scipy=1.6.2
pip install --upgrade tables


running the file: python3 ddcrnn_train.py --config_filename=data/ddcrnn_config.yaml
