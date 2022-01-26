## Steps to run the STGNN ML Model


Install Dependecies:

    pip3 install tables

    pip3 install dgl


Change to featurizers folder and run the [csv_to_hdf.py](https://github.com/esnet/netpredict/blob/main/featurizers/csv_to_hdf.py) file. Remember to change your input file path on line 7 to load current dataset. All dataset can be found [here](https://github.com/esnet/netpredict/tree/main/datasets/snmp_esnet).

	cd netpredict/featurizers/

	python3 csv_to_hdf.py


Change to spatio-temporalGCN folder and run the main.py file:

	cd netpredict/mlmodels/spatio-temporalGCN/

	python3 main.py