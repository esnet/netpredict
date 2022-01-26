## Steps to run the STGNN ML Model


Install Dependecies:

    pip3 install tables

    pip3 install dgl


Change to featurizers folder and run the csv_to_hdf.py file: Remember to change your input file to load current dataset:

	cd netpredict/featurizers/

	python3 csv_to_hdf.py


Change to spatio-temporalGCN folder and run the main.py file:

	cd netpredict/mlmodels/spatio-temporalGCN/

	python3 main.py