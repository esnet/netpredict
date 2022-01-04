# Netpredict: Flask-GUI App



## Step 1 : Installation 

 To get started I recommend to first setup a clean Python environment for your project with at least Python 3.6 using any of your favorite tool for instance, ([conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html "conda-env"), [venv](https://docs.python.org/3/library/venv.html), [virtualenv](https://virtualenv.pypa.io/en/latest/) with or without [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)).


Navigate into the "web_flask" folder.


    cd netpredict/esnet_gui_deployment/current/web_flask/

Create a virtual environmnet called "netpredvenv". 

    python3 -m virtualenv netpredvenv

Activate your virtual environment

    source netpredvenv/bin/activate

Install all the required dependencies

    pip3 install -r requirements.txt


## Step 2: Connecting to NetPredict Database

Login to MySQL on via your terminal as a root user

    /usr/local/mysql/bin/mysql -u root -p
   
    password: rootroot
    
Create a NetPredict database by running the following MYSQL syntax

    create database NetPredict
    
Then go to: netpredict/esnet_gui_deployment/current/web_flask/  and run the following:

    python3 database_connectivity.py

## Step 3: Launch your NetPredict app

Launch your netpredict app to run on port 5000:

    python3 main.py

Go to http://127.0.0.1:5000 to access your GUI.


## Contact Us
See attached Licence to Lawrence Berkeley National Laboratory
Email: Mariam Kiran <mkiran@es.net>
