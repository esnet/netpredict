# init a base image (Alpine is small Linux distro)
#FROM python:3.9.9-slim-buster
# define the present working directory
#WORKDIR /web_flask
# copy the contents into the working dir
#ADD . /web_flask
# run pip to install the dependencies of the flask app
#RUN pip3 install -r requirements.txt
# define the command to start the container
#CMD ["python","app.py"]


# using ubuntu LTS version
FROM ubuntu:20.04

# avoid stuck build due to user prompt
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y python3 pip mysql-server wget curl && apt-get clean

# install requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN pip3 install mysql-connector-python-rf

# define the present working directory
WORKDIR /docker-netpredict


# copy the contents into the working dir
ADD . /docker-netpredict


EXPOSE 5000

# make sure all messages always reach console
ENV PYTHONUNBUFFERED=1

# /dev/shm is mapped to shared memory and should be used for gunicorn heartbeat
# this will improve performance and avoid random freezes
#run database_connection
RUN python3 database_connectivity.py

# define the command to start the container
CMD ["python3","app.py"]