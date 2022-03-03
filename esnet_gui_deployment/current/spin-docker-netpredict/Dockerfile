# syntax=docker/dockerfile:1

# init a base image (Alpine is small Linux distro)
FROM ubuntu:latest

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN \
    apt-get update              &&  \
    apt-get upgrade --yes       &&  \
    apt-get install --yes           \
        python3 pip  mysql-client  \
        python3-mysql.connector  && \
    apt-get clean

# define the present working directory
WORKDIR /web_flask

# copy requirements
COPY requirements.txt requirements.txt

# run pip to install the dependencies of the flask app
RUN pip install -r requirements.txt

# copy the contents into the working dir
COPY . .


# RUN python3 /srv/database_connectivity.py

# define the command to start the container
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host=0.0.0.0"]

ADD docker-entrypoint.sh /srv/docker-entrypoint.sh
RUN chmod +x /srv/docker-entrypoint.sh
ENTRYPOINT ["/srv/docker-entrypoint.sh"]
