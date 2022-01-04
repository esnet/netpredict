# init a base image (Alpine is small Linux distro)
FROM python:3.7.6-alpine
# define the present working directory
WORKDIR /web_flask
# copy the contents into the working dir
ADD . /web_flask
# run pip to install the dependencies of the flask app
RUN pip3 install -r requirements.txt
# define the command to start the container
CMD ["python","app.py"]