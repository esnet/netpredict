# pull official base image
#FROM node:13.12.0-alpine

# set working directory
#WORKDIR /web

# add `/app/node_modules/.bin` to $PATH
#ENV PATH /web/node_modules/.bin:$PATH

# install app dependencies
#COPY package.json ./
#COPY package-lock.json ./
#RUN yarn install 

# add app
#COPY . ./

# start app
#CMD ["npm", "start"]


# stage: 1
#FROM node:12 as react-build
#WORKDIR /app
#COPY . ./
#RUN yarn
#RUN yarn build

# stage: 2 — the production environment
#FROM nginx:alpine
#COPY — from=react-build /app/build /usr/share/nginx/html
#EXPOSE 80
#CMD [“npm”, “-g”, “daemon off;”]


#New workflow

# pull official base image
FROM node:13.12.0-alpine

# set working directory
WORKDIR /web

# add `/app/node_modules/.bin` to $PATH
ENV PATH /web/node_modules/.bin:$PATH

# install app dependencies
COPY package.json ./
COPY package-lock.json ./
RUN npm install --silent
RUN npm install react-scripts@3.4.1 -g --silent

# add app
COPY . ./

# start app
CMD ["npm", "start"]

