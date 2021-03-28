# pull official base image
FROM node:13.12.0-alpine

# set working directory
WORKDIR /web

# add `/app/node_modules/.bin` to $PATH
ENV PATH /web/node_modules/.bin:$PATH

# install app dependencies
COPY package.json ./
COPY package-lock.json ./
RUN yarn install 

# add app
COPY . ./

# start app
CMD ["npm", "start"]
