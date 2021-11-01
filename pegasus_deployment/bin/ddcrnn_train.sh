#!/usr/bin/env bash

tar -xzf ddcrnn_package.tar.gz

python3 ddcrnn_train.py $@

tar -czf model.tar.gz model
tar -czf results.tar.gz results
