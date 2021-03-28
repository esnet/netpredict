import logging
import os
from flask import escape

from google.cloud import storage

def hello_world(request):
    client = storage.Client()
    bucket = client.get_bucket('latest24predictions-esnet')
    blob=bucket.get_blob('test_blob')
    
    return blob.download_as_string()
