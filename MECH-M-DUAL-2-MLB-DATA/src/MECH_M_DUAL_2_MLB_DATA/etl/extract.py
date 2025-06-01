import numpy as np
import scipy
import logging
import requests
import io

def extract(url: str, key: str):
    logging.debug(f"Extract data from {url}")
    response = requests.get(url)
    image = scipy.io.loadmat(io.BytesIO(response.content))[key]
    return image