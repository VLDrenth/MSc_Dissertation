import pickle
import time
import json
import os

def generate_experiment_id():
    return time.strftime("%Y%m%d-%H%M%S")

experiment_id = generate_experiment_id()