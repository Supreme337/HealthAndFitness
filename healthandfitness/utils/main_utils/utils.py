import yaml
import pickle
import os,sys
import numpy as np
from healthandfitness.exception.exception import HealthAndFitnessException
from healthandfitness.logging.logger import logging
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import r2_score

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HealthAndFitnessException(e,sys)

def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise HealthAndFitnessException(e,sys)
