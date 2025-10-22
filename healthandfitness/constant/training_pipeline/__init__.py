import os
import sys
import pandas as pd
import numpy as np

"""Some common constants"""
TARGET_COLUMN="Burns_Calories_Bin"
PIPELINE_NAME="HealthAndFitnessPipeline"
ARTIFACTS_DIR="Artifacts"
FILE_NAME="Final_Data.csv"

TRAIN_FILE_NAME="train.csv"
TEST_FILE_NAME="test.csv"
PREPROCESSING_OBJECT_FILE_NAME="preprocessing.pkl"

SCHEMA_FILE_PATH=os.path.join("data_schema","schema.yaml")

"""Data Ingestion Constants"""
DATA_INGESTION_COLLECTION_NAME="Raw Data"
DATA_INGESTION_DATABASE_NAME="HealthAndFitness"
DATA_INGESTION_DIR_NAME="data_ingestion"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO=0.2
DATA_INGESTION_FEATURE_STORE_DIR="feature_store"
DATA_INGESTION_INGESTED_DIR="ingested_data"

"""Data Validation Constants"""
DATA_VALIDATION_DIR_NAME="data_validation"
DATA_VALIDATION_VALID_DIR="valid"
DATA_VALIDATION_INVALID_DIR="invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR="drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME="report.yaml"
PREPROCESSING_OBJECT_FILE_NAME="preprocessing.pkl"