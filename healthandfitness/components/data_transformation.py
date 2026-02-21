import os
import sys
import pandas as pd
import numpy as np
from typing import List
from healthandfitness.exception.exception import HealthAndFitnessException
from healthandfitness.constant.training_pipeline import COLUMNS_TO_DROP
from healthandfitness.logging.logger import logging
from healthandfitness.entity.config_entity import DataTransformationConfig
from healthandfitness.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from healthandfitness.constant.training_pipeline import TARGET_COLUMN
from healthandfitness.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from healthandfitness.utils.main_utils.utils import save_numpy_array_data,save_object
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig,data_validation_artifact:DataValidationArtifact):
        self.data_transformation_config=data_transformation_config
        self.data_validation_artifact=data_validation_artifact

    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        df=pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}, shape={df.shape}")
        return df

    def separate_columns(self,df:pd.DataFrame):
        num_cols=df.select_dtypes(include=np.number).columns.to_list()
        cat_cols=df.select_dtypes(exclude=np.number).columns.to_list()

        if TARGET_COLUMN in num_cols:
            num_cols.remove(TARGET_COLUMN)
        if TARGET_COLUMN in cat_cols:
            cat_cols.remove(TARGET_COLUMN)

        logging.info(f"Numeric cols: {num_cols}")
        logging.info(f"Categorical cols: {cat_cols}")

        return num_cols, cat_cols

    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info('Started Data Transformation')

            train_df=self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=self.read_data(self.data_validation_artifact.valid_test_file_path)

            train_df.drop(columns=[c for c in COLUMNS_TO_DROP if c in train_df.columns],inplace=True)
            test_df.drop(columns=[c for c in COLUMNS_TO_DROP if c in test_df.columns],inplace=True)
            num_cols,cat_cols=self.separate_columns(train_df)

            preprocessor={
                "num_cols":num_cols,
                "cat_cols":cat_cols,
                "column_order":train_df.drop(columns=[TARGET_COLUMN]).columns.tolist()
            }

            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_object_file_path),exist_ok=True)
            save_object(self.data_transformation_config.preprocessor_object_file_path,preprocessor)
            logging.info("Minimal CatBoost preprocessor saved")

            if num_cols:
                imputer=KNNImputer(n_neighbors=3)
                train_df[num_cols]=imputer.fit_transform(train_df[num_cols])
                test_df[num_cols]=imputer.transform(test_df[num_cols])
            else:
                imputer=None

            for col in cat_cols:
                train_df[col]=train_df[col].fillna("Missing")
                test_df[col]=test_df[col].fillna("Missing")

            save_object(self.data_transformation_config.imputer_object_file_path,imputer)
            save_object(self.data_transformation_config.categorical_columns_file_path,cat_cols)

            logging.info("Saved imputer and categorical columns")
            
            train_path=self.data_transformation_config.transformed_train_file_path
            test_path=self.data_transformation_config.transformed_test_file_path

            os.makedirs(os.path.dirname(train_path),exist_ok=True)

            train_df.to_csv(train_path,index=False)
            test_df.to_csv(test_path,index=False)

            logging.info("Saved transformed CSV files")
            
            return DataTransformationArtifact(
                transformed_train_file_path=train_path,
                transformed_test_file_path=test_path,
                transformed_object_file_path="",
                categorical_columns_file_path=self.data_transformation_config.categorical_columns_file_path,
                imputer_object_file_path=self.data_transformation_config.imputer_object_file_path,
                preprocessor_object_file_path=self.data_transformation_config.preprocessor_object_file_path)

        except Exception as e:
            raise HealthAndFitnessException(e,sys)
