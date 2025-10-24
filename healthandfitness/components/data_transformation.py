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

class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig,data_validation_artifact:DataValidationArtifact):
        try:
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
        except Exception as e:
            raise HealthAndFitnessException(e,sys)
    
    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            df=pd.read_csv(file_path)
            logging.info(f"Loaded data from {file_path}, shape={df.shape}")
            return df
        except Exception as e:
            raise HealthAndFitnessException(e,sys)
    
    def separate_columns(elf,df:pd.DataFrame):
        try:
            num_cols=df.select_dtypes(include=np.number).columns.to_list()
            cat_cols=df.select_dtypes(exclude=np.number).columns.to_list()
            logging.info(f"Found len({num_cols}) numeric and len({cat_cols}) categorical columns")
            return num_cols, cat_cols
        except Exception as e:
            raise HealthAndFitnessException(e,sys)
    
    def handle_missing_values(self,df=pd.DataFrame, num_cols:List[str], cat_cols:List[str])->pd.DataFrame:
        try:
            if num_cols:
                imputer=KNNImputer(n_neighbors=3)
                df[num_cols]=imputer.fit_transform(df[num_cols])
                save_object(self.data_transformation_config.imputer_object_file_path,imputer)
                logging.info("KNNImputer initiated")
            
            if cat_cols:
                for col in cat_cols:
                    df[col]=df[col].fillna("Missing")
                    logging.info(" Missing values in categorical columns filled with 'Missing'")

            return df
        except Exception as e:
            raise HealthAndFitnessException(e,sys)
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info('Started Data Transformation')
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            columns_to_drop=COLUMNS_TO_DROP
            train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns],inplace=True)
            test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns],inplace=True)

            num_cols,cat_cols=self.separate_columns(train_df)
            train_df=self.handle_missing_values(train_df,num_cols,cat_cols)
            test_df=self.handle_missing_values(test_df,num_cols,cat_cols)

            cat_col_path=self.data_transformation_config.categorical_columns_file_path
            os.makedirs(os.path.dirname(cat_col_path),exist_ok=True)
            save_object(cat_col_path,cat_cols)
            logging.info(f"Saved categorical column list with {len(cat_cols)} features.")

            save_object(self.data_transformation_config.imputer_object_file_path,imputer_object)

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path),exist_ok=True)
            train_df.to_csv(self.data_transformation_config.transformed_train_file_path,index=False)
            test_df.to_csv(self.data_transformation_config.transformed_test_file_path,index=False)

            datatransformationartifact=DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                categorical_columns_path=cat_col_path,
                imputer_object_file_path=self.data_transformation_config.imputer_object_file_path,
            )
            logging.info("Data Transformation successfully completed")
            return datatransformationartifact
        except Exception as e:
            raise HealthAndFitnessException(e,sys)
        





