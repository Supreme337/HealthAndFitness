import os
import sys
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from healthandfitness.exception.exception import HealthAndFitnessException
from healthandfitness.constant.training_pipeline import TARGET_COLUMN
from healthandfitness.logging.logger import logging
from sklearn.metrics import accuracy_score
from healthandfitness.entity.config_entity import ModelTrainerConfig
from healthandfitness.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from healthandfitness.utils.main_utils.utils import load_object,save_object

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=ModelTrainerConfig
            self.data_transformation_artifact:DataTransformationArtifact
        except Exception as e:
            raise HealthAndFitnessException(e,sys)

    def train_model(self,x_train,y_train,cat_cols):
        try:
            model=CatBoostClassifier(iterations=500,learning_rate=0.05,depth=8,cat_features=cat_cols,verbose=False)
            model.fit(x_train,y_train)
            return model
        except Exception as e:
            raise HealthAndFitnessException(e,sys)
    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info("Model Trainer Initiated...")

            train_df=pd.read_csv(self.data_transformation_artifact.transformed_train_file_path)
            test_df=pd.read_csv(self.data_transformation_artifact.transformed_test_file_path)
        
            cat_cols=load_object(self.data_transformation_artifact.categorical_columns_file_path)
            target_column=TARGET_COLUMN

            x_train=train_df.drop(TARGET_COLUMN)
            y_train=train_df[TARGET_COLUMN]
        
            x_test=test_df.drop(TARGET_COLUMN)
            y_test=test_df[TARGET_COLUMN]

            logging.info("Training CatBoost model...")
            model=self.train_model(x_train,y_train,cat_cols)

            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            train_score=accuracy_score(y_train,y_train_pred)
            test_score=accuracy_score(y_test,y_test_pred)

            logging.info(f"Train R2 Score:{train_score}")
            logging.info(f"Test R2 Score:{test_score}")

            if(train_score-test_score)>self.model_trainer_config.overfitting_threshold:
                logging.warning("Model may be overfitting")
        
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path),exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path,model)
            logging.info(f"Saved trained model at:{self.model_trainer_config.trained_model_file_path}")

            modeltrainerartifact=ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_score=train_score,
                test_score=test_score
            )
            logging.info("Model Training completed successfully.")
            return modeltrainerartifact
        except Exception as e:
            raise HealthAndFitnessException(e,sys)

