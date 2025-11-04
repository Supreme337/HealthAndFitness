import os
import sys
import pandas as pd
import numpy as np
import tempfile
from urllib.parse import urlparse
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from healthandfitness.exception.exception import HealthAndFitnessException
from healthandfitness.constant.training_pipeline import TARGET_COLUMN
from healthandfitness.logging.logger import logging
from sklearn.metrics import accuracy_score, confusion_matrix
from healthandfitness.utils.ml_utils.classification_metric import get_classification_score
from healthandfitness.entity.config_entity import ModelTrainerConfig
from healthandfitness.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from healthandfitness.utils.main_utils.utils import load_object,save_object
import mlflow
import dagshub
dagshub.init(repo_owner="Supreme337",repo_name="HealthAndFitness",mlflow=True)

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config:ModelTrainerConfig=model_trainer_config
            self.data_transformation_artifact:DataTransformationArtifact=data_transformation_artifact
        except Exception as e:
            raise HealthAndFitnessException(e,sys)

    def train_model(self,x_train,y_train,cat_cols):
        try:
            model=CatBoostClassifier(iterations=500,learning_rate=0.05,depth=8,cat_features=cat_cols,verbose=False)
            if TARGET_COLUMN in cat_cols:
                cat_cols.remove(TARGET_COLUMN)
            model.fit(x_train,y_train)
            return model
        except Exception as e:
            raise HealthAndFitnessException(e,sys)
    
    def log_mlflow(self,model,x_test,y_test,train_acc,test_acc):
        with mlflow.start_run(run_name="Catboost_Trainer"):
            mlflow.log_param("iterations",500)
            mlflow.log_param("learning_rate",0.05)
            mlflow.log_param("depth",8)
            mlflow.log_metric("train_accuracy",train_acc)
            mlflow.log_metric("test_accuracy",test_acc)
            
            y_pred=model.predict(x_test)
            cm=confusion_matrix(y_test,y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")

            with tempfile.TemporaryDirectory() as temp_dir:
                cm_path=os.path.join(temp_dir,"confusion_matrix.png")
                plt.savefig(cm_path)
                mlflow.log_artifact(cm_path,"confusion_matrix")
            plt.close()

            tracking_uri=mlflow.get_tracking_uri()
            tracking_scheme=urlparse(tracking_uri).scheme

            if "dagshub" in tracking_uri.lower() or tracking_scheme in ["http","https"]:
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path=os.path.join(tmpdir,"model.pkl")
                    joblib.dump(model,model_path)
                    mlflow.log_artifact(model_path,artifact_path="model")
                logging.info("Model logged as an artifact (DagsHub-compatible).")
            else:
                mlflow.catboost.log_model(model,"model")
                logging.info("Model logged as catboost model (local MLflow).")

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info("Model Trainer Initiated...")

            train_df=pd.read_csv(self.data_transformation_artifact.transformed_train_file_path)
            test_df=pd.read_csv(self.data_transformation_artifact.transformed_test_file_path)
        
            cat_cols=load_object(self.data_transformation_artifact.categorical_columns_file_path)
            target_column=TARGET_COLUMN

            x_train=train_df.drop([TARGET_COLUMN],axis=1)
            y_train=train_df[TARGET_COLUMN]
        
            x_test=test_df.drop([TARGET_COLUMN],axis=1)
            y_test=test_df[TARGET_COLUMN]

            logging.info("Training CatBoost model...")
            model=self.train_model(x_train,y_train,cat_cols)

            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            train_metrics=get_classification_score(y_train,y_train_pred)
            test_metrics=get_classification_score(y_test,y_test_pred)

            logging.info(f"Train Accuracy Score:{train_metrics.accuracy_score}")
            logging.info(f"Test Accuracy Score:{test_metrics.accuracy_score}")

            self.log_mlflow(model,x_test,y_test,train_metrics.accuracy_score,test_metrics.accuracy_score)

            if(train_metrics.accuracy_score-test_metrics.accuracy_score)>self.model_trainer_config.overfitting_threshold:
                logging.warning("Model may be overfitting")
        
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path),exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path,model)
            logging.info(f"Saved trained model at:{self.model_trainer_config.trained_model_file_path}")

            modeltrainerartifact=ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_score=train_metrics.accuracy_score,
                test_score=test_metrics.accuracy_score
            )
            logging.info("Model Training completed successfully.")
            return modeltrainerartifact
        except Exception as e:
            raise HealthAndFitnessException(e,sys)

