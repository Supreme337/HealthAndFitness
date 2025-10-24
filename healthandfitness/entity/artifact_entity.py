from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_train_file_path:str
    invalid_test_file_path:str
    drift_report_file_path:str
    validation_status:bool

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str
    transformed_test_file_path:str
    transformed_object_file_path:str
    imputer_object_file_path:str
    categorical_columns_file_path:str

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    train_score:float
    test_score:float