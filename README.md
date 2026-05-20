## Health & Fitness Classification System

A Machine Learning-based classification project designed to predict and analyze health and fitness outcomes using structured health-related datasets and advanced classification algorithms.

## Overview

The **Health & Fitness Classification System** is an end-to-end machine learning project focused on predicting fitness or health-related categories based on user and lifestyle data.

The project demonstrates the complete ML workflow including:

* Data ingestion
* Data validation
* Data transformation
* Feature engineering
* Model training
* Model evaluation
* Prediction pipeline deployment

The system is built with a modular and scalable architecture to support experimentation and future enhancements.

## Features

*  Health and fitness dataset analysis
*  Automated ML pipeline architecture
*  Data preprocessing and transformation
*  Data validation and schema checks
*  Multiple classification model training
*  Model evaluation and performance comparison
*  Feature engineering and selection
*  FastAPI backend
*  Experiment tracking and modular workflow

##  Tech Stack

### Languages & Frameworks

* Python
* FastAPI

### Machine Learning

* Scikit-learn
* CatBoost
* Pandas
* NumPy

### Visualization & Analysis

* Matplotlib
* Seaborn

### MLOps & Utilities

* DagsHub
* Joblib


##  Machine Learning Pipeline

### 1. Data Ingestion

* Reads and imports raw datasets
* Handles train-test splitting
* Stores datasets for further processing

### 2. Data Validation

* Validates dataset schema and structure
* Checks missing values and duplicates
* Ensures correct feature formats
* Detects inconsistent or corrupted data
* Prevents invalid data from entering the pipeline

### 3. Data Transformation

* Handles categorical encoding
* Performs feature scaling and normalization
* Cleans and preprocesses data
* Applies feature engineering techniques

### 4. Model Training

* Trains multiple classification models
* Performs hyperparameter tuning
* Selects the best-performing model
* Saves trained model artifacts

### 5. Prediction Pipeline

* Accepts user input data
* Processes transformed features
* Generates classification predictions

##  Model Performance

This project uses CatBoost Classifier to reduce preprocessing overhead for categorical columns.

### Evaluation Metrics

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix

##  Web Application

A FastAPI interface is integrated for:

* Inputting health-related features
* Running model predictions
* Displaying classification results

##  Screenshots
<img width="1431" height="735" alt="Screenshot 2026-05-20 at 2 12 23 PM" src="https://github.com/user-attachments/assets/af5e929b-366c-4b5b-bd05-096b22bf6c6e" />
<img width="1431" height="735" alt="Screenshot 2026-05-20 at 2 13 46 PM" src="https://github.com/user-attachments/assets/f6197e3b-9cd0-48a3-9f4c-8a52fa91cefd" />
<img width="1431" height="735" alt="Screenshot 2026-05-20 at 2 25 27 PM" src="https://github.com/user-attachments/assets/0062c065-bb87-4405-8b07-f626d7dd7cdd" />
<img width="1431" height="735" alt="Screenshot 2026-05-20 at 2 25 49 PM" src="https://github.com/user-attachments/assets/7803793a-5e92-412b-b374-c5da07aa8abe" />
<img width="1431" height="735" alt="Screenshot 2026-05-20 at 2 26 55 PM" src="https://github.com/user-attachments/assets/cd0c7429-ce4b-40d1-b93a-3563b950ce71" />

##  License

This project is licensed under the MIT License.

##  Author

Developed by Harsh Malik
