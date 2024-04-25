import os, sys
from src.logger.log import logging
from src.exception.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils.util import save_object
from src.utils.util import eval_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts/pkls','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def inititate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting our data into dependent and independent features")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            model = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                "logistic Regression": {
                    'C': [0.1, 1, 10],
                    'max_iter': [100, 200, 300]
                }
               
            }

            model_report: dict = eval_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                models=model, params=params)

            # To get the best model from our report Dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = model[best_model_name]

            print(f"Best Model Found, Model Name is: {best_model_name}, Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"Best model found, Model Name is {best_model_name}, accuracy Score: {best_model_score}")

            save_object(file_path=self.model_trainer_config.train_model_file_path,
                        obj=best_model
                        )

        except Exception as e:
            raise CustomException(e, sys)     


