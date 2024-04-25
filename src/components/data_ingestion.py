import os, sys
import pandas as pd
import numpy as np
from src.logger.log import logging
from src.exception.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import StratifiedShuffleSplit

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts/data_ingestion',"train.csv")
    test_data_path = os.path.join('artifacts/data_ingestion',"test.csv")
    raw_data_path = os.path.join('artifacts/data_ingestion',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try :
            df = pd.read_csv(os.path.join('notebooks\data',"Predictive_Maintanance_TurbojetEngine.csv"))            
            logging.info("Read the dataset as dataframe")            
                      
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False) 
            
            logging.info("Created raw_data_path csv")
            
            logging.info("Stratified train test split initiated")
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
            for train_index, test_index in split.split(df, df["labels"]):  # Replace "target_column" with the actual column name you are predicting
                train_set = df.loc[train_index]
                test_set = df.loc[test_index]
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion is done")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Error occurred in data ingestion stage")
            raise CustomException(e, sys)        