from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger.log import logging
from src.exception.exception import CustomException
from dataclasses import dataclass


#src\pipeline\training_pipeline.py

@dataclass
class Training_Pipeline:
    def __init__(self):
        self.c = 0 
        print(f"-----------{self.c}------------")
        print("Training pipeline started !!")

#if __name__=="__main__":
   
        logging.info("Data ingestion Pipeline is started!! ")
        obj=DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        logging.info("Data ingestion Pipeline is ended!! ")
        
        logging.info("*"*60)

        logging.info("Data transformation Pipeline is started!! ")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ =data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data transformation Pipeline is ended!! ")
        
        logging.info("*"*60)
        print("Training pipeline Ended !!")
         #creating obj for model trainer file
        logging.info("Model Training Pipeline is started!! ")
        modeltrainer=ModelTrainer()
        print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
        logging.info("Model Training Pipeline is ended!! ")

'''if __name__=='__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    logging.info("Data Ingestion Completed")
    print(train_data_path, test_data_path)

    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    logging.info("Data Transformation Completed")

    model_training = ModelTrainer()
    model_training.inititate_model_trainer(train_array, test_array)'''