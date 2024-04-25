import os, sys
import pandas as pd
import numpy as np
from src.logger.log import logging
from src.exception.exception import CustomException
import pickle
from src.utils.util import load_object

PREDICTION_FOLDER='batch_Prediction'
PREDICTION_CSV='prediction_csv'
PREDICTION_FILE='prediction.csv'

FEATURE_ENG_FOLDER='batch_prediction'

ROOT_DIR=os.getcwd()
FEATURE_ENG=os.path.join(ROOT_DIR,PREDICTION_FOLDER,FEATURE_ENG_FOLDER)
BATCH_PREDICTION=os.path.join(ROOT_DIR,PREDICTION_FOLDER,PREDICTION_CSV)
BATCH_COLLECTION_PATH ='batch_prediction'

class batch_prediction:
    def __init__(self,input_file_path, 
                 model_file_path, 
                 transformer_file_path
                 ) -> None:   
        
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path

    def start_batch_prediction(self):
        try:
            logging.info("loading saved pipeline")
            
            # Loading the data transformation pipeline
            with open(self.transformer_file_path, 'rb') as f:
                preprocessor = pickle.load(f)

            logging.info(f"preprocessor  Object acessed:{self.transformer_file_path}")

            # Load the model 
            model =load_object(file_path=self.model_file_path)

            logging.info(f"Model File Path: {self.model_file_path}")

           
            # Read the input file
            df = pd.read_csv(self.input_file_path)

            #df.to_csv("df_if_satisfied_input_Data.csv")

            logging.info("Uploaded csv for batch prediction is transformed and saved !!")

             # Dropping target column
            
            df=df.drop('labels', axis=1)            
            #df.to_csv('dropped_satisfaction.csv')

            logging.info(f"Columns before transformation: {', '.join(f'{col}: {df[col].dtype}' for col in df.columns)}")

            # Transform the feature-engineered data using the preprocessor
            transformed_data = preprocessor.transform(df)
            logging.info(f"Transformed Data Shape: {transformed_data.shape}")


            predictions = model.predict(transformed_data)
            logging.info(f"Predictions done :{predictions}")

            # Create a DataFrame from the predictions array
            df_predictions = pd.DataFrame(predictions, columns=['prediction'])

             # Define the mapping dictionary                    
            def map_prediction(value):
                if value == 0:
                    return 'The Condition of the engine is Good'
                elif value == 1:
                    return 'The Condition of the engine is Moderate'
                else:
                    return 'Warning! The Condition of the engine is Bad'
            # Apply mapping function to the Prediction column
            df_predictions['prediction'] = df_predictions['prediction'].map(map_prediction)

            # Save the predictions to a CSV file
            BATCH_PREDICTION_PATH = BATCH_PREDICTION  # Specify the desired path for saving the CSV file
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION_PATH,'predictions.csv')
            df_predictions.to_csv(csv_path, index=False)
            logging.info(f"Batch predictions saved to  : '{csv_path}'.")
            return csv_path

        except Exception as e:
            raise CustomException(e,sys)    