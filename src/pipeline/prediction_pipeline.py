import os, sys
from src.logger.log import logging
from src.exception.exception import CustomException
import pandas as pd
import numpy as np

from src.utils.util import load_object


class PredictionPipeline:
    def __init__(self):
        pass
    # write helper function to load .pkl file in model_path in utils.py
    # function to load .pkl files
    def predict(self, features):
        preprocessor_path= os.path.join("artifacts/pkls","preprocessor.pkl")
        model_path= os.path.join ("artifacts/pkls","model.pkl")
        
        processor = load_object(preprocessor_path)
        model = load_object (model_path)
        logging.info("Models are loaded!!")
        
        logging.info("Sending data for scaling and transformation!!")
        scaled = processor.transform(features)
        logging.info("Feature transformation is done !!")
        
        logging.info("Sending data for prediction!!")
        pred = model.predict(scaled)
        logging.info(" Prediction is done!!")
        print(pred)
        return pred
    
    '''numerical_columns = ['Cycle', 'OpSet1', 'OpSet2', 'OpSet3', 'SensorMeasure1',
       'SensorMeasure2', 'SensorMeasure3', 'SensorMeasure4', 'SensorMeasure5',
       'SensorMeasure6', 'SensorMeasure7', 'SensorMeasure8', 'SensorMeasure9',
       'SensorMeasure10', 'SensorMeasure11', 'SensorMeasure12',
       'SensorMeasure13', 'SensorMeasure14', 'SensorMeasure15',
       'SensorMeasure16', 'SensorMeasure17', 'SensorMeasure18',
       'SensorMeasure19', 'SensorMeasure20', 'SensorMeasure21', 'Life_Ratio']'''    
    
class Customclass:
    def __init__(self,
                Cycle:int, 
                OpSet1:float, 
                OpSet2 :float,
                OpSet3:float, 
                SensorMeasure1:float,
                SensorMeasure2:float, 
                SensorMeasure3:float,
                SensorMeasure4:float, 
                SensorMeasure5:float,
                SensorMeasure6:float, 
                SensorMeasure7:float,
                SensorMeasure8:float,
                SensorMeasure9:float,
                SensorMeasure10:float,
                SensorMeasure11:float,
                SensorMeasure12:float,
                SensorMeasure13:float,
                SensorMeasure14:float,
                SensorMeasure15:float,
                SensorMeasure16:float,
                SensorMeasure17:float,
                SensorMeasure18:float,
                SensorMeasure19:float,
                SensorMeasure20:float,
                SensorMeasure21:float,
                Life_Ratio:float,
               ):  
            
        
            self.Cycle = Cycle
            self.OpSet1 = OpSet1 
            self.OpSet2 = OpSet2
            self.OpSet3 = OpSet3 
            self.SensorMeasure1 = SensorMeasure1
            self.SensorMeasure2 = SensorMeasure2
            self.SensorMeasure3 = SensorMeasure3
            self.SensorMeasure4 = SensorMeasure4
            self.SensorMeasure5 = SensorMeasure5
            self.SensorMeasure6 = SensorMeasure6 
            self.SensorMeasure7 = SensorMeasure7
            self.SensorMeasure8 = SensorMeasure8 
            self.SensorMeasure9 = SensorMeasure9
            self.SensorMeasure10 = SensorMeasure10
            self.SensorMeasure11 = SensorMeasure11
            self.SensorMeasure12 = SensorMeasure12
            self.SensorMeasure13 = SensorMeasure13
            self.SensorMeasure14 = SensorMeasure14
            self.SensorMeasure15 = SensorMeasure15
            self.SensorMeasure16 = SensorMeasure16
            self.SensorMeasure17 = SensorMeasure17
            self.SensorMeasure18 = SensorMeasure18
            self.SensorMeasure19 = SensorMeasure19
            self.SensorMeasure20 = SensorMeasure20
            self.SensorMeasure21 = SensorMeasure21
            self.Life_Ratio = Life_Ratio
            
    
    def get_data_into_DataFrame(self):
        try:
            custom_input ={
                'Cycle':[self.Cycle], 
                'OpSet1':[self.OpSet1], 
                'OpSet2':[self.OpSet2],
                'OpSet3':[self.OpSet3],
                'SensorMeasure1':[self.SensorMeasure1], 
                'SensorMeasure2':[self.SensorMeasure2],
                'SensorMeasure3' :[self.SensorMeasure3],
                'SensorMeasure4':[self.SensorMeasure4],
                'SensorMeasure5':[self.SensorMeasure5],
                'SensorMeasure6':[self.SensorMeasure6],
                'SensorMeasure7':[self.SensorMeasure7],
                'SensorMeasure8':[self.SensorMeasure8],
                'SensorMeasure9':[self.SensorMeasure9],
                'SensorMeasure10':[self.SensorMeasure10],
                'SensorMeasure11':[self.SensorMeasure11],
                'SensorMeasure12':[self.SensorMeasure12],
                'SensorMeasure13':[self.SensorMeasure13],
                'SensorMeasure14':[self.SensorMeasure14],
                'SensorMeasure15' :[self.SensorMeasure15],
                'SensorMeasure16':[self.SensorMeasure16],
                'SensorMeasure17':[self.SensorMeasure17],
                'SensorMeasure18':[self.SensorMeasure18],
                'SensorMeasure19':[self.SensorMeasure19],
                'SensorMeasure20':[self.SensorMeasure20],
                'SensorMeasure21':[self.SensorMeasure21],
                'Life_Ratio':[self.Life_Ratio],
                           
                }
            
            data=pd.DataFrame(custom_input)
            print(data)
            logging.info(" Data is entered in Dataframe !!")
            return data
        
        except Exception as e:
            raise CustomException (e,sys)    
