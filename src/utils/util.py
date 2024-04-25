import os, sys
import numpy as np
import pandas as pd
import pickle 
from src.exception.exception import CustomException
from src.logger.log import logging
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#function to save data transformation
def save_object(file_path, obj):
    try : 
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump (obj, file_obj)
            
            
    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    try :
        
        with open(file_path,"rb") as file_object:
            return pickle.load(file_object)
     
    except Exception as e:
            raise CustomException (e,sys)   
    


#function to evaluate models
#eval_models(X_train,y_train,X_test,y_test,model,params)
def eval_models(X_train,y_train,X_test,y_test,models,params):
    try:
        #creating dictionary for report 
        report={}
        logging.info("Report creation started !! ")
        for i in range (len(list(models))):
            model=list(models.values())[i]
            para= params[list(models.keys())[i]]
            #RSCV=RandomizedSearchCV(estimator=model,
            #                       param_distributions=para,
            #                       n_iter=7,
            ##                       cv=3,
             #                      verbose=2,
            #                       n_jobs=-1)
            #
            #RSCV.fit(X_train, y_train)
            logging.info("Grid search cv started !! ")
            GS = GridSearchCV(model, para, cv = 5)
            logging.info("Grid search cv fitting started !! ")
            GS.fit(X_train, y_train)

            logging.info("fitting completed !! ")
           # model.set_params(**RSCV.best_params_)
            model.set_params(**GS.best_params_)
            model.fit(X_train, y_train)
                
            #make predictions
            y_pred= model.predict(X_test)
            test_model_accuracy = accuracy_score(y_test,y_pred)
            
            #calculate report in list format
            report[list(models.values())[i]] = test_model_accuracy
            
            return report
            
        
    except Exception as e:
            raise CustomException (e,sys)