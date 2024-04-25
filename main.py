import os, sys
from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import PredictionPipeline, Customclass
from src.logger.log import logging
from src.pipeline.batch_prediction import batch_prediction
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.pipeline.training_pipeline import Training_Pipeline
from werkzeug.utils import  secure_filename


UPLOAD_FOLDER = 'batch_prediction/Uploaded_CSV_FILE'
model_file_path ='artifacts/pkls/model.pkl'
transformer_file_path='artifacts/pkls/preprocessor.pkl'

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'csv'}

@app.route('/')
def home_page():
    logging.info("Main page is loaded !!")
    return render_template('Index.html')

#Single Prediction Option

@app.route("/Single_pred", methods= ["GET","POST"])

def predict_data():
    if request.method == "GET":
        logging.info("Single Prediction page is loaded !!") 
        return render_template("Single_pred.html")
    
  
    
    else:
        data = Customclass(
                Cycle=int(request.form.get("Cycle")),
                OpSet1=float(request.form.get("OpSet1")),#loyal / disloyal
                OpSet2 =float(request.form.get("OpSet2")),
                OpSet3=float(request.form.get("OpSet3")), #personal/business
                SensorMeasure1=float(request.form.get("SensorMeasure1")), #business/eco/eco plus
                SensorMeasure2=float(request.form.get("SensorMeasure2")),
                SensorMeasure3= float(request.form.get("SensorMeasure3")),
                SensorMeasure4=float(request.form.get("SensorMeasure4")), 
                SensorMeasure5=float(request.form.get("SensorMeasure5")),
                SensorMeasure6=float(request.form.get("SensorMeasure6")), 
                SensorMeasure7=float(request.form.get("SensorMeasure7")),
                SensorMeasure8=float(request.form.get("SensorMeasure8")),
                SensorMeasure9=float(request.form.get("SensorMeasure9")),
                SensorMeasure10=float(request.form.get("SensorMeasure10")),
                SensorMeasure11=float(request.form.get("SensorMeasure11")),
                SensorMeasure12=float(request.form.get("SensorMeasure12")),
                SensorMeasure13=float(request.form.get("SensorMeasure13")),
                SensorMeasure14=float(request.form.get("SensorMeasure14")),
                SensorMeasure15=float(request.form.get("SensorMeasure15")),
                SensorMeasure16=float(request.form.get("SensorMeasure16")),
                SensorMeasure17=float(request.form.get("SensorMeasure17")),
                SensorMeasure18=float(request.form.get("SensorMeasure18")),
                SensorMeasure19=float(request.form.get("SensorMeasure19")),
                SensorMeasure20=float(request.form.get("SensorMeasure20")),
                SensorMeasure21=float(request.form.get("SensorMeasure21")),
                Life_Ratio=float(request.form.get("Life_Ratio"))
                #Arrival_Delay_in_Minutes=int(request.form.get("Arrival_Delay_in_Minutes"))
            
            )
        
    final_data= data.get_data_into_DataFrame()  
    logging.info("Sending data to pred pipeline !!")
    Pred_Pipeline = PredictionPipeline()
    pred= Pred_Pipeline.predict(final_data)
    
    result = pred 
    logging.info("Got the result  !!")
    if result == 0:
        logging.info(f"The Engine is in {result} !!")
        return render_template("result.html",final_result= "The Engine is in Good Condition:{}".format(result))
    
    elif result == 1:
        logging.info(f"The Engine is in  {result} !!")
        return render_template("result.html",final_result= "The Engine is in Moderate Condition:{}".format(result))
    else:
        logging.info(f"The Engine is in  {result} !!")
        return render_template("result.html",final_result= "Warning! he Engine is in Bad Condition:{}".format(result))
        
    
#Batch Prediction Option

@app.route("/batch_pred", methods=['GET','POST'])
def perform_batch_prediction():
    
    
    if request.method == 'GET':
        return render_template('batch_pred.html')
    else:
        file = request.files['csv_file']  # Update the key to 'csv_file'
        # Directory path
        directory_path = UPLOAD_FOLDER
        # Create the directory
        os.makedirs(directory_path, exist_ok=True)

 # Check if the file has a valid extension
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            # Delete all files in the file path
            for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Save the new file to the uploads directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print(file_path)

            logging.info("CSV received and Uploaded")    

            # Perform batch prediction using the uploaded file
            batch = batch_prediction(file_path,
                                    model_file_path,
                                    transformer_file_path
                                    )
            pred_path=batch.start_batch_prediction()
            new_line = '\n'
            output = f"Batch Prediction Done!!!. '{new_line}' Please find the predictions.csv at : '{new_line}'    '{pred_path}'"
            return render_template("result.html", final_result=output, prediction_type='batch')
        else:
            return render_template('result.html', prediction_type='batch', error='Invalid file type')
        

# Train Model Option


@app.route('/train', methods=['GET', 'POST'])
def train():
    #if request.method == 'GET':
     #   return render_template('result.html')
   # else:
        try:
            pipeline = Training_Pipeline()
            #pipeline.main()
            print ("Training Completed!!")
            output = f"Training pipeline completed!!."
            return render_template("result.html", final_result=output, prediction_type='train')
            

        except Exception as e:
            logging.error(f"{e}")
            error_message = str(e)
            return render_template("result.html", final_result=error_message, prediction_type='train')
           # return render_template('Index.html', error=error_message)
        

if __name__ == "__main__":
     app.run( debug=True)
     
     # host = "0.0.0.0.",