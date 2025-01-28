from flask import Flask,render_template,jsonify,request
import numpy as np
import pickle,joblib

flask_app=Flask(__name__)

model=joblib.load('Models/model.pkl')
#Mapping numerical disease to disease result
disease_mapping={
    1:'Heart Disease',
    0:'No Heart Disease(Normal)'
}
@flask_app.route('/')
def home():
    return render_template("index.html")
   
    
@flask_app.route("/predict",methods=["POST"])
def predict():
    #Get data from the form request
    float_features=[float(x) for x in request.form.values()]
    #Array for prediction
    features=np.array(float_features)
    #Predicting from model
    prediction=model.predict([features])
    #Mapping the final predicted answer
    predicted_disease=disease_mapping.get(prediction[0],"Unknown Disease")
    return render_template("index.html",prediction_text="{}".format(predicted_disease))



if __name__=="__main__":
    flask_app.run(debug=True)
    