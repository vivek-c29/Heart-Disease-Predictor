import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
#Load model
try:
    model=joblib.load(r'Models/model.pkl')
    st.write('Model Loaded successfully')
except Exception as e:
    st.write(f"Error loading model: {e}")
    st.stop()

disease_mapping={
    0:'No heart disease(Normal)',
    1:'Heart Disease'
}

def main():
    # st.title('Heart Disease Prediction')
    html_temp="""
    <div style="background-color:#167fae;padding:10px;border-radius:20px;">
    <h2 style="color:black;text-align:center;">Streamlit Heart Disease Predictor</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age=st.number_input('Age',min_value=0,max_value=100)
    sex=st.selectbox('Sex',['Male','Female'])
    cp=st.selectbox('Chest pain type',['Normal','Stable','Mild','Unstable'])
    trestbps=st.number_input('Resting Blood Pressure',min_value=75,max_value=220)
    chol=st.number_input('Serum Cholestrol(in mg/dl)',min_value=125,max_value=280)
    thalach=st.number_input('Maximum heart rate',min_value=70,max_value=300)
    oldpeak=st.number_input('ST Depression',min_value=0.0,max_value=7.0)
    ca=st.selectbox('No. of major vessels colored by Fluoroscopy',[0,1,2,3,4])
    thal=st.selectbox('Thalasemmia',['Normal','Fixed defect','Reversable defect','Abnormal'])
    
    data=pd.DataFrame({
        'age':[age],
        'sex': [1 if sex == 'Male' else 0],
        'cp':[cp],
        'trestbps':[trestbps],
        'chol':[chol],
        'thalach':[thalach],
        'oldpeak':[oldpeak],
        'ca':[ca],
        'thal':[thal]
    })
    
    cp_mapping = {'Normal': 0, 'Stable': 1, 'Mild': 2, 'Unstable': 3}
    thal_mapping = {'Normal': 0, 'Fixed defect': 1, 'Reversable defect': 2, 'Abnormal': 3}
    data['cp'] = data['cp'].map(cp_mapping)
    data['thal'] = data['thal'].map(thal_mapping)
    
    st.write("Input Data for prediction")
    st.write(data)
    
    scaler=StandardScaler()
    colums_to_scale=['trestbps','chol','thalach']
    data[colums_to_scale]=scaler.fit_transform(data[colums_to_scale])
    
    if st.button('Predict'):
        try:
            prediction=model.predict(data)
            probabilites=model.predict_proba(data)
            
            predicted_class=prediction[0]
            confidence=probabilites[0][predicted_class]
            disease_category=disease_mapping.get(prediction[0],'Unknown Disease Category')
            st.write(f"Predicted Disease Category : {disease_category}")
            # st.write(f"Confidence Level : {confidence:.2f}")
        except Exception as e:
            st.write(f"Error occured during prediction:{e}")
            
            
    
if __name__=="__main__":
    main()