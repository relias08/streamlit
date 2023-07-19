# For complete details on how to run predictions in Streamlit by using either the API end-point from 
# TF Serving/ Torch Serve (as done in this file) OR by using model.predict(), see the following file in laptop:
# [204__how_to_run_predictions_in_Streamlit___super_stuff.ipynb]
# (C:\Users\hduser\Desktop\_ALL_MAIN_WDS_03_07_2020\__Deploying_Models___MAIN\___Streamlit_and_mlflow\worked___mitichondria___Sreeni_204)


import streamlit as st
import requests
import numpy as np

# just trying to create a multi-page thing
condition = st.sidebar.selectbox(
    "Select the visualization",
    ("Introduction", "EDA", "Model Prediction", "Model Evaluation")
)

st.title("LazyP's Linear Regression Model")

# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
#passengerid = st.text_input("Input Passenger ID", '123456') 
#pclass = st.selectbox("Choose class", [1,2,3])
#name  = st.text_input("Input Passenger Name", 'John Smith')
#sex = st.select_slider("Choose sex", ['male','female'])
#age = st.slider("Choose age",0,100)
#sibsp = st.slider("Choose siblings",0,10)
#parch = st.slider("Choose parch",0,2)
#ticket = st.text_input("Input Ticket Number", "12345") 
#cabin = st.text_input("Input Cabin", "C52") 
#embarked = st.select_slider("Did they Embark?", ['S','C','Q'])

x1 = st.number_input("x1", 0, 1000)
x2 = st.number_input("x2", 0, 1000)
x3 = st.number_input("x3", 0, 1000)


import json
import requests
import numpy as np

def predict():               # this function is run when we click on the 'Predict' button below
    DEPLOYED_ENDPOINT = "http://localhost:8601/v1/models/lr_model:predict"
    data = json.dumps({"instances": [[x1, x2, x3]]})    
    
    r = requests.post(url = DEPLOYED_ENDPOINT, data=data)
    st.write(r)

    prediction = r.json()       # r.json() returns a regular dictionary

    if type(prediction['predictions'][0][0]) == np.float: 
        st.success(prediction['predictions'][0][0])           # note the st.success()
    else: 
        st.error('error')                                     # note the st.error()
   
st.button('Predict', on_click=predict)

