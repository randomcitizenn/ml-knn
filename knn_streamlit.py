#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 08:51:40 2022

@author: dhruwin
"""

import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Loading our final trained KNn model
model = open("Knn_Classifier.pkl","rb")
knn_clf=joblib.load(model)

st.title("Iris flower species Classification App")
st.sidebar.title("Features")

# Loading Images
setosa= Image.open("setosa.jpg")
versicolor= Image.open("versicolor.jpg")
virginica= Image.open("virginica.jpg")

# Initializing
parameter_list = ['Sepal Length (cm)','Sepal Width (cm)','Petal Length (cm)','Petal Width (cm)']
parameter_input_values=[]
parameter_default_values=['5.2','3.2','4.2','1.2']
values=[]

# Display
for parameter, parameter_df in zip(parameter_list, parameter_default_values):
    values = st.sidebar.slider(label=parameter, key=parameter, value=float(parameter_df),min_value=0.0,max_value=8.0,step=0.1)
    parameter_input_values.append(values)
    
input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list, dtype=float)

st.write(input_variables)

prediction=0
if st.button("Click here to Classify"):
    prediction = knn_clf.predict(input_variables)
    
if prediction==0:
    st.image(setosa) 
elif prediction==1:
    st.image(versicolor) 
else:
    st.image(virginica)
    
    