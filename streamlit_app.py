import streamlit as st
import numpy as np 
import pandas as pd 
from sklearn import datasets

st.title('Automation ML App')

st.write('Explore ML algorithms')

data_set = st.selectbox("select dataset",("diabetes","Breast cancer","iris"))
Algorithm_name = st.selectbox("select ML algorithm",("SVC","Logistic Regression","Random forest"))

def get_dataset(data_set):
  if data_set == "diabetes":
    df = datasets.load_diabetes()
  elif data_set == "Breast cancer" : 
    df = datasets.load_breast_cancer()
  else :
    df = datasets.load_iris 
  x = df.data 
  y = df.target
  return x,y 

x,y = get_dataset(data_set)
st.write("shape of data : ",x.shape)

