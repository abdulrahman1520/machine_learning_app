import streamlit as st
import numpy as np 
import pandas as pd 
from sklearn import datasets

st.title('Automation ML App')

st.write('Explore ML algorithms')

data_set = st.selectbox("select dataset",("heart attack","Breast cancer","iris"))
Algorithm_name = st.selectbox("select ML algorithm",("SVC","Logistic Regression","Random forest"))
                              

