from altair import param
import streamlit as st
import numpy as np 
import pandas as pd 
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="ML App",
   
)


st.title('Automation ML Classification App')

st.write('Explore ML algorithms')


data_set = st.selectbox("select dataset",("diabetes","Breast cancer","Wine"))
Algorithm_name = st.selectbox("select ML algorithm",("SVM","Logistic Regression","Random forest"))

def get_dataset(data_set):
  if data_set == "diabetes":
    df = datasets.load_diabetes()
  elif data_set == "Breast cancer" : 
    df = datasets.load_breast_cancer()
  else :
    df = datasets.load_wine()
  x = df.data 
  y = df.target
  return x,y 

x,y = get_dataset(data_set)
st.write("shape of data : ",x.shape)


def add_parameter_ui(Algorithm_name):
    params = dict()
    
    if Algorithm_name == "SVM" :
        C = st.slider("C Parameter Value",0.01,10.0)
        params["C"]=C
    
    elif Algorithm_name ==  "Random forest" : 
        max_depth = st.slider("Max depth",2,15)
        n_estimators = st.slider("n_estimators", 1, 100)
        params["max_depth"]= max_depth
        params["n_estimators"]= n_estimators
        
        
    return params 

params= add_parameter_ui(Algorithm_name)

def get_classifier(algo_name , params) :
    if algo_name == "SVM" :
        classifier = SVC(C=params["C"])
    elif algo_name == "Logistic Regression" :
        classifier = LogisticRegression()
    else :
        classifier = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"])
    
    return classifier 

classifier = get_classifier(Algorithm_name,params)

#classify
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.8)

classifier.fit(X_train,y_train)
y_predicted = classifier.predict(X_test)
accuracy = accuracy_score(y_test,y_predicted)

st.write("accuracy : ",accuracy)
    
