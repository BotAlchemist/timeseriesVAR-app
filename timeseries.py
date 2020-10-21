# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 07:18:42 2020

@author: Sumit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error
import base64
st.set_option('deprecation.showPyplotGlobalUse', False)

global df, uploaded_file
df= pd.DataFrame(index=[0])

st.title('Time Series Forecasting using VAR')
st.subheader('''Developed by Sumit Srivastava (*sumvast@gmail.com*)''')



st.sidebar.header('Upload CSV file')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


def run_auto_var(data, n):
    best_aic = np.inf
    best_p = None
    tmp_model = None
    best_mdl = None
    for p in range(n+1):
        try:
            
            model = VAR(data)
            model_fit = model.fit(p)
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_p = p
        except:
            continue
    return best_p, best_aic


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="VAR_forecast.csv">Download Result</a>'
    return href

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    
    date_col= df.columns[0]
    val_col= df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    
    df['Week_name'] = df[date_col].dt.week
    df['Month_name'] = df[date_col].dt.month
    df['Quarter_name'] = df[date_col].dt.quarter
    
    
    df = df.set_index(date_col)
    
    st.markdown("### Dataset")
    st.write(df[[val_col]])
    st.markdown("### Plot")
    st.line_chart(df[val_col])
    
    
    st.markdown("### Correlation")
    sn.heatmap(df.corr(), annot=True)
    st.pyplot()
    
    ########################## Train-Test split
    split_value= int(0.9*len(df))
    train_df= df.iloc[:split_value]
    test_df= df.iloc[split_value:]
    
    st.markdown("## Train-Test split into 90:10 ratio")
    st.markdown("### Length of Training set: "+ str(len(train_df)))
    st.markdown("### Length of Test set: "+ str(len(test_df)))
    
    best_p, best_aic= run_auto_var(train_df, 12)
    model= VAR(train_df)
    model_fit= model.fit(best_p)
    
    st.markdown("### Best P value: "+ str(best_p))
    
    prediction= model_fit.forecast(model_fit.y, steps=len(test_df))
    var_prediction= pd.DataFrame(prediction, columns=['Prediction', 'Var1', 'Var2', 'Var3'])
    var_prediction= var_prediction['Prediction'].values
    
    test_df['Prediction']= var_prediction
    st.markdown("### Plot Actual vs Predicted")
    st.line_chart(test_df[['Prediction', val_col]])
    
    st.markdown("### Dataset Actual vs Predicted")
    st.write(test_df[[val_col, 'Prediction']])
    
    MSE= mean_squared_error(test_df[val_col], test_df['Prediction'])
    RMSE= MSE**0.5
    st.markdown("#### Root Means Squred Error: "+ str(round(RMSE,2)))
    
    
    
    st.markdown("### Overall graph")
    train_df= train_df.append(test_df)
    st.line_chart(train_df[['Prediction', val_col]])
    
    train_df= train_df.reset_index()
    train_df= train_df[[date_col, val_col, 'Prediction']]
    st.markdown(filedownload(train_df), unsafe_allow_html=True)
    
    
    
else:
    st.info('Awaiting CSV file to be uploaded. ')
    
    


    
    
    
    



    

