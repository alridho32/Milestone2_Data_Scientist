import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load all files
with open('best_dt_model.pkl', 'rb') as file_1:
  model = pickle.load(file_1)

def run():
    # Membuat form
    with st.form(key='form parameters'):
        st.subheader('Fill Your Data')
        st.markdown('<hr style="border:2px solid orange">', unsafe_allow_html=True)
        
        RowNumber = st.number_input('Row Number',min_value=0, max_value=99999,value=0,step=1,help='Max Input (5 Digits Number)')
        CustomerId = st.number_input('Customer Id', min_value=0, max_value=99999999,value=0,step=1,help='Max Input (8 Digits Number)')
        Surname = st.text_input('Surname', value='<Enter Your Last Name>', help='Input Only Your Last Name')
        CreditScore = st.number_input('Credit Score', min_value=0, max_value=1000,value=0,step=1,help='Max Input (1000)')
        Tenure = st.number_input('Tenure', min_value=0, max_value=99, value=0,step=1,help='Max Input (2 Digits Number)')
        Balance = st.number_input('Balance', min_value=0, max_value=999999, value=0,step=1,help='Max Input (6 Digits Number)')
        NumOfProducts = st.number_input('Number Of Products', min_value=0, max_value=10, value=0,step=1,help='Max Input (10)')
        EstimatedSalary = st.number_input('EstimatedSalary', min_value=0, max_value=999999, value=0,step=1,help='Max Input (6 Digits Number)')
        st.markdown('---')
        Age = st.slider('Age',20,99)
        IsActiveMember = st.radio('Is an Active Member?', (0,1), help='No = 0, Yes = 1')
        HasCrCard = st.radio('Has a Credit Card?', (0,1), help='No = 0, Yes = 1')
        st.markdown('---')
        Geography = st.selectbox('Geography',('Germany','Spain','France'),index=1)
        Gender = st.selectbox('Gender',('Male','Female'), index=1)
        st.markdown('---')

        submitted = st.form_submit_button('Predict')

    data_inf = {
        'RowNumber': RowNumber,
        'CustomerId': CustomerId,
        'Surname': Surname,
        'CreditScore': CreditScore,
        'Geography': Geography,
        'Gender': Gender,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary
    }
    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        y_pred_inf = model.predict(data_inf)
        # Mengubah hasil prediksi menjadi label
        predicted_labels = ['Churn' if pred == 1 else 'Not Churn' for pred in y_pred_inf]
        # Menampilkan hasil prediksi
        data_inf['Prediction'] = predicted_labels
        st.write('# Churn Prediction :', str(predicted_labels))

if __name__ == '__main__':
    run()