## Import Libraries
import streamlit as st
import joblib
import numpy as np
from utils import process_new


## Load the model
model_lgb = joblib.load('model.pkl')



def churn_classification():

    ## Title
    st.title('Churn Classification Prediction ....')
    st.markdown('<hr>', unsafe_allow_html=True)

    ## Choose Model
    model_type = st.selectbox('Choose the Model', options=['lgb'])

    ## Input fields
    # 'ID','AB', 'AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ', 'BC', 'BD ', 'BN',
    #    'BP', 'BQ', 'BR', 'BZ', 'CB', 'CC', 'CD ', 'CF', 'CH', 'CL', 'CR', 'CS',
    #    'CU', 'CW ', 'DA', 'DE', 'DF', 'DH', 'DI', 'DL', 'DN', 'DU', 'DV', 'DY',
    #    'EB', 'EE', 'EG', 'EH', 'EJ', 'EL', 'EP', 'EU', 'FC', 'FD ', 'FE', 'FI',
    #    'FL', 'FR', 'FS', 'GB', 'GE', 'GF', 'GH', 'GI', 'GL'
    Id = st.number_input('ID')
    AB = st.number_input('AB')
    AF = st.number_input('AF')
    AH = st.number_input('AH')
    AM = st.number_input('AM')
    AR = st.number_input('AR')
    AX = st.number_input('AX')
    AY = st.number_input('AY')
    AZ = st.number_input('AZ')
    BC = st.number_input('BC')
    BD = st.number_input('BD')
    BN = st.number_input('BN')
    BP = st.number_input('BP')
    BQ = st.number_input('BQ')
    BR = st.number_input('BR')
    BZ = st.number_input('BZ')
    CB = st.number_input('CB')
    CC = st.number_input('CC')
    CD = st.number_input('CD')
    CF = st.number_input('CF')
    CH = st.number_input('CH')
    CL = st.number_input('CL')
    CR = st.number_input('CR')
    CS = st.number_input('CS')
    CU = st.number_input('CU')
    CW = st.number_input('CW')
    DA = st.number_input('DA')
    DE = st.number_input('DE')
    DF = st.number_input('DF')
    DH = st.number_input('DH')
    DI = st.number_input('DI')
    DL = st.number_input('DL')
    DN = st.number_input('DN')
    DU = st.number_input('DU')
    DV = st.number_input('DV')
    DY = st.number_input('DY')
    EB = st.number_input('EB')
    EE = st.number_input('EE')
    EG = st.number_input('EG')
    EH = st.number_input('EH')
    EJ = st.number_input('EJ')
    EL = st.number_input('EL')
    EP = st.number_input('EP')
    EU = st.number_input('EU')
    FC = st.number_input('FC')
    FD = st.number_input('FD')
    FE = st.number_input('FE')
    FI = st.number_input('FI')
    FL = st.number_input('FL')
    FR = st.number_input('FR')
    FS = st.number_input('FS')
    GB = st.number_input('GB')
    GE = st.number_input('GE')
    GF = st.number_input('GF')
    GH = st.number_input('GH')
    GI = st.number_input('GI')
    GL = st.number_input('GL')
    Epsilon = st.number_input('Epsilon')
    st.markdown('<hr>', unsafe_allow_html=True)


    if st.button('Predict Churn ...'):

        ## Concatenate the users data
        new_data = np.array([Id,AB, AF, AH, AM, AR, AX, AY, AZ, BC, BD, BN, BP, BQ, BR, BZ, CB, CC, CD, CF, CH, CL, CR, CS, CU, CW, DA, DE, DF, DH, DI, DL, DN, DU, DV, DY, EB, EE, EG, EH, EJ, EL, EP, EU, FC, FD, FE, FI, FL, FR, FS, GB, GE, GF, GH, GI, GL,Epsilon])
        
        ## Call the function from utils.py to apply the pipeline
        X_processed = process_new(X_new=new_data)

        ## Predict using Model
        if model_type == 'RF':
            y_pred = model_lgb.predict(X_processed)[0]


        y_pred = bool(y_pred)

        ## Display Results
        st.success(f'Churn Prediction is ... {y_pred}')



if __name__ == '__main__':
    ## Call the function
    churn_classification()

