## main
import numpy as np
import pandas as pd
import os

## skelarn -- preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

df=pd.read_csv('train.csv')
## To Features and Target
X = df.drop(columns=['Class'], axis=1)
y = df['Class'].astype(np.int64)

## split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, shuffle=True, stratify=y)

## Slicing cols
num_cols_process = [X.coulmns]
categ_cols_process = X_train.select_dtypes(include='object').columns.tolist()
## The rest of columns does not need any thing
cols_ready = list(set(X_train.columns.tolist()) - set(num_cols_process) - set(categ_cols_process))


## Pipeline

## For Numerical: num_cols_process -------> Imputing, Standardization
## For Categorical: categ_cols_process ---> Imputing, OneHotEncoding
## For Cols without any needs ------------> Imputing only

## For Numerical: num_cols_process
num_pipeline = Pipeline(steps=[
                        ('selector', ColumnTransformer(num_cols_process)),
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ])

## For Categorical: categ_cols_process
categ_pipeline = Pipeline(steps=[
                        ('selector', ColumnTransformer(categ_cols_process)),
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
                    ])


## For ready cols
readycols_pipeline = Pipeline(steps=[
                        ('selector', ColumnTransformer(cols_ready)),
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                    ])

## Combine all
all_pipeline = FeatureUnion(transformer_list=[
                            ('numerical', num_pipeline),
                            ('categorical', categ_pipeline),
                            ('ready_cols', readycols_pipeline)
                        ])

## apply
_ = all_pipeline.fit(X_train)




def process_new(X_new):
    ''' This Function is to apply the pipeline to user data. Taking a list.
    
    Args:
    *****
        (X_new: List) --> The users input as a list.

    Returns:
    *******
        (X_processed: 2D numpy array) --> The processed numpy array of userf input.
    '''
    
    ## To DataFrame
    df_new = pd.DataFrame([X_new])
    df_new.columns = X_train.columns

    ## Adjust the Datatypes
    ## Feature Engineering is here ... 
    ## ...... ### 
    df_new['AB'] = df_new['AB'].astype(np.int64)
    df_new['AF'] = df_new['AF'].astype(np.int64)
    df_new['AH'] = df_new['AH'].astype(np.int64)
    df_new['AM'] = df_new['AM'].astype(np.int64)
    df_new['AR'] = df_new['AR'].astype(np.int64)
    df_new['AX'] = df_new['AX'].astype(np.int64)
    df_new['AY'] = df_new['AY'].astype(np.int64)
    df_new['AZ'] = df_new['AZ'].astype(np.int64)
    df_new['BC'] = df_new['BC'].astype(np.int64)
    df_new['BD'] = df_new['BD'].astype(np.int64)
    df_new['BN'] = df_new['BN'].astype(np.int64)
    df_new['BP'] = df_new['BP'].astype(np.int64)
    df_new['BQ'] = df_new['BQ'].astype(np.int64)
    df_new['BR'] = df_new['BR'].astype(np.int64)
    df_new['BZ'] = df_new['BZ'].astype(np.int64)
    df_new['CB'] = df_new['CB'].astype(np.int64)
    df_new['CC'] = df_new['CC'].astype(np.int64)
    df_new['CD'] = df_new['CD'].astype(np.int64)
    df_new['CF'] = df_new['CF'].astype(np.int64)
    df_new['CH'] = df_new['CH'].astype(np.int64)
    df_new['CL'] = df_new['CL'].astype(np.int64)
    df_new['CR'] = df_new['CR'].astype(np.int64)
    df_new['CS'] = df_new['CS'].astype(np.int64)
    df_new['CU'] = df_new['CU'].astype(np.int64)
    df_new['CW'] = df_new['CW'].astype(np.int64)
    df_new['DA'] = df_new['DA'].astype(np.int64)
    df_new['DE'] = df_new['DE'].astype(np.int64)
    df_new['DF'] = df_new['DF'].astype(np.int64)
    df_new['DH'] = df_new['DH'].astype(np.int64)
    df_new['DI'] = df_new['DI'].astype(np.int64)
    df_new['DL'] = df_new['DL'].astype(np.int64)
    df_new['DN'] = df_new['DN'].astype(np.int64)
    df_new['DU'] = df_new['DU'].astype(np.int64)
    df_new['DV'] = df_new['DV'].astype(np.int64)
    df_new['DY'] = df_new['DY'].astype(np.int64)
    df_new['EB'] = df_new['EB'].astype(np.int64)
    df_new['EE'] = df_new['EE'].astype(np.int64)
    df_new['EG'] = df_new['EG'].astype(np.int64)
    df_new['EH'] = df_new['EH'].astype(np.int64)
    df_new['EJ'] = df_new['EJ'].astype(np.int64)
    df_new['EL'] = df_new['EL'].astype(np.int64)
    df_new['EP'] = df_new['EP'].astype(np.int64)
    df_new['EU'] = df_new['EU'].astype(np.int64)
    df_new['FC'] = df_new['FC'].astype(np.int64)
    df_new['FD'] = df_new['FD'].astype(np.int64)
    df_new['FE'] = df_new['FE'].astype(np.int64)
    df_new['FI'] = df_new['FI'].astype(np.int64)
    df_new['FL'] = df_new['FL'].astype(np.int64)
    df_new['FR'] = df_new['FR'].astype(np.int64)
    df_new['FS'] = df_new['FS'].astype(np.int64)
    df_new['GB'] = df_new['GB'].astype(np.int64)
    df_new['GE'] = df_new['GE'].astype(np.int64)
    df_new['GF'] = df_new['GF'].astype(np.int64)
    df_new['GH'] = df_new['GH'].astype(np.int64)
    df_new['GI'] = df_new['GI'].astype(np.int64)
    df_new['GL'] = df_new['GL'].astype(np.int64)
    ## ...... ###
    ## Apply the pipeline
    X_processed = all_pipeline.transform(df_new)


    return X_processed