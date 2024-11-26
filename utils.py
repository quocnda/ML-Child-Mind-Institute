import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
#-------------Code------------
def handle_data_classifier(data: DataFrame) -> DataFrame:
    df_encoded = data.copy()
    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object' or df_encoded[column].dtype.name == 'category':
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
        if df_encoded[column].dtype in ['int64', 'float64']: 
            df_encoded[column].fillna(df_encoded[column].mean(), inplace=True)
    return df_encoded