import pandas as pd
import numpy as np
import category_encoders
import pickle
import xgboost
def calculate_missing_data(dataset):
    total_missing = dataset.isnull().sum()
    percent_missing = (dataset.isnull().sum() / len(dataset)).round(4) * 100

    missing_data = pd.DataFrame({'Feature': total_missing.index, 'Total': total_missing.values, 'Percent': percent_missing.values})
    missing_data = missing_data[missing_data['Total'] > 0].sort_values(by='Percent', ascending=False)

    return missing_data
def log_transform(y):
    return np.log1p(y)
# Inverse log transotm akan digunakan pada hasil prediksi data test, untuk mengembalikan nilai ke nilai sebelum log transform
def inverse_log_transform(y_log):
    return np.expm1(y_log)
def make_prediction(dict):
    df = pd.DataFrame(dict, index=[0])
    log_transform = ["LotArea", "LowQualFinSF", "KitchenAbvGr", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal"]
    for col in log_transform:
        df[col] = np.log1p(df[col])
    loaded_model = pickle.load(open("../ext/Neighborhood.pkl", 'rb'))
    encoded = loaded_model.transform(df["Neighborhood"])
    df= pd.concat([df, encoded], axis=1)
    df.drop("Neighborhood", axis=1, inplace=True)
    loaded_model = pickle.load(open("../ext/MasVnrType.pkl", 'rb'))
    encoded = loaded_model.transform(df["MasVnrType"])
    df= pd.concat([df, encoded], axis=1)
    df.drop("MasVnrType", axis=1, inplace=True)
    loaded_model = pickle.load(open("../ext/scaler.pkl", 'rb'))
    df = loaded_model.transform(df)
    loaded_model = pickle.load(open("../ext/finalized_model.pkl", 'rb'))
    predict = loaded_model.predict(df)
    result = np.expm1(predict)
    return result