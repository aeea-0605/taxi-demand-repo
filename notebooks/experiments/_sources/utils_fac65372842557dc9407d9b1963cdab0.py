import numpy as np
import pandas as pd
import configparser
from sklearn.metrics import mean_absolute_error, mean_squared_error

config = configparser.ConfigParser()
config.read('./data.ini')

info = config['gcs']


def split_train_and_test_period(df, period):
    """
    Dataframe에서 train_df, test_df로 나눠주는 함수
    
    df : 시계열 데이터 프레임
    period : train/test 기준 일
    """
    criteria = str((max(df['pickup_hour']) - pd.Timedelta(days=period)).date())
    train_df = df[df['pickup_hour'] < criteria]
    test_df = df[df['pickup_hour'] >= criteria]
    
    return train_df, test_df


def evaluation(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    score = pd.DataFrame([mape, mae, mse], index=['mape', 'mae', 'mse'], columns=['score']).T

    return score


def split_x_and_y(train_df, test_df, y):
    train_df.drop(columns=['zip_code', 'pickup_hour'], inplace=True)
    test_df.drop(columns=['zip_code', 'pickup_hour'], inplace=True)
    
    y_train_raw = train_df.pop(y)
    y_test_raw = test_df.pop(y)
    
    x_train = train_df.copy()
    x_test = test_df.copy()
    
    return x_train, x_test, y_train_raw, y_test_raw
    
    
def split_x_and_y_log(train_df, test_df, y):
    train_df.drop(columns=['zip_code', 'pickup_hour'], inplace=True)
    test_df.drop(columns=['zip_code', 'pickup_hour'], inplace=True)
    
    y_train_raw = train_df.pop('cnt')
    y_train_log = train_df.pop(y)
    y_test_raw = test_df.pop('cnt')
    y_test_log = test_df.pop(y)
    
    x_train = train_df.copy()
    x_test = test_df.copy()
    
    return x_train, x_test, y_train_log, y_test_raw