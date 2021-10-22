import json
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
    period : 기간에 따른train/test split

    return : 기준 period에 따른 train_df, test_df
    """

    criteria = str((max(df['pickup_hour']) - pd.Timedelta(days=period)).date())
    train_df = df[df['pickup_hour'] < criteria]
    test_df = df[df['pickup_hour'] >= criteria]
    
    return train_df, test_df


def evaluation(y_true, y_pred):
    """
    실제값과 예측값에 대한 mape, mae, mse의 metric값을 데이터프레임 형태로 리턴해주는 함수

    return : metrics에 대한 score_df, error 발생시 None
    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        
        score = pd.DataFrame([mape, mae, mse], index=['mape', 'mae', 'mse'], columns=['score']).T
        
        return score
    except:
        return None


def split_x_and_y(train_df, test_df, y):
    """
    train, test 데이터셋을 인자로 받아 독립변수와 종속변수로 Split하는 함수
    해당 함수는 raw_target 데이터셋에 적용되는 함수

    return : x_train, x_test, y_train_raw, y_test_raw
    """

    train_df.drop(columns=['zip_code', 'pickup_hour'], inplace=True)
    test_df.drop(columns=['zip_code', 'pickup_hour'], inplace=True)
    
    y_train_raw = train_df.pop(y)
    y_test_raw = test_df.pop(y)
    
    x_train = train_df.copy()
    x_test = test_df.copy()
    
    return x_train, x_test, y_train_raw, y_test_raw
    
    
def split_x_and_y_log(train_df, test_df, y):
    """
    train, test 데이터셋을 인자로 받아 독립변수와 종속변수로 Split하는 함수
    해당 함수는 log scaling된 데이터셋에 적용되는 함수

    return : x_train, x_test, y_train_log, y_test_raw, y_test_log
    """

    train_df.drop(columns=['zip_code', 'pickup_hour'], inplace=True)
    test_df.drop(columns=['zip_code', 'pickup_hour'], inplace=True)
    
    y_train_raw = train_df.pop('cnt')
    y_train_log = train_df.pop(y)
    y_test_raw = test_df.pop('cnt')
    y_test_log = test_df.pop(y)
    
    x_train = train_df.copy()
    x_test = test_df.copy()
    
    return x_train, x_test, y_train_log, y_test_raw, y_test_log


def parsing_output(ex_id):
    """
    sacred로 기록된 experiment_id를 인자로 받아 해당 모델링 결과를 데이터프레임 형태로 리턴하는 함수

    return : model_name, ex_id, model_config, metrics에 대한 output_df
    """
    
    with open(f'../airflow/data/taxi-demand-prediction/experiments/{ex_id}/metrics.json') as metric_file:
        metric_data = json.load(metric_file)
    with open(f'../airflow/data/taxi-demand-prediction/experiments/{ex_id}/run.json') as run_file:
        run_data = json.load(run_file)
    
    output_df = pd.DataFrame(metric_data['model_name']['values'], columns=['model_name'], index=['score'])
    output_df['experiment_num'] = ex_id
    output_df['config'] = json.dumps(metric_data['model_params']['values'])

    metric_df = pd.DataFrame(run_data['result'])
    
    output_df = pd.concat([output_df, metric_df], axis=1)
    output_df = output_df.round(2)
    
    return output_df