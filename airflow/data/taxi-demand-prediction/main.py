import argparse
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder

import reg_trainer
from base_data_query import base_query
from modules import init_config, split_train_and_test_period, split_x_and_y_log


parser = argparse.ArgumentParser()
parser.add_argument("--dev_env", help="Development Env [local], [development], [production]", type=str, default='local')
parser.add_argument("--mode", help="[train], [predict]", type=str, default="predict")

flag = parser.parse_args()

config = init_config(flag.dev_env)

model_dir = f"{config['save_folder']}/models/"

# Feature Engineering (Using BigQuery)
print('load data')
base_df = pd.read_gbq(query=base_query, dialect='standard', project_id=config['project'])

# Data Preprocessing (Label Encoding)
le = LabelEncoder()
base_df['zip_code_le'] = le.fit_transform(base_df['zip_code'])

# log scaling >> 0인 값이 존재하기에 log1p를 사용해 inf값이 존재하지 않게함
base_df['log_cnt'] = np.log1p(base_df['cnt'])

# Split Train and Test
train_df, test_df = split_train_and_test_period(base_df, 7)
print('data split Done')

# Null값에 대해 각 행 기준 다음 행의 값을 채우고, 다음 행이 없다면 0값을 채움
train_df = train_df.fillna(method='backfill')
test_df = test_df.fillna(method='backfill')

# Split X, y
x_train, x_test, y_train_log, _, y_test_log = split_x_and_y_log(train_df, test_df, y='log_cnt')

# Run Modeling
if __name__ == '__main__':
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if flag.mode == 'train':
        print('train start')
        train_op = reg_trainer.Trainer(config)
        train_op.train(x_train, y_train_log)
    elif flag.mode == 'predict':
        print('predict start')
        train_op = reg_trainer.Trainer(config)
        train_op.predict(x_test, y_test_log)
    else:
        raise KeyError(f"Incorrect value flag.mode = {flag.mode}")
