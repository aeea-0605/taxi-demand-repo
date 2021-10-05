import argparse
import os
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder

import rf_trainer
from base_data_query import base_query
from utils import init_config, split_train_and_test


parser = argparse.ArgumentParser()
parser.add_argument("--dev_env", help="Development Env [local], [development], [production]", type=str, default='local')
parser.add_argument("--mode", help="[train], [predict]", type=str, default="predict")

flag = parser.parse_args()

config = init_config(flag.dev_env)
print(config)
model_dir = f"{config['save_folder']}/models/"

# Feature Engineering (Using BigQuery)
print('load data')
base_df = pd.read_gbq(query=base_query, dialect='standard', project_id=config['project'])

# Data Preprocessing (Label Encoding)
zip_code_le = LabelEncoder()
base_df['zip_code_le'] = zip_code_le.fit_transform(base_df['zip_code'])

# Split Train and Test
train_df, test_df = split_train_and_test(base_df, '2015-01-24')
print('data split Done')

# Delete Columns, Values & Split X, y
train_df.drop(columns=['zip_code', 'pickup_hour'], inplace=True)
test_df.drop(columns=['zip_code', 'pickup_hour'], inplace=True)

y_train_raw = train_df.pop('cnt')
y_test_raw = test_df.pop('cnt')

train_df = train_df.fillna(method='backfill')
test_df = test_df.fillna(method='backfill')

x_train = train_df.copy()
x_test = test_df.copy()

# Run Modeling
if __name__ == '__main__':
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if flag.mode == 'train':
        print('train start')
        train_op = rf_trainer.Trainer(config)
        train_op.train(x_train, y_train_raw)
    elif flag.mode == 'predict':
        print('predict start')
        train_op = rf_trainer.Trainer(config)
        train_op.predict(x_test, y_test)
    else:
        raise KeyError(f"Incorrect value flog.mode = {flog.mode}")
