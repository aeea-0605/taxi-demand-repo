import os
import json
import numpy as np
import pandas as pd
import joblib


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from modules import load_model, evaluation, root_path, parsing_output
from google.oauth2 import service_account


class Trainer:
    def __init__(self, config):
        self.project = config['project']
        self.jwt = os.path.join(root_path, config['jwt'])
        self.dataset = config['dataset']
        self.save_folder = config['save_folder']
        self.credentials = service_account.Credentials.from_service_account_file(self.jwt)

        if os.path.exists(f"{self.save_folder}/models/reg_model.pkl"):
            self.model = load_model(f"{self.save_folder}/models/reg_model.pkl")
        else:
            self.model = None


    def train(self, x_train, y_train):
        best_model = pd.concat([parsing_output(i) for i in range(1, 6)]).sort_values('mse').iloc[0]
        reg_name = best_model['model_name']
        reg_params = json.loads(best_model['config'])[0]

        if reg_name == "XGBRegressor":
            reg_model = XGBRegressor(**reg_params, n_jobs=-1)
        elif reg_name == "LGBMRegressor":
            reg_model = LGBMRegressor(**reg_params, n_jobs=-1)
        else:
            reg_model = RandomForestRegressor(**reg_params, n_jobs=-1)
        
        self.model = reg_model.fit(x_train, y_train)
        
        with open(f"{self.save_folder}/models/reg_model.pkl", "wb") as f:  
            joblib.dump(self.model, f)
        print(f'{reg_name} Model Save Success!')

        return self.model


    def predict(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        y_pred_reverse = np.expm1(y_pred)
        
        output_df = x_test.copy()
        output_df['predicted_demand'] = y_pred_reverse.astype(int)
        output_df['model_name'] = self.model.__class__.__name__

        evaluation_value = evaluation(y_test, y_pred_reverse)
        print(f"Evaluation Value : {evaluation_value}")
        
        output_df.to_gbq(destination_table=f"{self.dataset}.output", project_id=self.project, if_exists='append', credentials=self.credentials)

        return output_df


    def train_and_predict(self, x_train, y_train, x_test, y_test):
        self.train(x_train, y_train)
        self.predict(x_test, y_test)