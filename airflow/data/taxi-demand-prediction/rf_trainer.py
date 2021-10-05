import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from utils import load_model, evaluation, root_path


class Trainer:
    def __init__(self, config):
        self.project = config['project']
        self.jwt = os.path.join(root_path, config['jwt'])
        self.dataset = config['dataset']
        self.save_folder = config['save_folder']

        if os.path.exists(f"{self.save_folder}/models/rf_model"):
            self.model = load_model(f"{self.save_folder}/models/rf_model")
        else:
            self.model = None


    def train(self, x_train, y_train):
        rf_reg = RandomForestRegressor(n_estimators=10, n_jobs=-1)
        self.model = rf_reg.fit(x_train, y_train)
        joblib.dump(self.model, open(f"{self.save_folder}/models/rf_model", 'wb'))
        print('Model Save Success!')

        return self.model


    def predict(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        
        output_df = x_test.copy()
        output_df['predicted_demand'] = y_pred.astype(int)
        output_df['model_name'] = 'random_forest'

        evaluation_value = evaluation(y_test, y_pred)
        print(f"Evaluation Value : {evaluation_value}")

        output_df.to_gbq(destination_table=f"{self.dataset}.output", project_id=self.project, if_exists='append', private_key=self.jwt)

        return output_df


    def train_and_predict(self, x_train, y_train, x_test, y_test):
        self.train(x_train, y_train)
        self.predict(x_test, y_test)