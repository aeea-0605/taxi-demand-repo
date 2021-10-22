from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago


default_args = {
    'owner': 'seungju',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['tmdwnrj96@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'project_id': 'bigquery-project-327214'
}

with DAG(
        dag_id='Predict-taxi_demand_every_3hours',
        description='Predict taxi demand',
        schedule_interval='0 */3 * * *',
        default_args=default_args) as dag:
    bash_command = "python3 /home/airflow/gcs/data/taxi-demand-prediction/main.py --mode predict --dev_env production"
    predict_operator = BashOperator(
        task_id='predict_demand',
        bash_command=bash_command
    )
    
    complete_task = BashOperator(
        task_id='predict_demand_complete',
        bash_command='echo "Prediction Done!!"'
    )

    predict_operator >> complete_task