from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG(
    dag_id="initial_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 7, 1),
    schedule_interval="@daily",
    catchup=True
) as dag:

    crawl = BashOperator(
        task_id="crawl_videos",
        bash_command="python /opt/airflow/dags/load_to_df.py"
    )

    load_mysql = BashOperator(
        task_id="load_to_mysql",
        bash_command="python /opt/airflow/dags/hdfs_to_mysql.py"
    )

    # 실행 순서 정의
    crawl >> load_mysql
