from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
import os

# このDAGファイルが置かれているディレクトリを取得
# Airflowコンテナ内では /opt/airflow/dags/ml_pipeline/airflow_dag
DAG_DIR = os.path.dirname(__file__)

# プロジェクトのルートディレクトリを特定
# DAG_DIRから2つ上の階層がプロジェクトルートになる
PROJECT_ROOT = os.path.abspath(os.path.join(DAG_DIR, '..'))

# DAGの定義(DAGのスケジュールや開始日などを設定)
with DAG(
    dag_id='breast_cancer_pipeline', # DAGの名前
    start_date=datetime(2026, 4, 5), # DAGの開始日
    schedule_interval='@daily', # DAGのスケジュール（毎日実行）
    catchup=False, # 過去のスケジュールされた実行をキャッチアップしない
    tags=['ml', 'classification'], # DAGのタグ（Airflow UIでフィルタリングに使用）
) as dag:
    
    # Pythonがpipeline_tasksモジュールを見つけられるように、PYTHONPATHを設定する
    # AirflowのBashOperator内で共通して使う環境変数を定義
    common_env = {
        "PYTHONPATH": f"${{PYTHONPATH}}:{PROJECT_ROOT}"
    }

    # パイプライン実行前の準備タスク
    setup_task = BashOperator(
        task_id='setup_directories_and_data',
        env=common_env, # 環境変数を適用
        bash_command=(
            f"mkdir -p {PROJECT_ROOT}/artifacts/{{processed_data,trained_models,evaluation_results}} && "
            f"rm -rf {PROJECT_ROOT}/artifacts/*/* && "
            f"python {PROJECT_ROOT}/data/create_breast_cancer_dataset.py"
        )
    )

    # 各タスクをBashOperatorで定義
    task_validate = BashOperator(
        task_id='validate_data',
        env=common_env,
        bash_command=f"python -m pipeline_tasks.task_01_validate_data"
    )
    
    task_split = BashOperator(
        task_id='split_and_preprocess_data',
        env=common_env,
        bash_command=f"python -m pipeline_tasks.task_02_split_data"
    )

    task_train = BashOperator(
        task_id='train_model',
        env=common_env,
        bash_command=f"python -m pipeline_tasks.task_03_train_model"
    )

    task_evaluate = BashOperator(
        task_id='evaluate_model',
        env=common_env,
        bash_command=f"python -m pipeline_tasks.task_04_evaluate_model"
    )

    task_deploy = BashOperator(
        task_id='deploy_model',
        env=common_env,
        bash_command=f"python -m pipeline_tasks.task_05_deploy_model"
    )

    # タスクの依存関係（実行順序）を定義
    setup_task >> task_validate >> task_split >> task_train >> task_evaluate >> task_deploy