# --- START OF FILE Dockerfile ---

# 1. ベースとなる公式のAirflowイメージを指定
# このバージョンは、ダウンロードしたdocker-compose.yaml内のimage指定と合わせるのが望ましい
FROM apache/airflow:2.9.2

# 2. Airflowを実行するユーザー(airflow)に切り替える
# この後、pip installなどがこのユーザー権限で実行される
USER airflow

# 3. ホストPC上の requirements.txt をコンテナ内にコピーする
# ./dags/ml-pipeline-project/requirements.txt というパスにあるファイルを、
# コンテナ内の /requirements.txt という名前でコピーする
COPY ./dags/ml_pipeline/requirements.txt /requirements.txt

# 4. コピーした requirements.txt を使って、必要なライブラリをインストールする
RUN pip install --no-cache-dir -r /requirements.txt