FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY ./app /app/app

# uvicornサーバーを起動
# 0.0.0.0: コンテナ外部からのアクセスを許可
# --host 0.0.0.0 --port 80: コンテナの80番ポートで待ち受ける
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]