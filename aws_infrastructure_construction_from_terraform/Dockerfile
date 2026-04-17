FROM python:3.10-slim

WORKDIR /app

# ビルドコンテキスト (aws_infrastructure_construction_from_terraform/) からの相対パスを指定。
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]