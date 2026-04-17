# app/main.py
from fastapi import FastAPI

app = FastAPI(
    title="Terraform Demo API",
    description="A simple API deployed automatically with Terraform and GitHub Actions.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """
    ルートエンドポイント。デプロイが成功したことを示すメッセージを返す。
    """
    return {"message": "Hello, World from an API deployed by Terraform!"}