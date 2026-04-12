from fastapi import FastAPI, HTTPException, status
from typing import Dict
from contextlib import asynccontextmanager
from .models import UserIn, UserOut

# データベースの代わりとなるインメモリの辞書
db: Dict[int, UserOut] = {}
next_user_id = 0

@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """
    サーバーの起動と終了時に実行される関数
    サーバー起動時: データベースを初期化し、サンプルデータを追加
    サーバー終了時: メッセージを表示
    Args:
        app(FastAPI): FastAPIアプリケーションのインスタンス
    Yields:
        None
    """
    # サーバー起動時に実行される処理
    print("サーバーを起動します。サンプルデータを初期化します。")
    global db, next_user_id
    db = {
        0: UserOut(id=0, name="Alice", age=30),
        1: UserOut(id=1, name="Bob", age=25),
    }
    next_user_id = 2 # 初期データが2つあるので、次のユーザーIDは2から始まる
    yield
    # サーバー終了時に実行される処理
    print("サーバーをシャットダウンします。")

app = FastAPI(
    title="Simple User API",
    description="A simple API to manage users.",
    version="1.0.0",
    lifespan=lifespan # ここでlifespan関数(サーバーの起動と終了時に実行される関数)をFastAPIアプリに渡す
)

@app.post("/users/", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(user_in: UserIn) -> UserOut:
    """
    データベースに新しいユーザーを追加
    Args:
        user_in(UserIn): クライアントから受け取るユーザーデータ（UserInモデル）
    Returns:
        UserOut: 作成されたユーザーのデータ（UserOutモデル）
    """
    global next_user_id
    # .dict() を .model_dump() に変更
    new_user = UserOut(id=next_user_id, **user_in.model_dump())
    db[next_user_id] = new_user
    next_user_id += 1
    return new_user

@app.get("/users/{user_id}", response_model=UserOut)
def read_user(user_id: int) -> UserOut:
    """
    指定されたIDのユーザーを取得
    Args:
        user_id(int): 取得するユーザーのID
    Returns:
        UserOut: ユーザーのデータ（UserOutモデル）
    """
    if user_id not in db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return db[user_id]