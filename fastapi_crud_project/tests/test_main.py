from fastapi.testclient import TestClient
from app.main import app

def test_read_user() -> None:
    """
    ユーザー取得APIの正常系テスト
    Args:
        None
    Returns:
        None
    """
    # このテスト関数内でのみ有効なクライアントを作成
    with TestClient(app) as client:
        response = client.get("/users/0") # 初期データのユーザーID=0を取得
        # HTTP 200 OKが返されることを確認
        assert response.status_code == 200
        # 返されるJSONが初期データと一致することを確認
        assert response.json() == {"id": 0, "name": "Alice", "age": 30}

def test_read_user_not_found() -> None:
    """
    ユーザー取得APIの異常系（ユーザーが存在しない）テスト
    Args:
        None
    Returns:
        None
    """
    with TestClient(app) as client:
        response = client.get("/users/999") # 存在しないユーザーIDを指定して取得
        # HTTP 404 Not Foundが返されることを確認
        assert response.status_code == 404
        # 返されるJSONがエラーメッセージと一致することを確認
        assert response.json() == {"detail": "User not found"}

def test_create_user() -> None:
    """
    ユーザー作成APIのテスト
    Args:
        None
    Returns:
        None
    """
    # このテストは他のテストの状態に依存しない
    # withブロック開始時にlifespanが実行され、DBが初期化される
    with TestClient(app) as client:
        user_data = {"name": "Charlie", "age": 40} # 新しいユーザーデータ
        # ユーザー作成APIにPOSTリクエストを送信
        response = client.post("/users/", json=user_data)
        # HTTP 201 Createdが返されることを確認
        assert response.status_code == 201

        # 返されるJSONに作成したユーザーのデータが含まれていることを確認
        response_json = response.json()
        assert response_json["name"] == "Charlie"
        assert response_json["id"] == 2 # 初期データが2つあるので、次はID=2になるはず

        # 作成したユーザーが実際に取得できるか確認
        created_user_id = response_json["id"]
        # 作成したユーザーを取得するためのGETリクエストを送信
        response_get = client.get(f"/users/{created_user_id}")
        assert response_get.status_code == 200