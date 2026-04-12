from pydantic import BaseModel, Field

# ユーザーを作成・更新する際にクライアントから受け取るデータ
class UserIn(BaseModel):
    # ユーザー名は必須で、1文字以上50文字以下
    name: str = Field(..., min_length=1, max_length=50, description="ユーザー名")
    # 年齢は必須で、0以上の整数
    age: int = Field(..., ge=0, description="年齢")

# DB（今回は辞書）に保存され、クライアントに返すデータ
class UserOut(UserIn):
    # ユーザーIDは必須で、整数
    id: int = Field(..., description="一意のユーザーID")