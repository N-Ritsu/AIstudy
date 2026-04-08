import requests

url = "http://localhost:3000/classify"

data = {"input_series": [[5.9, 3.0, 5.1, 1.8]]}

response = requests.post(url, json=data)

print(f"ステータスコード: {response.status_code}")
print(f"レスポンス: {response.text}")
print(f"予測されたクラス: {response.json()[0]}")