# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# モデルとログファイルのパスを定義
MODEL_PATH = 'monitoring_artifacts/churn_model.pkl'
LOG_FILE_PATH = 'monitoring_artifacts/production_logs.csv'
model = None

def load_model() -> None:
    """
    アプリケーション起動時にモデルをロードする
    Args:
        None
    Returns:
        None
    """
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"モデル '{MODEL_PATH}' を正常にロードしました。")
    except FileNotFoundError:
        print(f"エラー: モデルファイル '{MODEL_PATH}' が見つかりません。")
        print("先に 'train_churn_model.py' を実行してください。")
        exit()

# 予測用のエンドポイントを定義
@app.route('/predict', methods=['POST'])
def predict() -> jsonify:
    """
    予測リクエストを受け取り、結果を返し、ログを記録する
    Args:
        None
    Returns:
        jsonify: 予測結果
    """
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    # --- 1. リクエストからデータを取得 ---
    data = request.get_json()
    # DataFrameに変換（モデルのパイプラインはDataFrameを期待するため）
    input_df = pd.DataFrame([data])
    
    # --- 2. 予測の実行 ---
    prediction_proba = model.predict_proba(input_df)[0][1] # 解約する確率
    prediction = int(model.predict(input_df)[0])
    
    # --- 3. ログの記録 ---
    # リクエストデータと予測結果を結合して1行のログを作成
    log_entry = input_df.copy()
    log_entry['prediction_proba'] = prediction_proba
    log_entry['prediction'] = prediction
    
    # ログファイルに追記
    header = not os.path.exists(LOG_FILE_PATH)
    log_entry.to_csv(LOG_FILE_PATH, mode='a', header=header, index=False)
    
    # --- 4. レスポンスを返す ---
    return jsonify({
        'prediction': prediction,
        'churn_probability': prediction_proba
    })

if __name__ == '__main__':
    load_model()
    # ログファイルが存在する場合は初期化
    if os.path.exists(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)
        print(f"既存のログファイル '{LOG_FILE_PATH}' を削除しました。")
    app.run(port=5000, debug=True)