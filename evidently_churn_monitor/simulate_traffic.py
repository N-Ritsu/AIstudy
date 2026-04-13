# simulate_traffic.py
import pandas as pd
import numpy as np
import requests
import time
import json

API_URL = "http://127.0.0.1:5000/predict"
REFERENCE_DATA_PATH = "monitoring_artifacts/reference_data.csv"
N_REQUESTS = 200 # 送信する総リクエスト数
DRIFT_START_INDEX = 100 # 何番目のリクエストからデータをドリフトさせるか

def generate_drifted_data(n_samples: int, seed: int) -> pd.DataFrame:
    """
    データドリフトを意図的に含んだ新しい顧客データを生成する
    Args:
        n_samples: 生成するサンプル数
        seed: 乱数シード
    Returns:
        DataFrame: ドリフトした顧客データセット
            - tenure: 契約期間（月数）
            - monthly_charges: 月額料金
            - total_charges: 総請求額
            - contract: 契約タイプ（Month-to-month）
    """
    np.random.seed(seed)
    # キャンペーンにより、契約期間が短く、月額料金が高い新しい顧客層が増えたと仮定
    drifted_tenure = np.random.randint(1, 12, n_samples) # 契約期間が短い
    drifted_monthly_charges = np.random.uniform(80, 130, n_samples) # 月額料金が高い
    drifted_total_charges = drifted_monthly_charges * drifted_tenure * (1 + np.random.uniform(-0.1, 0.1, n_samples))
    drifted_contract_type = np.random.choice(['Month-to-month'], n_samples)
    
    df = pd.DataFrame({
        'tenure': drifted_tenure,
        'monthly_charges': drifted_monthly_charges,
        'total_charges': drifted_total_charges,
        'contract': drifted_contract_type,
    })
    return df

def main() -> None:
    """
    APIにリクエストを送信し、トラフィックをシミュレートする
    Args:
        None
    Returns:
        None
    """
    print("--- 本番トラフィックのシミュレーション開始 ---")
    # 事前にベースラインデータを読み込む
    try:
        reference_df = pd.read_csv(REFERENCE_DATA_PATH)
    except FileNotFoundError:
        print(f"エラー: ベースラインデータ '{REFERENCE_DATA_PATH}' が見つかりません。")
        return
        
    # ドリフトデータを準備
    drifted_df = generate_drifted_data(N_REQUESTS - DRIFT_START_INDEX, seed=123)
    
    # APIにリクエストを送信してトラフィックをシミュレート
    # 前半はベースラインデータからランダムにサンプリングし、後半はドリフトデータを使用
    # 送信するデータから解約情報を除き、APIに解約情報を予測させる
    # APIの予測結果はログファイルに記録されるため、後でドリフトレポートで分析できるようになる
    for i in range(N_REQUESTS):
        if i < DRIFT_START_INDEX:
            # 前半は通常のデータ
            sample = reference_df.sample(1).drop('churn', axis=1)
            print(f"リクエスト {i+1}/{N_REQUESTS} (通常データ)... ", end="")
        else:
            # 後半はドリフトしたデータ
            sample = drifted_df.iloc[[i - DRIFT_START_INDEX]]
            print(f"リクエスト {i+1}/{N_REQUESTS} (ドリフトデータ)... ", end="")
            
        # DataFrameの行をJSONに変換
        payload = sample.to_dict(orient='records')[0]
        
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                print("成功")
            else:
                print(f"失敗 (Status: {response.status_code})")
        except requests.exceptions.ConnectionError:
            print("\nエラー: APIサーバーに接続できません。")
            print("'app.py' が別のターミナルで実行されているか確認してください。")
            return
            
        time.sleep(0.1)
    
    print("シミュレーションが完了しました。")

if __name__ == '__main__':
    main()