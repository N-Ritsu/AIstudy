import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import classification_report
import joblib
import os

def create_churn_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    顧客解約に関する擬似データセットを生成する
    Args:
        n_samples: 生成するサンプル数
        seed: 乱数シード
    Returns:
        DataFrame: 顧客データセット
            - tenure: 契約期間（月数）
            - monthly_charges: 月額料金
            - total_charges: 総請求額
            - contract: 契約タイプ（Month-to-month, One year, Two year）
            - churn: 解約フラグ（0: 非解約, 1: 解約）
    """
    np.random.seed(seed)
    tenure = np.random.randint(1, 72, n_samples) # 1ヶ月から6年までの契約期間をランダムに生成
    monthly_charges = np.random.uniform(20, 120, n_samples) # 月額料金を20ドルから120ドルの範囲でランダムに生成
    total_charges = monthly_charges * tenure * (1 + np.random.uniform(-0.1, 0.1, n_samples)) # 総請求額を計算
    contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.6, 0.3, 0.1]) # 契約タイプをランダムに生成
    churn_prob = 1 / (1 + np.exp(-( -0.05 * tenure + 0.02 * monthly_charges + (contract_type == 'Month-to-month') * 1.5 - 2))) # 解約確率を計算
    churn = np.random.binomial(1, churn_prob, n_samples) # 解約フラグを生成
    
    df = pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract': contract_type,
        'churn': churn
    })
    return df

def main() -> None:
    """
    モデルを学習し、必要なファイル（モデル、学習データ）を保存する
    Args:
        None
    Returns:
        None
    """
    print("--- モデル学習とベースラインデータ保存 (改善版) ---")
    
    # データの準備
    df = create_churn_dataset()
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 前処理パイプラインの定義
    numeric_features = ['tenure', 'monthly_charges', 'total_charges'] # 数値特徴量のリスト
    categorical_features = ['contract'] # 分類特徴量のリスト
    
    numeric_transformer = StandardScaler() # 数値特徴量を標準化
    # カテゴリカル特徴量をワンホットエンコード
    # カテゴリカル特徴量: 契約タイプのように、数値ではなくカテゴリで表される特徴量のこと。
    # ワンホットエンコード: カテゴリカル特徴量を数値に変換する方法の一つで、各カテゴリを二進数の列に変換する。
    # handle_unknown='ignore' は、学習時に見たことのないカテゴリがテストデータに出現した場合にエラーを防ぐための設定。 
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # 前処理の統合: 数値特徴量とカテゴリカル特徴量の両方を同時に処理するためのColumnTransformerを定義
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # モデルの定義と学習(LightGBMを使用)
    # is_unbalance=True は、データが不均衡であることをモデルに伝え、自動で重み調整を行わせる設定です。
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', lgb.LGBMClassifier(random_state=42, is_unbalance=True))])
    
    print("モデル (LightGBM) を学習しています...")
    model.fit(X_train, y_train)
    print("モデルの学習が完了しました。")

    # 学習済みモデルの性能を評価
    print("\n--- テストデータでの性能評価 ---")
    y_pred = model.predict(X_test)
    # 分類レポートを表示。特に minority class (churn=1) の Recall と F1-score が重要
    print(classification_report(y_test, y_pred, target_names=['非解約 (0)', '解約 (1)']))
    
    # モデルとベースラインデータの保存
    if not os.path.exists('monitoring_artifacts'):
        os.makedirs('monitoring_artifacts')
        
    joblib.dump(model, 'monitoring_artifacts/churn_model.pkl')
    print("\n学習済みモデルを 'monitoring_artifacts/churn_model.pkl' に保存しました。")
    
    reference_data = X_train.copy()
    reference_data['churn'] = y_train
    reference_data.to_csv('monitoring_artifacts/reference_data.csv', index=False)
    print("ベースラインデータを 'monitoring_artifacts/reference_data.csv' に保存しました。")

if __name__ == '__main__':
    main()