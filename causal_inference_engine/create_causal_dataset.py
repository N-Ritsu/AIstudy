import pandas as pd
import numpy as np

def create_causal_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    セレクションバイアスを含む擬似的な顧客データを生成する。
    この関数は、因果推論の分析でよく見られるセレクションバイアスを意図的に含んだデータセットを作成する。
    データの特徴:
    - 年齢が高いほど、また訪問回数が多いほど購買額が上がる傾向がある。
    - 訪問回数が多い「優良顧客」ほど、クーポンが送られやすくなっている（セレクションバイアス）。
    - クーポンには、真の効果として「購買額を平均+20する」という効果が設定されている。
    Args:
        n_samples (int, optional): 生成するサンプル（顧客）数。デフォルトは 1000。
        seed (int, optional): 乱数生成器のシード値。再現性のために使用。デフォルトは 42。
    Returns:
        pd.DataFrame: 生成された顧客データを含むDataFrame。以下のカラムを持つ:
            - 'age': 顧客の年齢 (int)
            - 'gender': 顧客の性別 (0: 女性, 1: 男性)
            - 'monthly_visits': 月間訪問回数 (int)
            - 'treatment': 介入の有無 (1: クーポンあり, 0: クーポンなし)
            - 'purchase_amount': 購買額 (float)
    """
    np.random.seed(seed)
    
    # 顧客の属性を生成
    age: np.ndarray = np.random.randint(20, 61, n_samples)
    gender: np.ndarray = np.random.choice([0, 1], n_samples, p=[0.5, 0.5]) # 0: 女性, 1: 男性
    monthly_visits: np.ndarray = np.random.randint(1, 31, n_samples)
    
    # --- セレクションバイアスの導入 ---
    # 訪問回数が多いほどクーポンが送られやすいように、介入確率を計算
    # ロジスティック関数を使って、訪問回数が多いほど1に近づく確率を生成
    treatment_prob: np.ndarray = 1 / (1 + np.exp(-(monthly_visits * 0.1 - 1.5)))
    treatment: np.ndarray = np.random.binomial(1, treatment_prob, n_samples)
    
    # --- 購買額の生成 ---
    # 属性から基本的な購買額を計算
    base_purchase: np.ndarray = 10 + age * 0.5 + monthly_visits * 2 + gender * 5 + np.random.normal(0, 5, n_samples)
    
    # --- 真の因果効果の導入 ---
    # 介入群（クーポンを送られた人）にだけ、真の効果 (+20) を上乗せ
    true_causal_effect: int = 20
    purchase_amount: np.ndarray = base_purchase + treatment * true_causal_effect
    
    # DataFrameにまとめる
    df: pd.DataFrame = pd.DataFrame({
        'age': age,
        'gender': gender,
        'monthly_visits': monthly_visits,
        'treatment': treatment, # 介入(1) or 対照(0)
        'purchase_amount': purchase_amount.round(2)
    })
    
    df.to_csv('causal_data.csv', index=False)
    print("擬似データセット 'causal_data.csv' を生成しました。")
    return df

if __name__ == '__main__':
    create_causal_dataset()