from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


FILE_PATH: str = 'causal_data.csv'
# 特徴量のカラム名
TREATMENT_COL: str = 'treatment'
# 結果のカラム名
OUTCOME_COL: str = 'purchase_amount'
# 傾向スコアの推定に使用する特徴量のカラム名リスト
FEATURE_COLS: List[str] = ['age', 'gender', 'monthly_visits']


def load_data(file_path: str) -> pd.DataFrame:
    """
    指定されたパスからCSVファイルを読み込み、DataFrameとして返す。
    Args:
        file_path (str): 読み込むCSVファイルのパス。
    Returns:
        pd.DataFrame: 読み込まれたデータ。
    Raises:
        FileNotFoundError: 指定されたファイルパスにファイルが存在しない場合。
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        print("まず 'create_causal_dataset.py' を実行して、データセットを生成してください。")
        exit()


def calculate_naive_effect(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> Tuple[float, float, float]:
    """
    介入群と対照群の単純な平均差（ナイーブな効果）を計算する。
    Args:
        df (pd.DataFrame): 分析対象のデータ。
        treatment_col (str): 介入の有無を示すカラム名。
        outcome_col (str): 結果（購買額など）を示すカラム名。
    Returns:
        Tuple[float, float, float]: 以下の3つの値を含むタプル:
            - 介入群の結果の平均値。
            - 対照群の結果の平均値。
            - 推定されたナイーブな効果（平均値の差）。
    """
    treatment_group = df[df[treatment_col] == 1]
    control_group = df[df[treatment_col] == 0]
    
    mean_treatment = treatment_group[outcome_col].mean()
    mean_control = control_group[outcome_col].mean()
    naive_effect = mean_treatment - mean_control
    
    return mean_treatment, mean_control, naive_effect


def estimate_propensity_scores(df: pd.DataFrame, feature_cols: List[str], treatment_col: str) -> pd.Series:
    """
    ロジスティック回帰を用いて傾向スコア(ある人に対する介入の可能性)を推定する。
    Args:
        df (pd.DataFrame): 分析対象のデータ。
        feature_cols (List[str]): 傾向スコアの推定に使用する共変量（特徴量）のカラム名リスト。
        treatment_col (str): 介入の有無を示すカラム名。
    Returns:
        pd.Series: 各サンプルの傾向スコア。
    """
    X = df[feature_cols] # 共変量: 介入の有無に影響を与える可能性のある特徴量のこと
    T = df[treatment_col] # 実際に介入を受けたかどうか
    
    model = LogisticRegression()
    model.fit(X, T)
    propensity_scores = model.predict_proba(X)[:, 1]
    
    return pd.Series(propensity_scores, index=df.index)


def perform_propensity_score_matching(df: pd.DataFrame) -> float:
    """
    傾向スコアを用いて最近傍マッチングを行い、平均処置効果（ATE）を推定する。
    実際に介入を受けたサンプルの傾向スコアに最も近い対照群のサンプルを見つける(傾向スコアが近い条件のサンプルを対象にすることで実際の効果を推定する)。
    その結果の差を平均してATEを計算する。
    Args:
        df (pd.DataFrame): 'propensity_score', 'treatment', 'purchase_amount' カラムを含むデータ。
    Returns:
        float: 傾向スコアマッチングによって推定された平均処置効果（ATE）。
    """
    treatment_df = df[df[TREATMENT_COL] == 1] # 介入群のデータフレームを作成
    control_df = df[df[TREATMENT_COL] == 0] # 対照群のデータフレームを作成
    
    # 最近傍探索モデルの準備(傾向スコアを特徴量として使用して、対照群の中から最も近いサンプルを見つけるためのモデルを作成)
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(control_df[['propensity_score']])
    
    # 介入群の各サンプルに対して、最も傾向スコアが近い対照群のサンプルを見つける
    distances, indices = nn.kneighbors(treatment_df[['propensity_score']])
    
    # マッチした対照群のインデックスを取得
    matched_control_indices = control_df.iloc[indices.flatten()].index
    
    # マッチング後の介入群と対照群の結果を取得
    treatment_outcomes = treatment_df[OUTCOME_COL].values
    matched_control_outcomes = df.loc[matched_control_indices, OUTCOME_COL].values
    
    # 平均処置効果（ATE）を計算
    ate = (treatment_outcomes - matched_control_outcomes).mean()
    
    return ate, treatment_outcomes.mean(), matched_control_outcomes.mean()


def main() -> None:
    """
    メールクーポンを送るマーケティング施策の効果を測定する因果推論を行うメイン処理。
    Args: 
        None
    Returns:
        None
    """
    # --- データの読み込み ---
    df = load_data(FILE_PATH)
    
    # --- ナイーブな分析（単純比較） ---
    # セレクションバイアスにより真の効果である+20よりも大きく出てしまうはず
    mean_treat, mean_ctrl, naive_effect = calculate_naive_effect(df, TREATMENT_COL, OUTCOME_COL)
    print("--- ナイーブな分析（単純比較） ---")
    print(f"介入群の平均購買額: {mean_treat:.2f}")
    print(f"対照群の平均購買額: {mean_ctrl:.2f}")
    print(f"推定された効果（単純差）: {naive_effect:.2f}")

    # --- 傾向スコアマッチングによる分析 ---
    print("--- 傾向スコアマッチングによる分析 ---")
    
    # Step 2-1: 傾向スコアの推定
    df['propensity_score'] = estimate_propensity_scores(df, FEATURE_COLS, TREATMENT_COL)
    
    # Step 2-2: マッチングと効果の推定
    ate_matched, matched_treatment_mean, matched_control_mean = perform_propensity_score_matching(df)
    print(f"マッチング後の介入群の平均購買額: {matched_treatment_mean:.2f}")
    print(f"マッチング後の対照群の平均購買額: {matched_control_mean:.2f}")
    print(f"傾向スコアマッチングによる推定効果 (ATE): {ate_matched:.2f}")


if __name__ == '__main__':
    main()