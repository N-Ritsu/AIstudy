import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split  # ★ KFoldをインポート
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Tuple, Dict, Any


def prepare_data() -> pd.DataFrame:
    """
    カリフォルニア住宅価格データセットをロードし、基本的な前処理を行う。
    Args:
        None
    Returns:
        pd.DataFrame: 前処理済みのデータフレーム。
    """
    print("--- データの準備を開始します ---")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    # 全てのカラムを数値型に変換し、変換できない値はNaNとし、NaNを含む行を削除
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    print("--- データの準備完了 ---\n")
    return df


def apply_feature_engineering(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    interaction_features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, KMeans, PolynomialFeatures]:
    """
    学習データと検証データに対して特徴量エンジニアリングを適用する。
    データリークを防ぐため、学習データで学習し、そのモデルで両方のデータを変換する。
    Args:
        X_train (pd.DataFrame): 学習データ。
        X_val (pd.DataFrame): 検証データ。
        interaction_features (List[str]): 交互作用特徴量を生成するための特徴量のリスト。
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, KMeans, PolynomialFeatures]:
            - 変換後の学習データ
            - 変換後の検証データ
            - 学習済みのKMeansモデル
            - 学習済みのPolynomialFeaturesモデル
    """
    # --- K-Meansによる地域クラスタリング ---
    # K-Meansモデルを学習データで学習
    # n_clusters: 6つのグループに分割
    kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto')
    kmeans.fit(X_train[['Latitude', 'Longitude']]) # 緯度と経度のみで学習
    # 学習データと検証データにクラスタ番号を特徴量として追加
    # 学習データの各住宅が6つのエリアのうちどれに属するかを予測
    X_train['GeoCluster'] = kmeans.predict(X_train[['Latitude', 'Longitude']])
    # 検証データの各住宅が6つのエリアのうちどれに属するかを予測
    X_val['GeoCluster'] = kmeans.predict(X_val[['Latitude', 'Longitude']])

    # --- PolynomialFeaturesによる交互作用特徴量の生成 ---
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    
    # 各特徴量同士を組み合わせた交互作用特徴量を学習データで生成
    poly_features_train = poly.fit_transform(X_train[interaction_features])
    # 生成された特徴量をDataFrameに変換
    poly_df_train = pd.DataFrame(poly_features_train, columns=poly.get_feature_names_out(interaction_features), index=X_train.index)
    # 元の特徴量を削除し、新しい特徴量と結合(重複を避ける)
    X_train = X_train.drop(columns=interaction_features).join(poly_df_train)

    # 検証データは学習済みのモデルで変換のみ行う(新しく学習はしない)
    poly_features_val = poly.transform(X_val[interaction_features])
    poly_df_val = pd.DataFrame(poly_features_val, columns=poly.get_feature_names_out(interaction_features), index=X_val.index)
    # 元の特徴量を削除し、新しい特徴量と結合
    X_val = X_val.drop(columns=interaction_features).join(poly_df_val)

    return X_train, X_val, kmeans, poly


def perform_cross_validation(X_train_orig: pd.DataFrame, y_train: pd.Series) -> None:
    """
    交差検証を実行してモデルの性能を評価する。
    交差検証: 単に80%を学習用、20%を検証用とすると、たまたまの可能性を捨てきれないため、学習用と検証用のデータを順番に変更しながら評価する。
    Args:
        X_train_orig (pd.DataFrame): オリジナルの学習データ（特徴量）
        y_train (pd.Series): オリジナルの学習データ（目的変数）
    Returns:
        None
    """
    print("--- 交差検証によるモデル評価を開始します ---")
    
    # --- 交差検証の設定 ---
    # n_splits: データを5分割する(4つを学習用(80%), 1つを検証用(20%)とする)
    # shuffle: データを分割する前にシャッフルする
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores: List[float] = [] # 各分割でのスコアを保存するリスト
    interaction_features = ['MedInc', 'AveRooms', 'HouseAge', 'AveOccup'] # この4つの特徴量に対してのみ、お互いを組み合わせた特徴量を生成する

    print("交差検証を開始します...")
    # --- 交差検証のループを開始 ---
    # 5つに分割したデータのうち4つを学習用、1つを検証用として、ローテーションで評価する
    for fold, (train_index, val_index) in enumerate(kf.split(X_train_orig)):
        print(f"--- Fold {fold+1}/5 ---")
        
        # 今回のループ用の学習データと検証データを取得
        # X: 2次元のデータ構造のため大文字、y: 1次元のデータ構造のため小文字
        X_train_fold = X_train_orig.iloc[train_index].copy() # 学習用データ
        X_val_fold = X_train_orig.iloc[val_index].copy() # 検証用データ
        y_train_fold = y_train.iloc[train_index] # 学習用の解答
        y_val_fold = y_train.iloc[val_index] # 検証用の解答

        # データフレームに新しいラベルを追加するイメージ
        # ループ内で特徴量エンジニアリングを実行（データリーク防止）
        X_train_fold_featured, X_val_fold_featured, _, _ = apply_feature_engineering(
            X_train_fold, X_val_fold, interaction_features
        )

        # --- モデルの学習と評価 ---
        # XGBoost回帰モデルを学習させる
        # xgb.XGBRegressor: XGBoostを、数値の予測問題に使用するための箱を作成
        # objective='reg:squarederror': 目的関数を二乗誤差に設定(目的関数がなるべく小さくなるように学習する)
        # n_estimators=100: 決定木の数を100に設定
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        # 実際に学習
        model.fit(X_train_fold_featured.to_numpy(), y_train_fold)
        # 検証データに対する予測を行う
        y_pred_val = model.predict(X_val_fold_featured.to_numpy())
        # 検証データに対する予測と実際の値を比較する
        mse = mean_squared_error(y_val_fold, y_pred_val)
        # 結果を記録
        mse_scores.append(mse)
        print(f"Fold {fold+1} MSE: {mse:.4f}")

    print("\n--- 交差検証 結果 ---")
    print(f"MSEの平均: {np.mean(mse_scores):.4f}")
    print(f"MSEの標準偏差: {np.std(mse_scores):.4f}")
    print("--- 交差検証完了 ---\n")


def train_final_model_and_prepare_shap_data(
    X_train_orig: pd.DataFrame,
    y_train: pd.Series,
    X_test_orig: pd.DataFrame
) -> Tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame]:
    """
    全ての学習データを使って最終的なモデルを学習し、SHAP分析用のデータを準備する。
    Args:
        X_train_orig (pd.DataFrame): 全ての学習データ（特徴量）
        y_train (pd.Series): 全ての学習データ（目的変数）
        X_test_orig (pd.DataFrame): 今まで一度もモデルに見せたことが無いテストデータ（特徴量）
    Returns:
        Tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame]:
            - 学習済みの最終モデル
            - 特徴量エンジニアリング適用後の学習データ
            - 特徴量エンジニアリング適用後のテストデータ
    """
    print("--- SHAP分析のための最終モデル学習とデータ準備を開始します ---")
    
    interaction_features = ['MedInc', 'AveRooms', 'HouseAge', 'AveOccup']
    
    # 特徴量エンジニアリングを、今度は全学習データとテストデータに適用
    # apply_feature_engineeringは学習/検証セットを想定しているが、ここでは学習/テストセットに適用して、最終モデル用のデータを準備する
    X_train_final, X_test_final, _, _ = apply_feature_engineering(
        X_train_orig.copy(), X_test_orig.copy(), interaction_features
    )
    
    # 最終モデルの学習
    final_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    final_model.fit(X_train_final.to_numpy(), y_train)
    
    print("最終モデルの学習が完了しました。")
    print("--- 最終モデル学習完了 ---\n")
    return final_model, X_train_final, X_test_final


def analyze_and_visualize_with_shap(model: xgb.XGBRegressor, X_test_final: pd.DataFrame) -> None:
    """
    学習済みモデルとテストデータを用いてSHAP分析を行い、結果を可視化して保存する。
    Args:
        model (xgb.XGBRegressor): 学習済みのXGBoostモデル。
        X_test_final (pd.DataFrame): 特徴量エンジニアリング適用後のテストデータ。
    Returns:
        None
    """
    print("--- SHAPによる分析と可視化を開始します ---")
    # SHAPのExplainerをモデルで初期化
    explainer = shap.TreeExplainer(model)
    
    print("SHAP値の計算中です...")
    # SHAP値: 各特徴量が予測にどれだけ寄与しているかを示す値。正の値は予測を増加させ、負の値は予測を減少させる。
    shap_values = explainer(X_test_final)
    print("SHAP値の計算が完了しました。")

    # 特徴量の重要度を示すサマリープロットを作成・保存
    print("\n特徴量の重要度を示すサマリープロットを作成します...")
    plt.figure()
    shap.summary_plot(shap_values, X_test_final, show=False)
    plt.tight_layout()
    plt.savefig('summary_plot.png')
    plt.close()
    print("`summary_plot.png` という名前でグラフを保存しました。")

    # テストデータの最初の1件に対する予測根拠を示すウォーターフォールプロットを作成・保存
    print("\nテストデータの最初の1件に対する予測根拠を示すウォーターフォールプロットを作成します...")
    plt.figure()
    # shap_values[0] はテストデータの最初のケースを指します
    # max_display=15 のように表示する特徴量の最大数を指定すると、特徴量が多くてもプロットが見やすくなります。
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    plt.tight_layout() # プロットが見切れないように調整
    plt.savefig('waterfall_plot.png')
    plt.close() # メモリ解放
    print("`waterfall_plot.png` という名前でグラフを保存しました。")
    print("--- SHAP分析完了 ---")


def main() -> None:
    """
    SHAPを用いたXAIによる予測根拠の可視化プログラムのメイン処理。
    Args:
        None
    Returns:
        None
    """
    # データの準備
    df = prepare_data()

    # 目的変数と特徴量に分割
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # 最終的な性能評価のためのホールドアウトテストデータを分割
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 交差検証によるモデル評価
    perform_cross_validation(X_train_orig, y_train)

    # SHAP分析のための最終モデル学習
    final_model, _, X_test_final = train_final_model_and_prepare_shap_data(
        X_train_orig, y_train, X_test_orig
    )

    # SHAPによる分析と可視化
    analyze_and_visualize_with_shap(final_model, X_test_final)


if __name__ == "__main__":
    main()