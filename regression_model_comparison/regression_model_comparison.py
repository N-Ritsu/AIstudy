import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import japanize_matplotlib

# 再現性を確保するための乱数シード
RANDOM_STATE = 42
# テストデータの割合
TEST_SIZE = 0.3


def get_regression_models() -> Dict[str, Dict[str, Any]]:
    """
    比較対象となる回帰モデルの辞書を取得する。
    各モデルについて、モデルのインスタンスと、入力データ（特徴量）のスケーリングが必要かどうかのフラグを返す。
    線形モデルやSVRは、特徴量のスケールに影響を受けやすいためスケーリングを行う。
    Args:
        None
    Returns:
        Dict[str, Dict[str, Any]]: 
            モデル名（str）をキーとし、
            'model'（モデルインスタンス）と'needs_scaling'（ブール値）を含む辞書を値とする辞書。
    """
    return {
        "Linear Regression": {"model": LinearRegression(), "needs_scaling": True},
        "Ridge": {"model": Ridge(random_state=RANDOM_STATE), "needs_scaling": True},
        "Lasso": {"model": Lasso(random_state=RANDOM_STATE), "needs_scaling": True},
        "SVR": {"model": SVR(), "needs_scaling": True},
        "Decision Tree": {"model": DecisionTreeRegressor(random_state=RANDOM_STATE), "needs_scaling": False},
        "Random Forest": {"model": RandomForestRegressor(random_state=RANDOM_STATE), "needs_scaling": False},
        "Gradient Boosting": {"model": GradientBoostingRegressor(random_state=RANDOM_STATE), "needs_scaling": False},
        "LightGBM": {"model": lgb.LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1), "needs_scaling": False},
    }


def run_regression_comparison(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    dataset_name: str,
    feature_names: Optional[List[str]] = None
) -> None:
    """
    指定されたデータセットで全モデルの性能評価と可視化を行う。
    Args:
        X_raw (np.ndarray): 生の特徴量データ。
        y_raw (np.ndarray): 生の目的変数データ。
        dataset_name (str): データセットの名前（グラフタイトルやログ出力用）。
        feature_names (Optional[List[str]], optional): 
            特徴量の名前のリスト。指定されない場合、"feature_0", "feature_1"...のように自動生成される。
            デフォルトは None。
    Returns:
        None
    """
    print(f"--- 分析開始: {dataset_name} ---")

    # --- データ準備 ---
    # NumPy配列からPandasのDataFrame/Seriesに変換し、後続の処理をしやすくする
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_raw.shape[1])]
    
    X = pd.DataFrame(X_raw, columns=feature_names)
    y = pd.Series(y_raw, name="target")

    # データを学習用(70%)とテスト用(30%)に分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 特徴量のスケーリング（標準化）を準備。これにより、スケールの異なる特徴量をモデルが公平に扱えるようになる。
    scaler = StandardScaler()
    # 学習データに基づいてスケーラーを学習（fit）し、学習データを変換（transform）
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    # 学習データで学習したスケーラーを使い、テストデータを変換
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # --- モデルの学習・評価・可視化 ---
    models_to_compare = get_regression_models()
    results = []
    
    # 特徴量が1つ（1次元）の場合のみグラフを描画する
    should_plot = X.shape[1] == 1
    
    if should_plot:
        # 2行4列のグリッドでグラフを描画する準備
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        axes_flat = axes.flatten()
    else:
        print(f"INFO: {dataset_name}は多次元データのため、グラフは表示されません。")
        # グラフ描画がない場合もループを回すため、ダミーのイテレータを用意
        axes_flat = [None] * len(models_to_compare)

    # 各モデルについてループ処理
    for ax, (name, model_info) in zip(axes_flat, models_to_compare.items()):
        model = model_info['model']
        needs_scaling = model_info['needs_scaling']

        # --- モデルの学習 ---
        # スケーリングが必要なモデルはスケール済みデータを、不要なモデルは元のデータを使用
        if needs_scaling:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # --- 性能評価 ---
        # MSE (Mean Squared Error, 平均二乗誤差)
        mse = mean_squared_error(y_test, y_pred)
        # R2 (R-squared, 決定係数): モデルがデータの変動をどれだけ説明できているかを示す指標。1に近いほど良い。
        r2 = r2_score(y_test, y_pred)
        results.append([name, mse, r2])

        # --- 可視化 (1次元データの場合のみ) ---
        if should_plot:
            # i. 元データの散布図
            ax.scatter(X_train, y_train, alpha=0.5, label="学習データ", s=15)
            ax.scatter(X_test, y_test, alpha=0.7, label="テストデータ", s=20, marker='x', c='orange')
            
            # ii. 予測線を描画するためのx軸データを生成
            x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
            x_range_np = np.linspace(x_min, x_max, 300).reshape(-1, 1)
            # モデルの入力形式に合わせてDataFrameに変換
            x_range_df = pd.DataFrame(x_range_np, columns=X.columns)

            # iii. 予測線のy軸データをモデルから取得
            if needs_scaling:
                x_range_scaled = pd.DataFrame(scaler.transform(x_range_df), columns=X.columns)
                y_range_pred = model.predict(x_range_scaled)
            else:
                y_range_pred = model.predict(x_range_df)
            
            # iv. グラフに予測線を描画し、体裁を整える
            ax.plot(x_range_df, y_range_pred, label="予測線", color='red', linewidth=2)
            ax.set_title(f"{name}\n(R2={r2:.3f}, MSE={mse:.3f})", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize='small')

    # --- 結果の表示 ---
    if should_plot:
        fig.suptitle(f"【{dataset_name}】各モデルの予測結果の比較", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # タイトルとの重なりを防ぐ
        plt.show()

    # 評価結果をDataFrameにまとめて表示
    results_df = pd.DataFrame(results, columns=["Model", "MSE (↓)", "R2 (↑)"])
    print(f"\n--- {dataset_name} の性能評価まとめ ---")
    # R2スコアが高い順（性能が良い順）にソートして表示
    print(results_df.sort_values("R2 (↑)", ascending=False))
    print("-" * (len(dataset_name) + 24) + "\n")


def main() -> None:
    """
    複数のデータセットで回帰モデルの比較を実行するメイン処理。
    Args:
        None
    Returns:
        None
    """
    # --- 人工的な線形データ ---
    # y = 2.5x + 5 + ノイズ の単純な線形関係を持つデータ
    X1_raw = np.random.rand(200, 1) * 10
    y1_raw = 2.5 * X1_raw.squeeze() + 5 + np.random.randn(200) * 2
    run_regression_comparison(X1_raw, y1_raw, "ケース1: 人工的な線形データ")

    # --- 人工的な非線形データ (sinカーブ) ---
    # y = sin(x) + ノイズ の非線形関係を持つデータ
    X2_raw = np.sort(5 * np.random.rand(200, 1), axis=0)
    y2_raw = np.sin(X2_raw).ravel() + np.random.randn(200) * 0.1
    run_regression_comparison(X2_raw, y2_raw, "ケース2: 人工的な非線形データ (sinカーブ)")

    # --- カリフォルニア住宅価格データ (実世界の複雑なデータ) ---
    # 複数の特徴量（所得、築年数など）から住宅価格を予測する
    housing = fetch_california_housing()
    X3_raw, y3_raw = housing.data, housing.target
    run_regression_comparison(
        X3_raw, y3_raw, "ケース3: カリフォルニア住宅価格データ", feature_names=housing.feature_names
    )

    # --- 不要な特徴量を多く含む人工データ ---
    # 100個の特徴量のうち、10個だけが目的変数と関係を持つデータ
    # LassoやRidgeなどの正則化モデルの有効性を確認するのに役立つ
    X4_raw, y4_raw = make_regression(
        n_samples=200, n_features=100, n_informative=10, noise=20, random_state=RANDOM_STATE
    )
    run_regression_comparison(X4_raw, y4_raw, "ケース4: 不要な特徴量を多く含むデータ")


if __name__ == "__main__":
    main()