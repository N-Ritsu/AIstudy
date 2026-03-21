import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import matplotlib


matplotlib.rcParams['font.family'] = 'IPAexGothic' # 日本語フォントを指定
matplotlib.rcParams['axes.unicode_minus'] = False      # マイナス記号の文字化け防止

# FutureWarning などの不要な警告を非表示にする
warnings.filterwarnings("ignore")


# --- サンプルデータの生成 ---
def generate_sample_data(n_points: int = 120, seed: int = 42) -> pd.Series:
    """
    月次の売上データを模した合成時系列データを生成する。
    - トレンド: 緩やかな上昇傾向
    - 季節性: 12ヶ月周期の波（夏に高く冬に低い）
    - ノイズ: ランダムなばらつき
    Args:
        n_points: データ点数（デフォルト: 10年分）
        seed: 乱数シード（再現性確保）
    Returns:
        pd.Series: 月次インデックス付き時系列データ
    """
    np.random.seed(seed)

    # 月次インデックスの作成（2015年1月スタート）
    index = pd.date_range(start="2015-01-01", periods=n_points, freq="MS")

    # 各成分を合成してリアルなデータを作成
    trend = np.linspace(100, 160, n_points) # 線形トレンド
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_points) / 12) # 12ヶ月周期
    noise = np.random.normal(0, 5, n_points) # ガウスノイズ

    values = trend + seasonal + noise
    return pd.Series(values, index=index, name="monthly_sales")


def generate_arima_friendly_data(n_points: int = 120, seed: int = 42) -> pd.Series:
    """
    ARIMAに有利な合成データを生成する。
    - トレンド  : 緩やかな上昇傾向
    - 季節性   : なし（ARIMAが得意とする条件）
    - ノイズ   : ランダムなばらつき
    """
    np.random.seed(seed)
    index = pd.date_range(start="2015-01-01", periods=n_points, freq="MS")

    trend = np.linspace(100, 160, n_points)
    noise = np.random.normal(0, 8, n_points) # 季節性なし、ノイズのみ

    values = trend + noise
    return pd.Series(values, index=index, name="monthly_sales_no_season")


# --- 定常性の確認（ADF検定） ---
# ARIMAを使う前に、データが定常かを確認する
def check_stationarity(series: pd.Series) -> None:
    """
    Augmented Dickey-Fuller (ADF) 検定で定常性を確認する。
    - p値 < 0.05: 定常性あり（ARIMAモデルに適用可能）
    - p値 >= 0.05: 差分を取るなどの前処理が必要
    Args:
        series: 検定対象の時系列データ
    Returns:
        None
    """
    # ADF検定を実行
    # 定常: トレンドや季節性なし、非定常: トレンドや季節性あり
    result = adfuller(series.dropna())
    print("\n--- ADF検定結果（定常性確認） ---")
    print(f"ADF統計量: {result[0]:.4f}")
    print(f"p値: {result[1]:.4f}")
    print(f"判定: {'定常（p < 0.05）' if result[1] < 0.05 else '非定常'}")


# --- データの分割（訓練・テスト） ---
def split_data(series: pd.Series, test_ratio: float = 0.2) -> tuple[pd.Series, pd.Series]:
    """
    時系列データを訓練データとテストデータに分割する。
    時系列はシャッフルせず、末尾をテストデータとして使う
    Args:
        series: 全時系列データ
        test_ratio: テストデータの割合（デフォルト20%）
    Returns:
        tuple
            - train: 訓練データ（最初の80%）
            - test: テストデータ（最後の20%）
    """
    split_idx = int(len(series) * (1 - test_ratio))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    print(f"\n--- データ分割 ---")
    print(f"  訓練データ: {len(train)} 件 ({train.index[0].date()} ～ {train.index[-1].date()})")
    print(f"  テストデータ: {len(test)} 件 ({test.index[0].date()} ～ {test.index[-1].date()})")
    return train, test


def fit_arima(train: pd.Series, test: pd.Series, order=(1, 1, 1)):
    """
    ARIMAモデルのローリング予測（1ステップずつ実データを更新）。
    毎ステップ実測値をモデルに追加して再予測するため、長期予測でも横ばいにならずデータの動きを追える。
    ただしローリング予想は、テストデータ部分についても教師データとして与え続ける必要があるため、あくまで性能評価に使え、本当の将来を予測するのはできない。
    ARIMAモデルを訓練データで学習し、テスト期間の予測を行う。
    Args:
        train: 訓練データ
        test: テストデータ（予測期間確認用）
        order:
            - p: 自己回帰次数（過去何期分を使うか）
            - d: 差分次数（1 = 1階差分で定常化）
            - q: 移動平均次数（過去の誤差を何期分使うか）
    Returns:
        tuple:
            - 予測値 pd.Series（テスト期間の予測）
            - 学習済みARIMAモデル（詳細分析用）
    """
    print(f"\n--- ARIMAモデル学習中（ローリング予測）... order={order} ---")

    history = list(train) # 訓練データを初期履歴として使用
    predictions = []

    for t in range(len(test)):
        # 毎ステップ最新の履歴でモデルを再学習
        model = ARIMA(history, order=order)
        fitted = model.fit()

        # 1ステップ先のみ予測
        pred = fitted.forecast(steps=1)[0]
        predictions.append(pred)

        # 実データ教師データとしてを履歴に追加
        history.append(test.iloc[t])

    forecast = pd.Series(predictions, index=test.index)
    print(f"最終モデル AIC: {fitted.aic:.2f}  BIC: {fitted.bic:.2f}")
    return forecast, fitted


def fit_sarima(train: pd.Series, test: pd.Series,
               order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """
    SARIMAモデルを訓練データで学習し、テスト期間の予測を行う。
    Args:
        train: 訓練データ
        test: テストデータ
        order:(非季節パラメータ)
            - p: 自己回帰次数
            - d: 差分次数
            - q: 移動平均次数
        seasonal_order:(季節パラメータ)
            - P: 季節自己回帰次数
            - D: 季節差分次数
            - Q: 季節移動平均次数
            - s: 季節周期
    Returns:
        tuple:
            - 予測値 pd.Series（テスト期間の予測）
            - 学習済みSARIMAモデル（詳細分析用）
    """
    print(f"\n--- SARIMAモデル学習中（ローリング予測）... order={order}, seasonal_order={seasonal_order} ---")

    history = list(train)
    predictions = []

    for t in range(len(test)):
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)

        pred = fitted.forecast(steps=1)[0]
        predictions.append(pred)

        history.append(test.iloc[t])

    forecast = pd.Series(predictions, index=test.index)
    print(f"  最終モデル AIC: {fitted.aic:.2f}  BIC: {fitted.bic:.2f}")
    return forecast, fitted


# --- 評価指標の計算・表示 ---
def evaluate_models(test: pd.Series, arima_pred: pd.Series, sarima_pred: pd.Series) -> pd.DataFrame:
    """
    MAE・RMSE・MAPEの3指標でARIMAとSARIMAを比較する。
    評価指標:
      MAE: 平均絶対誤差（値が小さいほど良い）
      RMSE: 二乗平均平方根誤差（外れ値の影響を受けやすい）
      MAPE: 平均絶対パーセント誤差（%で直感的に把握できる）
    Args:
        test: 実測値
        arima_pred: ARIMAの予測値
        sarima_pred: SARIMAの予測値
    Returns:
        pd.DataFrame: 評価指標の比較表
    """
    def mape(actual, predicted):
        # MAPE計算
        return np.mean(np.abs((actual - predicted) / actual)) * 100

    metrics = {
        "ARIMA": {
            "MAE" : mean_absolute_error(test, arima_pred),
            "RMSE": np.sqrt(mean_squared_error(test, arima_pred)),
            "MAPE": mape(test, arima_pred),
        },
        "SARIMA": {
            "MAE" : mean_absolute_error(test, sarima_pred),
            "RMSE": np.sqrt(mean_squared_error(test, sarima_pred)),
            "MAPE": mape(test, sarima_pred),
        },
    }
    df = pd.DataFrame(metrics).T
    print("\n--- モデル評価結果 ---")
    print(df.round(3).to_string())

    # より良いモデルを判定（MAPEで比較）
    better_model = df["MAPE"].idxmin()
    print(f"\nMAPE最小: {better_model}")
    return df


def run_comparison(series: pd.Series, title: str, filename: str, test_ratio: float = 0.2) -> None:
    """
    1つのデータセットに対してARIMA・SARIMAの比較を一通り実行する。
    Args:
        series: 比較対象の時系列データ
        title: グラフタイトルに使うラベル
        filename: 保存するグラフのファイル名
        test_ratio: テストデータの割合
    Returns:
        None
    """
    print(f"--- {title} ---")

    check_stationarity(series)
    train, test = split_data(series, test_ratio=test_ratio)
    arima_pred,  _ = fit_arima(train, test, order=(1, 1, 1))
    sarima_pred, _ = fit_sarima(train, test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    metrics_df = evaluate_models(test, arima_pred, sarima_pred)
    plot_results(series, train, test, arima_pred, sarima_pred, metrics_df, title=title, filename=filename)


# --- 結果の可視化 ---
def plot_results(series: pd.Series,
                 train: pd.Series,
                 test: pd.Series,
                 arima_pred: pd.Series,
                 sarima_pred: pd.Series,
                 metrics_df: pd.DataFrame,
                 title: str = "",
                 filename: str = "arima_sarima_comparison.png") -> None:
    """
    2段構成のグラフで結果を可視化する。
      上段: 全体の時系列 + 予測値の比較
      下段: 評価指標の棒グラフ
    Args:
        series      : 全時系列データ
        train       : 訓練データ
        test        : テストデータ（実測）
        arima_pred  : ARIMAの予測値
        sarima_pred : SARIMAの予測値
        metrics_df  : 評価指標の比較表
        title       : グラフタイトル
        filename    : 保存するグラフのファイル名
    Returns:
        None
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"ARIMA vs SARIMA：{title}", fontsize=16, fontweight="bold", y=0.98)

    # ---- 上段: 予測結果の折れ線グラフ ----
    ax1 = axes[0]
    ax1.plot(train.index, train.values, color="#2c3e50", linewidth=1.5, label="訓練データ（実測）")
    ax1.plot(test.index,  test.values,  color="#2c3e50", linewidth=1.5, linestyle="--", label="テストデータ（実測）")
    ax1.plot(arima_pred.index,  arima_pred.values,  color="#e74c3c", linewidth=2, label="ARIMA予測",  marker="o", markersize=4)
    ax1.plot(sarima_pred.index, sarima_pred.values, color="#3498db", linewidth=2, label="SARIMA予測", marker="s", markersize=4)

    # 訓練・テスト境界線
    ax1.axvline(x=test.index[0], color="gray", linestyle=":", linewidth=1.5, label="訓練/テスト境界")
    ax1.set_title("予測結果の比較", fontsize=13)
    ax1.set_ylabel("売上（万円）")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(axis="y", alpha=0.4)

    # ---- 下段: 評価指標の棒グラフ ----
    ax2 = axes[1]
    metrics_names = metrics_df.columns.tolist()   # ["MAE", "RMSE", "MAPE"]
    x = np.arange(len(metrics_names))
    bar_width = 0.35

    bars_arima  = ax2.bar(x - bar_width/2, metrics_df.loc["ARIMA"],  bar_width, label="ARIMA",  color="#e74c3c", alpha=0.85)
    bars_sarima = ax2.bar(x + bar_width/2, metrics_df.loc["SARIMA"], bar_width, label="SARIMA", color="#3498db", alpha=0.85)

    # 棒グラフの上に数値を表示
    for bar in bars_arima:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9, color="#e74c3c")
    for bar in bars_sarima:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9, color="#3498db")

    ax2.set_title("評価指標の比較（小さいほど良い）", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names, fontsize=11)
    ax2.set_ylabel("誤差")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nグラフを '{filename}' として保存しました。")
    plt.show()


def main() -> None:
    """
    ARIMAとSARIMAの予測性能を比較するプログラムのメイン処理。
    Args:
        None
    Returns:
        None
    """
    print("--- ARIMA / SARIMA 時系列予測 比較プログラム ---")

    # --- 季節性あり---
    series_seasonal = generate_sample_data(n_points=120)
    run_comparison(
        series_seasonal,
        title="季節性ありデータ（SARIMAに有利）",
        filename="comparison_seasonal.png",
        test_ratio=0.2
    )

    # --- 季節性なし---
    series_no_season = generate_arima_friendly_data(n_points=120)
    run_comparison(
        series_no_season,
        title="季節性なしデータ（ARIMAに有利）",
        filename="comparison_no_seasonal.png",
        test_ratio=0.3
    )

    print("\n両ケースの比較分析が完了しました")


# スクリプトとして直接実行する場合のエントリポイント
if __name__ == "__main__":
    main()