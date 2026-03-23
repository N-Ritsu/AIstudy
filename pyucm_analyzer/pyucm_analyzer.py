import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
from typing import Optional
from statsmodels.tsa.statespace.mlemodel import MLEResults


# データソースのURL
NOAA_CO2_DATA_URL = 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv'
# 出力するグラフのファイル名
OUTPUT_FILENAME = 'pyucm_analyzer_result_graph.png'
# トレンド比較の年数
TREND_COMPARISON_YEARS = 40


def fetch_co2_data() -> Optional[pd.Series]:
    """
    マウナロア観測所のCO2濃度データを取得する。
    最初にアメリカ海洋大気庁（NOAA）の公式サイトからの取得を試み、失敗した場合はstatsmodelsに同梱されている古典的なデータセットを返す。
    Args:
        None
    Returns:
        Optional[pd.Series]: 日付をインデックスとしCO2濃度を値とする時系列データ。
                             両方の取得に失敗した場合はNoneを返す。
    """
    print("Fetching latest CO2 data from NOAA...")
    try:
        # アメリカ海洋大気庁（NOAA）の公式サイトから最新の月次データを直接読み込む
        # CSVファイルを読み込み、冒頭のコメント部分をスキップする
        df = pd.read_csv(NOAA_CO2_DATA_URL, comment='#')

        # 日付のインデックスを作成する
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str))

        # pd.Series: それぞれのデータにインデックスを付与できるリストを作成
        # 'average'列を値とし、'date'列をインデックスとするSeriesを作成 -> dateをインデックスとすることで時系列データとして扱いやすくなる
        y = pd.Series(df['average'].values, index=df['date'], name='co2')

        print(f"Data successfully loaded. Latest data is from: {y.index[-1].strftime('%Y-%m-%d')}")
        return y

    except Exception as e:
        print(f"Failed to fetch live data: {e}")
        print("Falling back to the classic dataset (ends in 2001).")
        try:
            # データの取得に失敗した場合は、念のため従来のサンプルデータを使う
            data_df = sm.datasets.co2.load_pandas().data
            data_df.index = pd.to_datetime(data_df.index)
            y = data_df['co2'].resample('MS').mean().ffill()
            return y
        except Exception as e_fallback:
            print(f"Failed to load fallback dataset: {e_fallback}")
            return None


def build_and_fit_ucm_model(y: pd.Series) -> MLEResults:
    """
    時系列データに対して未観測成分モデル(UCM)を構築し、推定（フィッティング）を行う。
    Args:
        y (pd.Series): 分析対象の時系列データ。
    Returns:
        MLEResults: モデルの推定結果オブジェクト。
    """
    print("\nBuilding and fitting the Unobserved Components Model...")
    # 状態空間モデルの構築
    # level: トレンドの仮定 ('local linear trend' は局所線形トレンド)
    # seasonal: 季節周期 (月次データなので12)
    # stochastic_seasonal: 確率的季節成分(季節変動自体の変化)を許すか (Trueだと季節変動のパターンが時間で変化することを許容)
    model = sm.tsa.UnobservedComponents(
        y,
        level='local linear trend',
        seasonal=12,
        stochastic_seasonal=True
    )
    # モデルの推定を実行
    result = model.fit()

    print("--- Model Summary ---")
    print(result.summary())
    print("\n" + "="*80 + "\n")

    return result


def plot_ucm_results(y: pd.Series, result: MLEResults, output_filename: str) -> None:
    """
    UCMの分析結果（予測値と各成分）をグラフに描画し、ファイルに保存する。
    Args:
        y (pd.Series): 元の時系列データ。
        result (MLEResults): モデルの推定結果オブジェクト。
        output_filename (str): 保存するグラフのファイル名。
    Returns:
        None
    """
    print("Plotting results...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig.suptitle('Unobserved Components Model Analysis (未観測成分モデルによる分析)', fontsize=16)

    data_index = y.index

    # --- 予測と観測値 ---
    ax = axes[0]
    # グラフ描画開始位置
    start_loc = data_index[13]

    ax.plot(y.loc[start_loc:], label='Observed (実測値)', color='black', alpha=0.7)
    ax.plot(result.fittedvalues.loc[start_loc:], label='One-step-ahead predictions (予測値)', color='C0')
    # 予測の信頼区間を取得(conf_int)
    pred_ci = result.get_prediction(start=start_loc).conf_int()
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], alpha=0.2, color='C0', label='95% confidence interval (信頼区間)')
    ax.set_title('Predicted vs Observed', fontsize=14)
    ax.legend(loc='upper left')

    # --- 各成分をプロットするヘルパー関数 ---
    def plot_component(ax: plt.Axes, component_name: str, component_object: dict, original_index: pd.DatetimeIndex):
        """UCMの各成分（レベル、トレンド、季節）を一つのサブプロットに描画する"""
        smoothed_values = component_object['smoothed']
        smoothed_cov = component_object['smoothed_cov']

        # statsmodelsのバージョンにより共分散の次元が異なる場合があるため、分岐処理を行う
        if smoothed_cov.ndim == 3:
            std_err = np.sqrt(smoothed_cov[0, 0, :])
        elif smoothed_cov.ndim == 1:
            std_err = np.sqrt(smoothed_cov)
        else:
            std_err = np.zeros_like(smoothed_values)

        lower_ci = smoothed_values - 1.96 * std_err
        upper_ci = smoothed_values + 1.96 * std_err

        ax.plot(original_index, smoothed_values, label=f'{component_name} (smoothed)')
        ax.fill_between(original_index, lower_ci, upper_ci, alpha=0.2)
        ax.set_title(f'{component_name} component ({component_name}成分)', fontsize=14)
        ax.legend(loc='upper left')

    # --- グラフ2-4: 各成分のグラフを描画 ---
    plot_component(axes[1], 'Level', result.level, data_index)
    plot_component(axes[2], 'Trend', result.trend, data_index)
    plot_component(axes[3], 'Seasonal', result.seasonal, data_index)

    # 全体のレイアウトを調整
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # グラフをファイルに保存
    # dpiで解像度、bbox_inchesで余白を調整
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    print(f"Graph has been saved to '{output_filename}'")


def generate_trend_analysis_commentary(trend_series: pd.Series, comparison_years: int) -> str:
    """
    トレンド成分の時系列データから、分析コメントを自動生成する。
    Args:
        trend_series (pd.Series): トレンド成分の時系列データ（インデックスは日付）。
        comparison_years (int): 何年前と比較するかを指定する年数。
    Returns:
        str: 分析結果をまとめた文章。
    """
    # 最新時点のデータを取得（インデックスと値）
    latest_date = trend_series.index[-1]
    latest_value = trend_series.iloc[-1]

    # 比較する過去の時点のデータを取得
    past_date_target = latest_date - pd.DateOffset(years=comparison_years)
    # ターゲット日付に最も近い、存在するデータのインデックス番号を取得
    past_date_index_pos = trend_series.index.get_indexer([past_date_target], method='nearest')[0]
    past_date_actual = trend_series.index[past_date_index_pos]
    past_value = trend_series.iloc[past_date_index_pos]

    commentary = []

    # --- 現在のトレンドの方向性 ---
    # トレンドの傾き（月のCO2濃度変化量）がプラスかマイナスかで判断
    if latest_value > 0.001:
        direction = "明確な増加傾向"
    elif latest_value < -0.001:
        direction = "明確な減少傾向"
    else:
        direction = "ほぼ横ばい"

    commentary.append(f"最新のトレンドは「{direction}」を示しています。")
    commentary.append(f"  - 最新({latest_date.strftime('%Y年%m月')})の上昇ペース: {latest_value:.4f} ppm/月")

    # --- トレンドの加速/減速 (過去との比較) ---
    # 過去のトレンドの傾きがほぼゼロの場合は比較が難しいためスキップ
    if abs(past_value) < 1e-6:
        commentary.append(f"{comparison_years}年前のトレンドがほぼゼロのため、加速/減速の評価はスキップします。")
    else:
        if latest_value > past_value:
            change_description = "ペースは加速しています"
        elif latest_value < past_value:
            change_description = "ペースは減速しています"
        else:
            change_description = "ペースはほぼ同じです"

        # 上昇ペースが何倍になったかを計算
        ratio = latest_value / past_value

        commentary.append(f"{comparison_years}年前と比較して、{change_description}。")
        commentary.append(f"  - {comparison_years}年前({past_date_actual.strftime('%Y年%m月')})の上昇ペース: {past_value:.4f} ppm/月")
        commentary.append(f"  - 上昇ペースは約 {ratio:.2f} 倍に変化しました。")

    return "\n".join(commentary)


def main() -> None:
    """
    BSM（状態空間モデル）を用いて、CO2濃度データを分析するメイン関数。
    Args:
        None
    Returns:
        None
    """
    # データの取得
    co2_data = fetch_co2_data()
    if co2_data is None:
        print("Failed to get data. Exiting script.")
        return

    # 状態空間モデルの構築と推定
    ucm_result = build_and_fit_ucm_model(co2_data)

    # 分析結果のグラフ描画
    plot_ucm_results(co2_data, ucm_result, OUTPUT_FILENAME)

    # トレンド分析コメントの生成と表示
    print("\n--- Trend Analysis Commentary ---")

    # resultオブジェクトからトレンド成分（平滑化済み）を抽出
    smoothed_trend_series = ucm_result.trend['smoothed']

    # トレンド成分の時系列データから分析コメントを生成
    analysis_comment = generate_trend_analysis_commentary(
        smoothed_trend_series,
        comparison_years=TREND_COMPARISON_YEARS
    )

    # 結果を出力
    print(analysis_comment)
    print("\nScript finished successfully.")


if __name__ == '__main__':
    main()