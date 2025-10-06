import requests
import pandas as pd # pandasをpdという名前でインポート
import matplotlib.pyplot as plt # matplotlib.pyplotをpltという名前でインポート
import matplotlib.axes as Axes
import datetime

# CoinGecko APIのURL
# 'bitcoin'の過去の市場データを取得する
URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

# APIに渡すパラメータ
# vs_currency: 日本円 (jpy)
# days: 過去365日分
# interval: データの粒度 (daily = 日ごと)
PARAMS = {
  "vs_currency": "jpy",
  "days": "365",
  "interval": "daily",
}

# 注釈を加えたいイベントの日付とテキストを定義
EVENTS = {
  # Key: 日付オブジェクト, Value: 表示したいテキスト
  datetime.date(2024, 11, 5): "Trump wins election,\nprice surges",
  datetime.date(2024, 12, 15): "New SEC commissioner\nfuels rally over $100k",
  datetime.date(2025, 3, 15): "Sustained ETF inflows\npush to new highs",
  datetime.date(2025, 8, 10): "Whale sells 24k BTC,\ncausing sharp drop",
}

OUTPUT_CSV_FILENAME = 'bitcoin_data_analysis.csv'
OUTPUT_ANALYSIS_CHART_FILENAME = 'bitcoin_analysis_chart_with_annotation.png'


def get_bitcoin_data(url: str, params: dict) -> dict | None:
  """
  受け取ったurlにアクセスし、APIからビットコインの価格データを取得
  Args:
    url (str): アクセス先のurl
    params (dict): APIに渡すパラメータ
  Returns:
    dict | None
    - APIから入手したビットコインの価格データ
    - アクセス中にエラーが発生した際の戻り値
  """
  print("CoinGecko APIからビットコインの価格データを取得します...")
  try:
    # APIにリクエストを送信
    response = requests.get(url, params)
    # HTTPエラーがあれば例外を発生させる (例: 404 Not Found)
    response.raise_for_status() 
    # レスポンスをJSON形式に変換
    data = response.json()
    print("データの取得に成功しました！")
    return data
  
  except requests.exceptions.RequestException as e:
    print(f"エラー: APIからのデータ取得に失敗しました。: {e}")
    return


def create_dataframe_from_pricelist(price_list: list) -> pd.core.frame.DataFrame:
  """
  取得した価格データをpandasのDataFrameに変換する
  Args:
    price_list (list): APIから取得した、タイムスタンプとその時のビットコインの価格データ
  Returns:
    pd.core.frame.DataFrame: タイムスタンプが日付に変換された、ビットコインの価格のデータフレーム
  """
  # 1. 取得した価格データをpandasのDataFrameに変換
  # price_listをpandasで分析できるDataFrameの表に変換
  # １列目を'timestamp'、２列目を'price'に指定
  df = pd.DataFrame(price_list, columns=['timestamp', 'price'])
  # 2. タイムスタンプを人間が読める「日付」に変換
  # timestamp内の時間はミリ秒表記のため、それを日付に変換する
  # pd.to_datetimeでタイムスタンプ(ミリ秒: unit = "ms")を日付に変換し、
  # dt.dateで日付部分だけを取り出す
  df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
  # 3. 新しく作った 'date' 列を、このDataFrameの「インデックス（行のラベル）」に設定
  # 元々、dbの行はindex = 1, 2, 3...だったものを、date = 2024-09-01, 2024-09-02...に変更
  # これによりグラフ化が簡単になる
  df = df.set_index('date')
  print("\n--- pandas DataFrameの作成完了 ---")
  return df


def calculate_financial_metrics(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
  """
  移動平均・日次リターン・ボラティリティをそれぞれ計算し、データフレームに追加する
  Args:
    df (pd.core.frame.DataFrame): 日付と、その時のビットコインの価格のデータフレーム
  Returns:
    pd.core.frame.DataFrame: 移動平均・日次リターン・ボラティリティが追加された、ビットコインの価格データフレーム
  """
  # 1. 移動平均線の計算
  # .rolling(window=25).mean() は、過去25日間の価格の平均を計算する
  # .rolling(window=25): 25日間の期間を区切って、窓(window)を一つずつずらしながらデータを眺める準備をする
  # .mean(): 窓の中に入っているデータすべての平均値を計算する
  df['ma_25'] = df['price'].rolling(window=25).mean()  # 25日移動平均 (短期トレンド)
  df['ma_75'] = df['price'].rolling(window=75).mean()  # 75日移動平均 (長期トレンド)
  print("\n移動平均線の計算完了。")
  # 2. 日次リターン（価格変動率）の計算
  # .pct_change(): は、ある時点のデータが、その直前のデータからどれだけ変化したかをパーセンテージで計算する
  # 日々の変化率のデータをリターンと呼ぶ
  df['returns'] = df['price'].pct_change()
  print("日次リターンの計算完了。")
  # 3. ボラティリティ（価格変動の激しさ）の計算
  # リターンの標準偏差を計算する。今回は30日間の標準偏差を計算
  # 30日の窓を設定、その中で、.std(標準偏差を計算)を行う
  df['volatility'] = df['returns'].rolling(window=30).std() * (365**0.5) # 年率換算
  print("ボラティリティの計算完了。")
  return df


def save_dataframe_to_csv(calculated_df: pd.core.frame.DataFrame, output_csv_filename: str) -> None:
  """
  データフレームをcsvに保存する
  Args:
    calculated_df (pd.core.frame.DataFrame): 移動平均・日次リターン・ボラティリティが追加された、ビットコインの価格データフレーム
    output_csv_filename (str): データの保存先のcsvファイル名
  Returns:
    None
  """
  calculated_df.to_csv(output_csv_filename)
  print(f"DataFrameを '{output_csv_filename}' として保存しました。")


def _plot_price_and_ma(ax: plt.Axes, df: pd.core.frame.DataFrame) -> None:
  """
  ビットコインの価格と移動平均線のデータをグラフにプロットする
  Args:
    ax (plt.axes._axes.Axes): matplotlib.pyplotのグラフ枠組み
    df (pd.core.frame.DataFrame): 移動平均・日次リターン・ボラティリティが追加された、ビットコインの価格データフレーム
  Returns:
    None
  """
  # --- 上段のグラフ (価格と移動平均線) ---
  ax.plot(df.index, df['price'], label='Bitcoin Price (JPY)', color='black', linewidth=1.5) # label: 凡例を設定。.legendで必要。
  ax.plot(df.index, df['ma_25'], label='25-Day Moving Average', color='blue', linestyle='--')
  ax.plot(df.index, df['ma_75'], label='75-Day Moving Average', color='red', linestyle='--')
  ax.set_title('Bitcoin Price and Moving Averages')
  ax.set_ylabel('Price (JPY)')
  ax.legend() # 凡例を表示

def _plot_volatility(ax: plt.Axes, df: pd.core.frame.DataFrame) -> None:
  """
  ビットコインのボラティリティのデータをグラフにプロットする
  Args:
    ax (plt.axes._axes.Axes): matplotlib.pyplotのグラフ枠組み
    df (pd.core.frame.DataFrame): 移動平均・日次リターン・ボラティリティが追加された、ビットコインの価格データフレーム
  Returns:
    None
  """
  # --- 下段のグラフ (ボラティリティ) ---
  ax.plot(df.index, df['volatility'], label='30-Day Volatility (Annualized)', color='purple')
  ax.set_title('Volatility of Bitcoin Price')
  ax.set_xlabel('Date')
  ax.set_ylabel('Volatility')
  ax.legend()

def _add_annotations(ax: plt.Axes, df: pd.core.frame.DataFrame, events: dict) -> None:
  """
  ビットコインの価格と移動平均線のグラフにアノテーションを追加する
  Args:
    ax (plt.axes._axes.Axes): ビットコインの価格と移動平均線のグラフ
    df (pd.core.frame.DataFrame): 移動平均・日次リターン・ボラティリティが追加された、ビットコインの価格データフレーム
    events (dict): 最近の、ビットコインの変動に関わる出来事の日時とその内容
  Returns:
    None
  """
  # 上段の価格グラフ (ax1) に注釈を追加
  # 辞書に対してforループを回したいとき、.items()を使うと、ループの各回でキーと値の両方を同時に受け取る
  # .items() を使わずにループを回すと、デフォルトではキーだけが取り出される
  for event_date, event_text in events.items():
    if event_date in df.index:
      # df.loc[]: インデックスのラベル名を指定して、データにアクセスする
      # event_dateの日の、priceの列を入手
      price_on_event_date = df.loc[event_date]['price']
      # 価格が上昇したイベントは緑の矢印、下落したイベントは赤い矢印にする
      arrow_color = 'red' if "drop" in event_text else 'green'
      
      ax.annotate(
        event_text,
        xy=(event_date, price_on_event_date),
        # テキストの位置を、価格が高いときは下に、低いときは上に自動調整
        # df['price'].mean(): 全期間の価格の平均値
        # price_on_event_date < df['price'].mean(): イベント発生時の価格が平均より下なら
        # price_on_event_date * 1.2: 価格の1.2倍上の場所に配置する
        xytext=(event_date, price_on_event_date * 1.2 if price_on_event_date < df['price'].mean() else price_on_event_date * 0.8),
        # facecolor: 矢印の塗りつぶしの色、shrink = 0.5: 矢印を5%縮める、width: 胴体の太さ、headwidth: 矢印の頭の幅
        arrowprops=dict(facecolor=arrow_color, shrink=0.05, width=2, headwidth=8),
        ha='center', # 水平方向に、中央揃え
        fontsize=9,
        # アノテーションのテキストを囲むボックスのスタイル
        # boxstyle="round,pad=0.3": ボックスの角を丸く、テキストとボックスの枠線の間に0.3文字分の余白を空ける
        # fc: 塗りつぶしの色
        # ec: ボックスの枠線の色
        # lw: ボックスの枠線の太さ
        # alpha: ボックスの透明度
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.7) # テキストに見やすい背景ボックスを追加
      )
      print(f"注釈を追加: {event_date} - {event_text}")

def create_and_save_analysis_chart(df: pd.core.frame.DataFrame, events: dict, output_filename: str) -> None:
  """
  ２種類のグラフを作成し、指定された名前のpngファイルに保存する
  グラフの内容は、ビットコインの価格推移と移動平均線のグラフと、ビットコインのボラティリティ推移のグラフである
  Args:
    df (pd.core.frame.DataFrame): 移動平均・日次リターン・ボラティリティが追加された、ビットコインの価格データフレーム
    events (dict): 最近の、ビットコインの変動に関わる出来事の日時とその内容
    output_filename (str): グラフの保存先のファイル名
  Returns:
    None
  """
  print("\n分析結果のグラフを作成します...")
  # plt.style.use(): これから描画するすべてのグラフに、このスタイルを適用する
  # seaborn-v0_8-whitegrid: matplotlibをより統計的に可視化しやすくするためのライブラリ名。グラフにグリッド線が入る。
  plt.style.use('seaborn-v0_8-whitegrid')
  # figは図全体、axes1, axes2は各グラフを指す
  # 縦に2行、横に1列のグラフを作成し、X軸を共有することで日付を連動させる
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

  _plot_price_and_ma(ax1, df)
  _plot_volatility(ax2, df)
  _add_annotations(ax1, df, events) # アノテーションはax1にのみ追加
  
  # matplotlibによって、タイトル・ラベル・グラフエリアが重ならないように最適化する
  plt.tight_layout()
  plt.savefig(output_filename)
  print(f"グラフを '{output_filename}' として保存しました。")


def main() -> None:
  """
  暗号通貨ボラティリティ分析ツールのメイン処理
  Args:
    None
  Returns:
    None
  """
  data = get_bitcoin_data(URL, PARAMS)
  if data == None:
    return
  # data['prices'] は [[タイムスタンプ1, 価格1], [タイムスタンプ2, 価格2], ...] という形式
  price_list = data['prices']
  df = create_dataframe_from_pricelist(price_list)
  calculated_df = calculate_financial_metrics(df.copy())
  print(type(EVENTS))
  save_dataframe_to_csv(calculated_df, OUTPUT_CSV_FILENAME)

  # --- 可視化部分を、2つのグラフを縦に並べるように変更 ---
  create_and_save_analysis_chart(calculated_df, EVENTS, OUTPUT_ANALYSIS_CHART_FILENAME)  


if __name__ == "__main__":
  main()