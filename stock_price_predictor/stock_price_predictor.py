import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from typing import Tuple

TIME_STEP = 60 # 過去何日分のデータを見るか
EPOCHS = 100 # 学習のエポック数

class LSTM(nn.Module):
    """
    株価予測のためのLSTMモデルを定義するクラス
    """
    def __init__(self, input_size: int = 1, hidden_layer_size: int = 50, output_size: int = 1) -> None:
        """
        LSTMモデルを初期化
        Args:
          input_size (int): ネットワークに入力する特徴量の数。デフォルトは1（終値のみ）
          hidden_layer_size (int): LSTM層の記憶能力の大きさ。デフォルトは50
          output_size (int): ネットワークが出力する値の数。デフォルトは1（次の日の株価）
        Returns:
            None
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        # LSTM層: 時間軸を考慮できる層。60日分のデータを入力として受け取り記憶しておくことができ、50次元の特徴量を出力。
        # 逆に、たくさんのデータから１つの結論を導き出すのは苦手(情報量が極端に減ってしまう)
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        # 全結合層: LSTM層からの50の特徴量を受け取り、最終的な１つの値を導き出す。
        # 多くの特徴量に重みづけを行い1つの答えをだすのは得意だが、時間軸など順番が関係するデータを扱えない(同時に渡されたデータのみ処理できる)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        モデルの順伝播処理を定義
        Args:
            input_seq (torch.Tensor): 入力シーケンスデータ(シーケンス: 順番に意味があるデータの連なり)
        Returns:
            torch.Tensor: 予測結果
        """
        lstm_out, _ = self.lstm(input_seq) # LSTM層による特徴量抽出
        # lstm_out[:, -1, :]: LSTM層が抽出した特徴量(60日×50個)のうち、最終日のものだけを取り出す
        predictions = self.linear(lstm_out[:, -1, :]) # 全結合層による最終予測
        return predictions

def get_stock_data(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    指定されたティッカーシンボル(金融商品の略称。例えばApple Inc. → AAPL)と期間の株価データをyfinanceからダウンロード
    Args:
        ticker_symbol (str): 株式市場のティッカーシンボル
        start_date (str): データ取得開始日 ('YYYY-MM-DD'形式)
        end_date (str): データ取得終了日 ('YYYY-MM-DD'形式)
    Returns:
        pd.DataFrame: OHLC（始値、高値、安値、終値）、出来高などを含む株価データのDataFrame
    Raises:
        ValueError: 指定されたティッカーシンボルのデータが見つからない場合に発生
    """
    print(f"'{ticker_symbol}'の株価データを取得中...")
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    if stock_data.empty:
        raise ValueError(f"ティッカーシンボル '{ticker_symbol}' のデータが見つかりませんでした。")
    print("データ取得完了。")
    print("データの先頭5行:")
    print(stock_data.head())
    return stock_data

def create_dataset(dataset: np.ndarray, time_step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    時系列データをLSTMが学習できるシーケンス形式（入力データと教師データ）に変換
    Args:
        dataset (np.ndarray): 変換対象の時系列データ
        time_step (int): 1つの入力シーケンスを構成する日数（ステップ数）。デフォルトは1
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - dataX (np.ndarray): 入力シーケンスデータ
            - dataY (np.ndarray): 各入力シーケンスに対応する教師データ（次の日の値）
    """
    # dataXに60日分のデータ(問題)を、dataYにその直後の日のデータ(答え)を格納
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        input_sequence = dataset[i:(i + time_step), 0]
        dataX.append(input_sequence)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def preprocess_data(stock_data: pd.DataFrame, time_step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler, int]:
    """
    データの前処理（正規化、訓練/テスト分割、シーケンス作成）を行う
    Args:
        stock_data (pd.DataFrame): 元の株価データ
        time_step (int): 1つの入力シーケンスを構成する日数
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler, int]:
            - X_train (torch.Tensor): 訓練用の入力シーケンスデータ
            - y_train (torch.Tensor): 訓練用の教師データ
            - X_test (torch.Tensor): テスト用の入力シーケンスデータ
            - y_test (torch.Tensor): テスト用の教師データ
            - scaler (MinMaxScaler): データの正規化に使用したMinMaxScalerオブジェクト
            - training_data_len (int): 訓練データの長さ
    """
    print("\n--- データ前処理開始 ---")
    # 株価データの中から終値(その日の最終的な値段)のみを抽出
    close_prices = stock_data['Close'].values.reshape(-1, 1)

    # データの正規化
    # 株価〇ドル～〇ドル → 0 ～ 1 の範囲にスケーリング
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices) # scalerに合わせて、実際のデータに正規化を実行

    # 訓練データとテストデータに分割
    training_data_len = int(len(scaled_prices) * 0.8) # 80%をモデルの学習用に、残りの20%をモデルの性能評価用にする
    train_data = scaled_prices[0:training_data_len, :]

    # 訓練用シーケンスデータの作成
    X_train, y_train = create_dataset(train_data, time_step)

    # テスト用シーケンスデータの作成
    test_data_inputs = scaled_prices[training_data_len - time_step:, :] # テストデータの直近60日分を取得
    X_test, y_test = create_dataset(test_data_inputs, time_step)

    # データをPyTorch Tensorに変換(誤差逆伝播用)
    # また、nn.LSTM、nn.LinearといったPyTorchのモデルに入力できる形にする必要があるから
    X_train = torch.from_numpy(X_train).float().reshape(-1, time_step, 1)
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float().reshape(-1, time_step, 1)
    y_test = torch.from_numpy(y_test).float()

    print("--- データ前処理完了 ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test, scaler, training_data_len

def train_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int) -> None:
    """
    LSTMモデルを学習させる関数
    Args:
        model (nn.Module): 学習させるLSTMモデルのインスタンス
        X_train (torch.Tensor): 訓練用の入力シーケンスデータ
        y_train (torch.Tensor): 訓練用の教師データ
        epochs (int): 学習のエポック数
    Returns:
        None
    """
    print("\n--- モデル学習開始 ---")
    loss_function = nn.MSELoss() # 平均二乗誤差の損失関数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam: パラメータの更新量を自動で調整してくれるアルゴリズム

    for i in range(epochs):
        model.train() # モデルを学習モードに設定
        optimizer.zero_grad() # 前回の勾配の初期化
        y_pred = model(X_train) # X_train(問題)を解かせ、算出された予測値を格納(順伝播)
        single_loss = loss_function(y_pred.squeeze(), y_train) # 予測値と正解値の誤差を計算
        single_loss.backward() # 誤差逆伝播で勾配を計算
        optimizer.step() # 逆伝播で導き出された勾配を使って実際にモデルのパラメータを更新

        # 10エポックごとに、途中経過として損失を表示
        if (i + 1) % 10 == 0:
            print(f'Epoch: {i+1:3} Loss: {single_loss.item():10.8f}')
    print("--- モデル学習完了 ---")

def evaluate_and_plot(model: nn.Module, stock_data: pd.DataFrame, scaler: MinMaxScaler, X_test: torch.Tensor, training_data_len: int, ticker_symbol: str) -> None:
    """
    学習済みモデルで予測を行い、結果を実績値とともにグラフにプロットする
    Args:
        model (nn.Module): 学習済みのLSTMモデル
        stock_data (pd.DataFrame): 元の株価データ
        scaler (MinMaxScaler): データの正規化に使用したMinMaxScalerオブジェクト
        X_test (torch.Tensor): テスト用の入力シーケンスデータ
        training_data_len (int): 訓練データの長さ
        ticker_symbol (str): グラフのタイトルに表示するティッカーシンボル
    Returns:
        None
    """
    print("\n--- 予測と結果の可視化開始 ---")

    model.eval() # 評価モード
    with torch.no_grad():
        if X_test.shape[0] == 0:
            print("テストシーケンスが1つも作れませんでした（テスト期間が短すぎます）")
            predictions = np.array([]).reshape(-1, 1)
        else:
            test_predictions = model(X_test)
            # torch → numpy に変換
            predictions = scaler.inverse_transform(test_predictions.cpu().numpy())

    # predictions を確実に numpy 配列（1次元または2次元）に統一
    predictions = np.asarray(predictions) # リストやDataFrameでもnumpyに変換
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions.flatten() # (n,1) → (n,)
    elif predictions.ndim > 2:
        predictions = predictions.squeeze() # 余計な次元を削除

    print(f"predictions shape: {predictions.shape}")
    print(f"predictions length: {len(predictions)}")

    # bias補正
    if len(predictions) > 0:
        # 訓練データの最後の実績値（floatに変換）
        last_train_value = float(stock_data['Close'].iloc[training_data_len - 1])

        # predictions の最初の値を確実にスカラーとして取得
        first_pred = float(predictions[0])

        bias = last_train_value - first_pred
        print(f"last_train_value: {last_train_value:.4f}")
        print(f"first_pred:       {first_pred:.4f}")
        print(f"bias:             {bias:.4f}")

        # ここで broadcasting が確実に働くように numpy で加算
        predictions = predictions + bias
    else:
        print("予測値がありません。bias補正をスキップします。")
        predictions = np.array([])

    # 表示用データ準備
    train_display_data = stock_data['Close'].iloc[:training_data_len]

    # iloc を使って確実にスライス（ラベルインデックス問題を回避）
    valid_data = stock_data.iloc[training_data_len:].copy()

    print(f"train_display_data length: {len(train_display_data)}")
    print(f"valid_data length:         {len(valid_data)}")

    # Predictions列の挿入
    if len(predictions) == len(valid_data) and len(predictions) > 0:
        valid_data['Predictions'] = predictions
        print("Predictions列を正常に追加しました")
    else:
        print("警告：長さが一致しません → Predictions列は追加しません")
        print(f"  predictions の長さ: {len(predictions)}")
        print(f"  valid_data の長さ : {len(valid_data)}")
        valid_data['Predictions'] = np.nan  # とりあえずNaNで表示

    # ── グラフ描画 ──
    plt.figure(figsize=(16, 8))
    plt.title(f'{ticker_symbol} Stock Price Prediction', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)

    plt.plot(train_display_data, 'b', label='Train Actual Price')
    plt.plot(valid_data['Close'], 'orange', label='Test Actual Price')

    # Predictionsがある場合だけプロット
    if 'Predictions' in valid_data.columns and not valid_data['Predictions'].isna().all():
        plt.plot(valid_data['Predictions'], 'green', linestyle='--', label='Predicted Price')

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    print("--- 予測と結果の可視化完了 ---")

def main() -> None:
    """
    LSTMによる株価予測プログラムのメイン処理関数
    Args:
        None
    Returns:
        None
    """
    # --- 設定値 ---
    ticker_symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2025-01-31' # 予測期間を少し伸ばして表示

    try:
        # 1. データの取得
        stock_data = get_stock_data(ticker_symbol, start_date, end_date)

        # 2. データの前処理
        X_train, y_train, X_test, y_test, scaler, training_data_len = preprocess_data(stock_data, TIME_STEP)

        # 3. モデルのインスタンス化
        model = LSTM()
        print("\n--- モデル定義 ---")
        print(model)

        # 4. モデルの学習
        train_model(model, X_train, y_train, EPOCHS)

        # 5. 評価と結果の可視化
        evaluate_and_plot(model, stock_data, scaler, X_test, training_data_len, ticker_symbol)

    except ValueError as e:
        print(f"エラーが発生しました: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


if __name__ == '__main__':
    main()