import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple


# 設定
@dataclass
class TrainingConfig:
    """学習に関する設定を管理するデータクラス"""
    sequence_size: int = 3
    n_samples: int = 1000
    n_anomalies: int = 50
    anomaly_magnitude: float = 5.0
    n_epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    embedding_dim: int = 128
    model_path: str = 'lstm_autoencoder.pth'
    x_test_path: str = 'X_test.pt'
    scaler_path: str = 'scaler.npy'
    original_data_path: str = 'original_data.npy'


# モデル定義
class Lambda(nn.Module):
    """
    任意の関数を適用するためのカスタムnn.Moduleラッパー
    nn.Sequential内に記述するために定義
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class Reshape(nn.Module):
    """
    Tensorの形状を変更するためのカスタムnn.Moduleラッパー
    nn.Sequential内に記述するために定義
    """
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class LstmAutoencoder(nn.Module):
    """LSTMベースのオートエンコーダモデル"""
    def __init__(self, seq_len: int, n_features: int, embedding_dim: int = 64):
        super(LstmAutoencoder, self).__init__()

        # nn.Sequential: コンテナに追加した順番通りに、自動的にデータを通過させてくれる機能
        self.encoder = nn.Sequential(
            nn.LSTM(n_features, 128, batch_first=True), # 時系列のパターンを学習し、各時点での特徴を捉えたテンソルと、内部状態のタプルを出力( (バッチサイズ, シーケンス長, 128), (h, c) )
            Lambda(lambda x: x[0]), # LSTMが出力したタプルから、最初の要素である特徴テンソルだけを取り出す (バッチサイズ, シーケンス長, 128)
            # LSTMの3D出力(batch, seq, feature)を2D(batch, seq*feature)に平坦化
            nn.Flatten(start_dim=1),# nn.Linear層は3次元のデータを扱えないため、シーケンス長と特徴量の次元を一つにまとめ（平坦化）て、2次元のベクトルに変換(バッチサイズ, シーケンス長 * 128)
            nn.Linear(seq_len * 128, embedding_dim) # 平坦化されたベクトルを受け取り、最終的なembedding_dim次元のベクトルに変換(バッチサイズ, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, seq_len * 128), # 圧縮ベクトルを、元の平坦化されたベクトルの次元に復元
            Reshape(-1, seq_len, 128), # 平坦なベクトルを、次のLSTM層が扱える3次元のシーケンス形式に戻す
            nn.LSTM(128, n_features, batch_first=True), # 3次元のシーケンス情報を受け取り、元の時系列データを再構成
            Lambda(lambda x: x[0]) # LSTMの出力タプルから、再構成された時系列データのテンソルだけを取り出す
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力データをエンコードしてからデコードする
        Args:
            x (torch.Tensor): 入力データ (バッチサイズ, シーケンス長, 特徴量数)
        Returns:
            torch.Tensor: 再構成されたデータ (バッチサイズ, シーケンス長, 特徴量数)
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# データ処理
def generate_data(config: TrainingConfig) -> pd.DataFrame:
    """
    サイン波に基づいた正常データと、ランダムな異常を含む擬似時系列データを生成する
    Args:
        config (TrainingConfig): データ生成に関する設定
    Returns:
        pd.DataFrame: 'value'と'is_anomaly'列を持つデータフレーム
    """
    print("擬似データを生成します。")
    time = np.linspace(0, 100, config.n_samples) # 時間軸を生成([0.0, 0.1, ..., 100.0])
    # np.sin(time): サイン波
    # np.random.normal(0, 0.2, config.n_samples): 平均0、標準偏差0.2の正規分布に従う乱数を1000個生成
    # 上記２つを足し合わせることで、多少の揺らぎを持つ通常値データを作成
    normal_data = np.sin(time) + np.random.normal(0, 0.2, config.n_samples)
    
    anomaly_indices = np.random.choice(config.n_samples, config.n_anomalies, replace=False) # 異常値を挿入する場所をランダムに選択
    anomalous_data = normal_data.copy() # 正常データのコピーを作成し、異常値を挿入するためのベースとする
    anomalous_data[anomaly_indices] += np.random.normal(0, config.anomaly_magnitude, config.n_anomalies) # コピーしたデータを異常値で上書き
    
    # 作られたデータをPandasのデータフレームにまとめる
    df = pd.DataFrame({
        'value': anomalous_data, # 全体の時系列データ
        'is_anomaly': [1 if i in anomaly_indices else 0 for i in range(config.n_samples)] # 時系列データのなかで、異常値が挿入された位置を1、そうでない位置を0とする正解ラベル
    })
    return df

def create_sequences(data: np.ndarray, seq_length: int) -> np.ndarray:
    """
    時系列データからスライディングウィンドウを用いてシーケンスを作成する
    Args:
        data (np.ndarray): 入力データ
        seq_length (int): 1シーケンスの長さ
    Returns:
        np.ndarray: シーケンスの配列
    """
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length]) # データのうち、seq_length分の長さを１つのシーケンスとし、リストに追加
    return np.array(sequences)

def preprocess_data(df: pd.DataFrame, config: TrainingConfig) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
    """
    データを正規化し、学習用および評価用のシーケンスに変換する
    Args:
        df (pd.DataFrame): 入力データフレーム
        config (TrainingConfig): 前処理に関する設定
    Returns:
        Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]: 学習データ、評価データ、学習に使用したスケーラーのタプル
    """
    # 正常データのみを使用してスケーラーを学習
    # .reshape(-1, 1): (-1,: データの行数は全部 1): 1列の配列に変換
    normal_values = df[df['is_anomaly'] == 0]['value'].values.reshape(-1, 1) # 学習のため、Numpyの2次元配列に変換
    scaler = MinMaxScaler(feature_range=(-1, 1)) # -1～1の範囲で正規化するスケーラーを作成
    scaler = scaler.fit(normal_values) # 正規化実行
    print("正常データを用いてスケーラーを学習しました。")

    # 全データを正規化
    all_values = df['value'].values.reshape(-1, 1)
    all_data_normalized = scaler.transform(all_values)
    train_data_normalized = scaler.transform(normal_values)
    print("データの正規化が完了しました。")

    # シーケンスを作成
    X_train = create_sequences(train_data_normalized, config.sequence_size)
    X_test = create_sequences(all_data_normalized, config.sequence_size)
    print("スライディングウィンドウ処理が完了しました。")

    # PyTorch Tensorに変換
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    print(f"学習データ (X_train) の形状: {X_train_tensor.shape}")
    
    return X_train_tensor, X_test_tensor, scaler


# 学習処理
def train_model(model: LstmAutoencoder, train_loader: DataLoader, config: TrainingConfig) -> LstmAutoencoder:
    """
    モデルの学習を実行する
    Args:
        model (LstmAutoencoder): 学習対象のモデル
        train_loader (DataLoader): 学習データローダー
        config (TrainingConfig): 学習に関する設定
    Returns:
        LstmAutoencoder: 学習済みモデル
    """
    loss_fn = nn.MSELoss(reduction='none') # 損失関数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) # 自動最適化アルゴリズム
    
    print("学習を開始します...")
    for epoch in range(config.n_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad() # 勾配の初期化
            reconstructed = model(batch) # 順伝播
            loss = loss_fn(reconstructed, batch).mean() # 損失の計算
            loss.backward() # 逆伝播
            optimizer.step() # パラメータの更新
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"エポック [{epoch+1}/{config.n_epochs}] - 損失: {avg_loss:.6f}")
    
    print("学習が完了しました。")
    return model


# 保存処理
def save_artifacts(model: LstmAutoencoder, x_test: torch.Tensor, scaler: MinMaxScaler, df: pd.DataFrame, config: TrainingConfig) -> None:
    """
    学習済みモデルと関連データ（評価データ、スケーラー、元データ）を保存する
    Args:
        model (LstmAutoencoder): 学習済みモデル
        x_test (torch.Tensor): 評価データ
        scaler (MinMaxScaler): 学習に使用したスケーラー
        df (pd.DataFrame): 元のデータフレーム
        config (TrainingConfig): 保存パスに関する設定
    Returns:
        None
    """
    torch.save(model.state_dict(), config.model_path)
    torch.save(x_test, config.x_test_path)
    np.save(config.scaler_path, np.array([scaler]))
    np.save(config.original_data_path, df.to_numpy())
    
    print(f"モデルを '{config.model_path}' として保存しました。")
    print("評価用の関連データも保存しました。")


# メイン処理
def main() -> None:
    """
    異常検知モデルの学習パイプラインを実行するメイン処理
    Args:
        None
    Returns:
        None
    """
    print("--- 異常検知モデルの学習パイプラインを開始 ---")
    
    # 1. 設定の初期化
    config = TrainingConfig()
    
    # 2. データ準備と前処理
    print("\n--- Step 1: データ準備と前処理 ---")
    df = generate_data(config)
    x_train, x_test, scaler = preprocess_data(df, config)
    print("--- Step 1: 完了 ---")
    
    # 3. モデルの構築と学習の準備
    print("\n--- Step 2: モデル構築と学習準備 ---")
    model = LstmAutoencoder(
        seq_len=config.sequence_size,
        n_features=x_train.shape[2],
        embedding_dim=config.embedding_dim
    )
    train_loader = DataLoader(x_train, batch_size=config.batch_size, shuffle=True)
    print("--- Step 2: 完了 ---")
    
    # 4. 学習の実行
    print("\n--- Step 3: 学習 ---")
    trained_model = train_model(model, train_loader, config)
    print("--- Step 3: 完了 ---")
    
    # 5. モデルと関連データの保存
    print("\n--- Step 4: モデルとデータの保存 ---")
    save_artifacts(trained_model, x_test, scaler, df, config)
    print("--- Step 4: 完了 ---")
    
    print("\n--- パイプラインが正常に完了しました ---")

if __name__ == '__main__':
    main()