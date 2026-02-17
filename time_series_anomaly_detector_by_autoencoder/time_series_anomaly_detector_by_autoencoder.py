import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from typing import Tuple, List
from run_anomaly_detection_training import LstmAutoencoder, Lambda, Reshape 


# 設定
@dataclass
class EvaluationConfig:
    """評価に関する設定を管理するデータクラス"""
    sequence_size: int = 3
    embedding_dim: int = 128
    threshold_percentile: float = 90.0
    model_path: str = 'lstm_autoencoder.pth'
    x_test_path: str = 'X_test.pt'
    scaler_path: str = 'scaler.npy'
    original_data_path: str = 'original_data.npy'


# データ処理・モデル読み込み
def load_artifacts(config: EvaluationConfig) -> Tuple[LstmAutoencoder, torch.Tensor, pd.DataFrame]:
    """
    学習済みのモデルと評価に必要なデータをファイルから読み込む
    Args:
        config (EvaluationConfig): ファイルパスなどの設定
    Returns:
        Tuple[LstmAutoencoder, torch.Tensor, pd.DataFrame]: 
            - model: 学習済みモデル
            - x_test: 評価データ(X_test)
            - df: 元のデータフレーム
    """
    print("--- 学習済みモデルとデータを読み込んでいます... ---")
    # 学習時と同じようにLstmAutoencoderクラスのインスタンスを生成
    model = LstmAutoencoder(
        seq_len=config.sequence_size,
        n_features=1, # 特徴量は1つ
        embedding_dim=config.embedding_dim
    )
    model.load_state_dict(torch.load(config.model_path))
    model.eval() # 評価モードへ

    x_test = torch.load(config.x_test_path) # すでに作成されたデータであるX_test.ptを読み込む
    df_np = np.load(config.original_data_path, allow_pickle=True) # すでに作成されたデータであるoriginal_data.npyを読み込む
    df = pd.DataFrame(df_np, columns=['value', 'is_anomaly']) # 読み込んだNumPy配列をPandasのデータフレームに変換し、列名を設定
    
    print("読み込みが完了しました。\n")
    return model, x_test, df


# 異常検知ロジック
def calculate_reconstruction_errors(model: LstmAutoencoder, x_test: torch.Tensor) -> List[float]:
    """
    テストデータセットの各シーケンスに対して再構成誤差（MSE）を計算する
    異常なデータが含まれていた場合、うまく復元できないことから、再構成誤差が大きくなると予想される
    Args:
        model (LstmAutoencoder): 学習済みモデル
        x_test (torch.Tensor): 評価データ
    Returns:
        List[float]: 各シーケンスの再構成誤差のリスト
    """
    print("--- 全テストデータに対して再構成誤差を計算しています... ---")
    reconstruction_errors = []
    loss_fn = nn.MSELoss() # 平均二乗誤差
    with torch.no_grad(): # 評価モードで勾配計算を無効化
        for seq in x_test:
            # .unsqueeze(0): 0番目の位置にサイズ1の次元を追加
            seq = seq.unsqueeze(0) # (シーケンス長, 特徴量数)の次元を、(バッチサイズ, シーケンス長, 特徴量数)にする(PyTorchで処理するため)
            reconstructed = model(seq) # 順伝播
            loss = loss_fn(reconstructed, seq) # 再構成誤差の計算
            reconstruction_errors.append(loss.item()) # 再構成誤差をリストに格納
    print("計算が完了しました。\n")
    return reconstruction_errors

def detect_anomalies(errors: List[float], config: EvaluationConfig) -> Tuple[List[int], List[int], float]:
    """
    再構成誤差から異常点を検出する
    中心点評価と中央点フィルタリングの2段階の処理を行う
    Args:
        errors (List[float]): 再構成誤差のリスト
        config (EvaluationConfig): しきい値のパーセンタイルやシーケンスサイズなどの設定
    Returns:
        Tuple[List[int], List[int], float]:
            - final_indices: 最終的に異常と判断された点のインデックスリスト
            - candidate_indices: フィルタリング前の異常候補点のインデックスリスト
            - threshold: 異常判定に用いたしきい値
    """
    print("--- 異常候補の中心点評価と、中央点フィルタリングを行っています... ---")
    
    # 誤差分布の上位パーセンタイルをしきい値として設定
    threshold = np.percentile(errors, config.threshold_percentile)
    print(f"決定したしきい値 (上位{100-config.threshold_percentile}%): {threshold:.6f}")

    # --- 処理1: 中心点評価 ---
    # スライディングウィンドウの中心をそのウィンドウの代表点とするため、インデックスをずらす
    offset = config.sequence_size // 2
    # しきい値を超えたシーケンスを異常「候補」とし、その中心点のインデックスを保存する
    candidate_indices = np.where(np.array(errors) > threshold)[0] + offset
    print(f"中心点評価により、{len(candidate_indices)}件の異常候補を検出しました。")

    # --- 処理2: 中央点フィルタリング ---
    # 連続して異常と判定された候補点を1つのグループとみなし、その中央点のみを最終的な異常点とする
    # これにより、1つの異常イベントに対して複数の点が検出されるのを防ぐ
    final_anomalies_indices = []
    if len(candidate_indices) > 0:
        current_group = [candidate_indices[0]]
        for i in range(1, len(candidate_indices)):
            # 候補点同士のインデックスが近いなら、同じ異常イベントグループとみなす
            if candidate_indices[i] - candidate_indices[i-1] <= 2:
                current_group.append(candidate_indices[i])
            else:
                # 候補点のインデックスが離れていたら、前のグループは終了
                # これまでのグループの中央値（median）にあたる点を、そのグループの代表として採用
                median_index = current_group[len(current_group) // 2]
                final_anomalies_indices.append(median_index)
                # 新しいグループを開始する
                current_group = [candidate_indices[i]]
        
        # ループ終了後、最後のグループの中央点も追加
        median_index = current_group[len(current_group) // 2]
        final_anomalies_indices.append(median_index)

    print(f"中央点フィルタリングにより、異常点を{len(final_anomalies_indices)}件に絞り込みました。\n")
    return final_anomalies_indices, candidate_indices.tolist(), threshold


# 可視化
def visualize_results(
    df: pd.DataFrame, 
    errors: List[float], 
    final_indices: List[int], 
    candidate_indices: List[int], 
    threshold: float, 
    config: EvaluationConfig
) -> None:
    """
    異常検知の結果をグラフに描画する
    Args:
        df (pd.DataFrame): 元のデータフレーム
        errors (List[float]): 再構成誤差のリスト
        final_indices (List[int]): 最終的な異常点のインデックス
        candidate_indices (List[int]): フィルタリング前の候補点のインデックス
        threshold (float): 異常判定のしきい値
        config (EvaluationConfig): シーケンスサイズなどの設定
    Returns:
        None
    """
    print("--- 結果を可視化しています... ---")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    plt.style.use('ggplot')

    # --- グラフ1: 時系列データと検出された異常点 ---
    ax1.plot(df.index, df['value'], label='Sensor Value', color='blue', linewidth=0.8)
    # 正解の異常点をプロット
    true_anomalies = df[df['is_anomaly'] == 1]
    ax1.scatter(true_anomalies.index, true_anomalies['value'], color='red', marker='o', s=50, label='True Anomaly')
    # 最終的に検出された異常点をプロット
    detected_points = df.iloc[final_indices]
    ax1.scatter(detected_points.index, detected_points['value'], color='darkorange', marker='X', s=80, label='Predicted Anomaly')
    ax1.set_title('Time Series Data with Detected Anomalies (Median Filter)', fontsize=16)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(loc='upper left')

    # --- グラフ2: 再構成誤差と異常候補点 ---
    offset = config.sequence_size // 2
    error_indices = np.arange(len(errors)) + offset
    ax2.plot(error_indices, errors, label='Reconstruction Error', color='green', linewidth=0.9)
    ax2.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    # フィルタリング前の全候補点を参考としてプロット
    ax2.scatter(candidate_indices, np.array(errors)[np.array(candidate_indices) - offset], 
                marker='.', color='gray', s=50, label='All Candidates (before filtering)')
    # 最終的に採用された中央点を強調してプロット
    final_error_indices = np.array(final_indices) - offset
    ax2.scatter(final_indices, np.array(errors)[final_error_indices], 
                marker='v', color='purple', s=100, label='Final Detections (Median)')
    ax2.set_title('Reconstruction Error with Median Filtering', fontsize=16)
    ax2.set_xlabel('Time Point (Sequence Center)', fontsize=12)
    ax2.set_ylabel('MSE Loss', fontsize=12)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Autoencoderによる時系列データの異常検知システムのメイン処理
    Args:
        None
    Returns:
        None
    """
    print("--- 異常検知モデルの評価パイプラインを開始 ---")
    
    # 1. 設定の初期化
    config = EvaluationConfig()
    
    # 2. 学習済みモデルとデータの読み込み
    model, x_test, df = load_artifacts(config)
    
    # 3. 再構成誤差の計算
    reconstruction_errors = calculate_reconstruction_errors(model, x_test)
    
    # 4. 誤差に基づいた異常検知
    final_indices, candidate_indices, threshold = detect_anomalies(reconstruction_errors, config)

    # 5. 結果の可視化
    visualize_results(df, reconstruction_errors, final_indices, candidate_indices, threshold, config)
    
    print("\n--- パイプラインが正常に完了しました ---")

if __name__ == '__main__':
    main()