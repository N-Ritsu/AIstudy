import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from extract_features import FeatureExtractor

FEATURES_FILE = Path("features.pkl")
QUERY_IMAGE_PATH = Path("dataset_images/cat_101.png")
NUM_RESULTS = 5

def load_features(filepath: Path) -> Optional[Tuple[np.ndarray, List[str]]] | None:
    """
    pickle形式で保存された特徴量データベースを読み込む
    Args:
        filepath (Path): 特徴量データベースのファイルパス
    Returns:
        Optional[Tuple[np.ndarray, List[str]]]: 特徴量ベクトルの配列と対応する画像ファイルパスのリスト
                                               それぞれが存在しない場合はNoneを返す
    """
    if not filepath.is_file():
        print(f"エラー: 特徴量ファイルが見つかりません: {filepath}")
        return None
    
    with open(filepath, "rb") as f:
        all_features = pickle.load(f) # バイナリ形式で保存されたデータを読み込み、pythonのリストに復元
        
    # all_features: (ファイルパス, 特徴量ベクトル)のタプル → item[0]: ファイルパス, item[1]: 特徴量ベクトル
    db_features = np.array([item[1] for item in all_features])
    db_filepaths = [item[0] for item in all_features]
    return db_features, db_filepaths


def display_results(query_path: Path, results: List[Tuple[str, float]]) -> None:
    """
    クエリ画像と検索結果の画像を並べて表示する
    Args:
        query_path (Path): クエリ画像のファイルパス
        results (List[Tuple[str, float]]): 検索結果の画像ファイルパスと類似度のタプルのリスト
    Returns:
        None
    """
    num_results = len(results)
    # axes には6個の描画領域オブジェクトが入る(axes[0], axes[1], ..., axes[5])
    fig, axes = plt.subplots(1, num_results + 1, figsize=(15, 4))
    
    # axes[0]にクエリ画像を表示
    query_img = Image.open(query_path)
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image")
    axes[0].axis('off')

    # axes[1]~axes[5]
    for i, (result_path, result_similarity) in enumerate(results):
        result_img = Image.open(result_path)
        ax = axes[i + 1]
        ax.imshow(result_img)
        ax.set_title(f"Rank {i+1}\nSim: {result_similarity:.4f}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    類似画像検索エンジンのメイン処理
    Args:
        None
    Returns:
        None
    """
    # 1. 特徴抽出器をインスタンス化
    extractor = FeatureExtractor()

    # 2. 特徴量データベースをファイルから読み込み
    loaded_data = load_features(FEATURES_FILE)
    if loaded_data is None:
        return
    db_features, db_filepaths = loaded_data
    
    # 3. 検索対象の画像（クエリ画像）の特徴量を抽出
    if not QUERY_IMAGE_PATH.is_file():
        print(f"エラー: クエリ画像が見つかりません: {QUERY_IMAGE_PATH}")
        return
    query_feature = extractor.extract(QUERY_IMAGE_PATH) # 特徴量ベクトルを計算
    if query_feature is None:
        return

    # 4. 全画像とのコサイン類似度(-1 ~ 1)を計算
    similarities = cosine_similarity(query_feature, db_features).flatten()
    # 5. 類似度が高い順にインデックスをソート
    sorted_indices = np.argsort(similarities)[::-1]
    # 6. 上位N件の結果を取得
    top_indices = sorted_indices[1:NUM_RESULTS + 1]
    results: List[Tuple[str, float]] = [
        (db_filepaths[idx], similarities[idx]) for idx in top_indices
    ]

    # 7. 結果を表示
    print(f"検索画像: {QUERY_IMAGE_PATH}")
    print("---------------------------------")
    print(f"検索結果 (上位{NUM_RESULTS}件):")
    for i, (path, sim) in enumerate(results):
        print(f"  - Rank {i+1}: 類似度: {sim:.4f}, ファイル: {path}")

    display_results(QUERY_IMAGE_PATH, results)


if __name__ == "__main__":
    main()