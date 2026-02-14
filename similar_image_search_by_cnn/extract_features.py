import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# 画像が保存されているフォルダのパス
DATASET_DIR = Path("dataset_images")
# 特徴量の出力先ファイルパス
OUTPUT_FILEPATH = Path("features.pkl")

class FeatureExtractor:
    """
    事前学習済みのVGG16モデルを使用して、画像から特徴量を抽出するクラス
    Attributes:
        model (torch.nn.Module): 特徴抽出に使用する事前学習済みVGG16モデル
        transform (transforms.Compose): 画像の前処理パイプライン
    """

    def __init__(self) -> None:
        """
        FeatureExtractorのインスタンスを初期化
        Args:
            None
        Returns:
            None
        """
        # VGG16モデルをロードし、事前学習済みの重みを使用する
        # weights=models.VGG16_Weights.DEFAULT: 最新の推奨される重みを指定する方法
        self.model: torch.nn.Module = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # VGG16の最終的な分類層は特徴抽出には不要なため、その手前までを取得
        # self.model.features は畳み込み層とプーリング層（特徴抽出部）
        # self.model.avgpool は特徴マップをベクトル化する部分
        # 最後の self.model.classifier は全結合層（分類部）なので今回は使わない
        
        # モデルを評価モードに設定
        # これにより、学習時にのみ使われるDropoutなどが無効になる
        self.model.eval()

        # 画像をモデルに入力するために必要な前処理を定義
        self.transform: transforms.Compose = transforms.Compose([
            transforms.Resize(224), # 画像サイズを224x224にリサイズ
            transforms.CenterCrop(224), # 画像の中央部分を224x224で切り出す
            transforms.ToTensor(), # 画像をPyTorchのテンソル形式に変換
            # ImageNetの学習時に使われた平均と標準偏差で正規化
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, img_path: Path) -> Optional[np.ndarray]:
        """
        指定されたパスの画像から特徴量を抽出
        Args:
            img_path (Path): 特徴量を抽出したい画像のファイルパス
        Returns:
            Optional[np.ndarray]: 抽出された特徴量ベクトル(1次元のNumpy配列)
                                  エラーが発生した場合はNoneを返す
        """
        try:
            # 画像ファイルを開き、RGB形式に変換(VGG16はRGB画像を想定しているため)
            img = Image.open(img_path).convert("RGB")
            # 定義した前処理を適用
            img_tensor = self.transform(img)

            # PyTorchでは通常、バッチ単位でデータを扱う([バッチサイズ, チャネル数, 高さ, 幅])
            # バッチサイズは１(画像は１つづつのみ)だが、バッチ形式（[1, C, H, W]）に合わせる
            img_tensor = img_tensor.unsqueeze(0)
            
            # 勾配計算を無効にする（推論時には不要で、メモリ効率が良くなる）
            with torch.no_grad():
                # モデルに画像を入力し、特徴マップを取得
                features = self.model.features(img_tensor)
                
                # 特徴マップをベクトル化
                vector = self.model.avgpool(features)

                # バッチの次元を削除して1次元ベクトルに変換
                vector = torch.flatten(vector, 1)

            # PyTorchテンソルをCPU上のNumpy配列に変換して返す
            return vector.cpu().numpy()
        
        except FileNotFoundError:
            print(f"エラー: ファイルが見つかりません: {img_path}")
            return None
        except Exception as e:
            print(f"エラー: {img_path} の処理中に問題が発生しました: {e}")
            return None


def main() -> None:
    """
    データセット内の全画像から特徴量を抽出し、ファイルに保存するメイン処理
    Args:
        None
    Returns:
        None
    """
    # 特徴抽出器をインスタンス化
    extractor = FeatureExtractor()

    # フォルダ内のすべての画像ファイルパスを取得
    # .pngで終わるファイルのみを対象とする
    image_paths: List[Path] = sorted(list(DATASET_DIR.glob("*.png")))
    
    if not image_paths:
        print(f"エラー: '{DATASET_DIR}' フォルダ内に.png画像が見つかりませんでした。")
        return

    # 特徴量とファイルパスを保存するためのリスト
    # (ファイルパス(str), 特徴量ベクトル(np.ndarray)) のタプルを格納
    all_features: List[Tuple[str, np.ndarray]] = []

    print(f"{len(image_paths)}枚の画像から特徴量を抽出します...")

    # tqdmを使ってプログレスバーを表示しながらループ処理
    for img_path in tqdm(image_paths):
        # 特徴量を抽出
        feature = extractor.extract(img_path)
        
        if feature is not None:
            # (ファイルパス, 特徴量ベクトル) のタプルとしてリストに追加
            # Pathオブジェクトを文字列に変換して保存する
            all_features.append((str(img_path), feature))
    
    # 特徴量リストをファイルに保存
    # pickleはPythonのオブジェクト（今回はリスト）をそのままファイルに保存/復元できるライブラリ
    with open(OUTPUT_FILEPATH, "wb") as f:
        pickle.dump(all_features, f)
        
    print(f"---")
    print(f"完了: {len(all_features)}個の特徴量を '{OUTPUT_FILEPATH}' に保存しました。")
    print(f"---")


if __name__ == "__main__":
    main()