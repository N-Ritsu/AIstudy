import torch
import torchvision
from PIL import Image
import os

# 画像を保存するフォルダの名前
SAVE_DIR = "dataset_images"

# 今回抽出したいクラスのインデックス ( 'cat' は 3番目)
TARGET_CLASS_IDX = 3
TARGET_CLASS_NAME = "cat"

# 保存する画像の最大枚数
NUM_IMAGES_TO_SAVE = 500

def prepare_dataset() -> None:
    """
    CIFAR-10データセットをダウンロードし、特定のクラスの画像を指定されたフォルダに画像ファイルとして保存
    Args:
        None
    Returns:
        None
    """
    print("CIFAR-10データセットのダウンロードを開始します...")
    
    # CIFAR-10の訓練データをダウンロードする
    # PIL.Image形式でデータを取得する
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    
    print("ダウンロードが完了しました。")

    # 画像の保存先フォルダを作成
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"フォルダ '{SAVE_DIR}' を作成しました。")

    saved_count = 0
    print(f"クラス '{TARGET_CLASS_NAME}' (インデックス: {TARGET_CLASS_IDX}) の画像を抽出・保存します...")
    
    # データセット全体をループして、目的のクラスの画像を探す
    for i, (image, label) in enumerate(trainset):
        if label == TARGET_CLASS_IDX:
            # ファイル名を生成 (例: cat_001.png, cat_002.png, ...)
            # zfill(3)は、数値を3桁のゼロ埋め文字列にする (1 -> "001")
            filename = f"{TARGET_CLASS_NAME}_{str(saved_count+1).zfill(3)}.png"
            filepath = os.path.join(SAVE_DIR, filename)
            
            # PIL.Imageオブジェクトとして取得した画像をファイルに保存
            image.save(filepath)
            
            saved_count += 1
            
            # 指定した枚数に達したらループを終了
            if saved_count >= NUM_IMAGES_TO_SAVE:
                break
    
    if saved_count > 0:
        print(f"---")
        print(f"完了: {saved_count}枚の画像をフォルダ '{SAVE_DIR}' に保存しました。")
        print(f"---")
    else:
        print("エラー: 指定されたクラスの画像が見つかりませんでした。")

if __name__ == "__main__":
    prepare_dataset()