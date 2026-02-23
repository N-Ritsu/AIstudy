import cv2
import numpy as np
from ultralytics import YOLO

INPUT_IMAGE = 'image.jpg'
OUTPUT_PATH = 'output.jpg'

def draw_detection_boxes(image_path: str, output_path: str) -> None:
    """
    指定された画像内のオブジェクトをYOLOv8モデルで検出し、検出された各オブジェクトにバウンディングボックスとクラス名、信頼度スコアを描画する
    Args:
        image_path (str): 入力となる画像ファイルのパス
        output_path (str, optional): 処理結果を保存する画像のファイルパス
    Returns:
        None: この関数は値を返さず、結果を指定されたパスに画像ファイルとして保存します。
    """
    # 事前学習済みのYOLOv8モデルをロード
    # バウンディングボックスの情報のみが必要なため、'yolov8n.pt'を使用します
    model = YOLO('yolov8n.pt')
    
    # 画像を読み込む
    img = cv2.imread(image_path)
    if img is None:
        print(f"エラー: 画像ファイル {image_path} を読み込めません。")
        return

    # 描画するために、元の画像のコピーを作成
    output_image = img.copy()

    # モデルで推論を実行
    results = model(img)
    result = results[0] # 画像１枚だけのため
    boxes = result.boxes # バウンディングボックスに関する情報のみ抽出

    # オブジェクトが検出された場合のみ描画処理を行う
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            # バウンディングボックスの座標を取得
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # クラスID、クラス名、信頼度を取得
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            conf = float(box.conf[0])
            
            # オブジェクトを囲む四角形を描画
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ラベル（クラス名と信頼度）を描画
            label = f'{class_name} {conf:.2f}'
            cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
    else:
        print("オブジェクトが検出されませんでした。")

    # 結果を保存
    cv2.imwrite(output_path, output_image)
    print(f"検出処理が完了し、結果を '{output_path}' に保存しました。")


if __name__ == '__main__': 
    draw_detection_boxes(INPUT_IMAGE, OUTPUT_PATH)