from ultralytics import YOLO
import cv2
from typing import Tuple

INPUT_VIDEO_PATH = 'video.mp4'
OUTPUT_VIDEO_PATH = 'output.mp4'
CONF_THRESHOLD=0.55

def setup_video_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
    """
    入力動画のプロパティに基づいてVideoWriterオブジェクトを設定
    Args:
        cap (cv2.VideoCapture): 読み込み元のVideoCaptureオブジェクト
        output_path (str): 出力動画ファイルのパス
    Returns:
        cv2.VideoWriter: 設定済みのVideoWriterオブジェクト
    """
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def track_objects_in_video(input_video_path: str, output_video_path: str, model_name: str = 'yolov8n.pt', conf_threshold: float = 0.55) -> None:
    """
    指定された動画をフレームごとに読み込み、YOLOv8の追跡機能を使って各オブジェクトのバウンディングボックス、追跡ID、クラス名、信頼度を描画
    指定された信頼度閾値未満のオブジェクトは描画されない
    Args:
        input_video_path (str): 入力動画ファイルのパス
        output_video_path (str): 処理結果を保存する動画ファイルのパス
        model_name (str, optional): 使用するYOLOv8モデルファイル名。デフォルトは 'yolov8n.pt'。
        conf_threshold (float, optional): 描画対象とするオブジェクトの信頼度の閾値。デフォルトは 0.55。
    Returns:
        None: この関数は値を返さず、結果を指定されたパスに動画ファイルとして保存
    """
    # 1. モデルの読み込み
    model = YOLO(model_name)

    # 2. 動画の読み込み設定
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイル '{input_video_path}' を開けません。")
        return

    # 3. 出力用動画の設定
    writer = setup_video_writer(cap, output_video_path)

    print(f"動画の追跡処理を開始します... (信頼度{conf_threshold}未満は除外)")

    try:
        # 4. フレームごとにループ処理
        while cap.isOpened(): # フレームが続く限りループ
            ret, frame = cap.read() # フレームを１枚だけ読み込む
            if not ret:
                break
            
            # 5. 各フレームで物体追跡を実行
            results = model.track(source=frame, persist=True)
            result = results[0]
            
            # 6. 追跡結果を描画
            if result.boxes.id is not None:
                # 必要な情報を一括で取得
                # .cpu().numpy(): CPUで扱えるNumPy配列に変換
                # .astype(int): 座標やIDを整数に変換
                boxes = result.boxes.xyxy.cpu().numpy().astype(int) # 座標
                track_ids = result.boxes.id.cpu().numpy().astype(int) # 追跡ID
                confs = result.boxes.conf.cpu().numpy() # 信頼度スコア
                class_ids = result.boxes.cls.cpu().numpy().astype(int) # クラスID
                
                # 追跡された各オブジェクトに対してループ
                # zip(...): 検出された物体ひとつ分の座標・追跡ID・信頼度・クラスIDをセットにする
                for box, track_id, conf, class_id in zip(boxes, track_ids, confs, class_ids):
                    # 信頼度が閾値未満の場合はスキップ
                    if conf < conf_threshold:
                        continue

                    # 座標、クラス名、ラベルテキストを準備
                    x1, y1, x2, y2 = box
                    class_name = result.names[class_id]
                    label = f'ID:{track_id} {class_name} {conf:.2f}'
                    
                    # バウンディングボックスとラベルを描画
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 7. 描画したフレームを出力用動画に書き込む
            writer.write(frame)
    
    finally:
        # 8. 後処理（リソースの解放）
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"処理が完了しました。結果を '{output_video_path}' に保存しました。")


if __name__ == '__main__':
    # 処理の実行
    track_objects_in_video(
        input_video_path=INPUT_VIDEO_PATH,
        output_video_path=OUTPUT_VIDEO_PATH,
        conf_threshold=CONF_THRESHOLD
    )