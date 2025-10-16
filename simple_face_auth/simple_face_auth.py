import face_recognition
import numpy as np

KNOWN_FACE_PATH = "known_face.jpg"
UNKNOWN_FILES_PATHS = [
  "same_face.jpg",
  "different_face.jpg",
]
TOLERANCE = 0.5  # 顔認証の判定の厳しさ(小さいほど厳しい)

def load_first_face_encoding(image_path: str) -> np.ndarray | None:
  """
  指定された画像パスから、最初に検出された顔の特徴量を読み込む
  Args:
    image_path (str): 顔を読み込む画像のパス
  Returns:
    np.ndarray | None
      - 顔の特徴量(128次元ベクトル)
      - エラー時の返り値
  """
  try:
    # 写真のピクセルデータのリスト
    image = face_recognition.load_image_file(image_path)
    # 特徴量のリスト
    encodings = face_recognition.face_encodings(image)
    
    if not encodings:
      # 顔が検出できなかった場合
      print(f"警告: '{image_path}' から顔を検出できませんでした。")
      return None
      
     # 最初に見つかった顔の特徴量(写真に写る人物のうち１人目。複数人写っていない前提)を返す
    return encodings[0]

  except FileNotFoundError:
    print(f"エラー: '{image_path}' が見つかりませんでした。")
    return None


def are_faces_same(known_encoding: np.ndarray, unknown_encoding: np.ndarray, tolerance: float) -> bool:
  """
  2つの顔の特徴量を比較し、同一人物かどうかを判定する
  Args:
    known_encoding (np.ndarray): 基準となる顔の特徴量
    unknown_encoding (np.ndarray): 比較対象の顔の特徴量
    tolerance (float): 判定の閾値 (小さいほど厳しい)
  Returns:
    bool: 同一人物であればTrue、別人であればFalse
  """
  # compare_facesはリストを返すため、最初の要素 [0] をbool値として返す
  # 本来、既知の顔写真を複数設定できるため、リストとして返還される
  return face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=tolerance)[0]


def main() -> None:
  """
  顔認証プログラムのメイン処理
  Args:
    None
  Returns:
    None
  """
  print("顔認証を開始します...")
  
  # 基準となる顔を学習
  known_face_encoding = load_first_face_encoding(KNOWN_FACE_PATH)
  if known_face_encoding is None:
    print("基準顔の学習に失敗したため、プログラムを終了します。")
    return
  print("基準画像の学習... 完了")
  print(f"\n{len(UNKNOWN_FILES_PATHS)}件の画像をチェックします...")
  print("-" * 30)

  # 各テスト画像をループで処理
  for unknown_image_path in UNKNOWN_FILES_PATHS:
    print(f"'{unknown_image_path}' の認証を開始...")
    unknown_face_encoding = load_first_face_encoding(unknown_image_path)
    # テスト画像から顔が読み込めなければ、次の画像へ
    if unknown_face_encoding is None:
      print("-" * 30)
      continue
    
    # ２つの顔の特徴量を比較
    is_match = are_faces_same(known_face_encoding, unknown_face_encoding, tolerance=TOLERANCE)
    if is_match:
      print(f" -> 認証成功: '{unknown_image_path}' は同一人物の可能性が高いです。")
    else:
      print(f" -> 認証失敗: '{unknown_image_path}' は別人の可能性が高いです。")        
    print("-" * 30)

  print("全ての認証プロセスが完了しました。")

if __name__ == "__main__":
  main()