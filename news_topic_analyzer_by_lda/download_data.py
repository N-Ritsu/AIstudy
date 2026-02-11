import requests
import tarfile
import os

def download_and_extract_data() -> None:
    """
    livedoorニュースコーパスをダウンロードし、展開する関数
    Args:
        None
    Returns:
        None
    """
    # 保存先のディレクトリ名
    data_dir = "data"
    
    # データのURL
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    
    # 保存するファイル名
    file_name = "ldcc-20140209.tar.gz"

    # 保存先ディレクトリがなければ作成
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"ディレクトリ '{data_dir}' を作成しました。")

    # ファイルの保存パス
    save_path = os.path.join(data_dir, file_name)

    # --- ダウンロード処理 ---
    if not os.path.exists(save_path):
        print(f"'{file_name}' をダウンロードします...")
        try:
            response = requests.get(url, stream=True)
            # HTTPステータスコードが200（成功）以外の場合はエラーを送出
            response.raise_for_status()
            
            # バイナリとしてファイルを書き込み
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("ダウンロードが完了しました。")
        except requests.exceptions.RequestException as e:
            print(f"ダウンロード中にエラーが発生しました: {e}")
            return
    else:
        print(f"'{file_name}' は既に存在します。ダウンロードをスキップします。")


    # --- 展開処理 ---
    extract_path = os.path.join(data_dir, "text")
    if not os.path.exists(extract_path):
        print(f"'{file_name}' を展開します...")
        try:
            with tarfile.open(save_path, 'r') as tar:
                # .extractall(): すべてのファイルを展開
                tar.extractall(path=data_dir)
            print(f"展開が完了しました。'{extract_path}' にデータが保存されています。")
        except tarfile.TarError as e:
            print(f"展開中にエラーが発生しました: {e}")
    else:
        print(f"'{extract_path}' は既に存在します。展開をスキップします。")

if __name__ == '__main__':
    download_and_extract_data()