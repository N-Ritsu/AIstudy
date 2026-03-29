import requests
import zipfile
from pathlib import Path

# MovieLens 100kデータセットのダウンロードURL
DOWNLOAD_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
# データを保存する親ディレクトリ
DATA_DIR = Path("./ml-100k-data")
# ダウンロードしたZIPファイルの一時的な保存パス
ZIP_PATH = DATA_DIR / "ml-100k.zip"
# 展開後のデータが格納されるディレクトリパス
EXTRACTED_PATH = DATA_DIR / "ml-100k"
# 処理の完了（データセットの存在）を確認するための代表的なファイルパス
# このファイルが存在すれば、ダウンロードと展開は完了していると判断します。
FINAL_DATA_FILE = EXTRACTED_PATH / "u.data"


def download_movielens_100k() -> None:
    """
    MovieLens 100kデータセットをダウンロードし、指定されたディレクトリに展開する。
    - 最終的なデータファイル（u.data）が存在するかどうかを確認します。
       - 存在する場合：メッセージを表示して処理をスキップします。
       - 存在しない場合：次の処理に進みます。
    - データを保存するためのディレクトリを作成します。
    - 指定されたURLからデータセットのZIPファイルをダウンロードします。
    - ダウンロードしたZIPファイルを展開します。
    - 展開後、不要になったZIPファイルを削除します。
    Args:
        None
    Returns:
        None
    """

    # データが既に存在すれば処理をスキップ
    if FINAL_DATA_FILE.exists():
        print(f"データは既に '{FINAL_DATA_FILE}' に存在します。ダウンロードをスキップします。")
        return

    # --- ディレクトリの作成 ---
    # exist_ok=True に設定することで、ディレクトリが既に存在している場合にエラーが発生するのを防ぐ。
    DATA_DIR.mkdir(exist_ok=True)

    # --- ダウンロード ---
    print(f"'{DOWNLOAD_URL}' からデータセットをダウンロードしています...")
    try:
        # stream=Trueにすることで、レスポンスのコンテンツをすぐにメモリに読み込むのではなく、ストリーミングするようにする。
        # これにより、大きなファイルをダウンロードする際のメモリ使用量を抑えられる。
        response = requests.get(DOWNLOAD_URL, stream=True)

        # HTTPステータスコードがエラー（4xxや5xx）を示している場合に例外を発生させる。
        response.raise_for_status()

        # ダウンロードしたデータをファイルに書き込む。
        # 'wb': バイナリ書き込みモード。ZIPファイルはバイナリファイルのため、このモードになる。
        with open(ZIP_PATH, 'wb') as f:
            # iter_contentを使用して、レスポンスデータを指定したサイズのチャンクに分割(メモリ効率のため)。
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("ダウンロードが完了しました。")
    except requests.exceptions.RequestException as e:
        print(f"ダウンロード中にエラーが発生しました: {e}")
        return

    # --- 展開 ---
    print(f"'{ZIP_PATH}' を展開しています...")
    try:
        # zipfile.ZipFileを使用してZIPファイルを扱う。
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # ZIPファイル内のすべてのファイルを指定したディレクトリに展開。
            zip_ref.extractall(DATA_DIR)
        print(f"展開が完了しました。データは '{EXTRACTED_PATH}' にあります。")
    except zipfile.BadZipFile as e: # ダウンロードしたファイルが壊れているなど、ZIPファイルとして不正な場合に発生するエラー。
        print(f"ZIPファイルの展開中にエラーが発生しました: {e}")
    finally:
        # finallyブロック内のコードは、tryブロックでエラーが発生したかどうかに関わらず、必ず実行される。
        # ダウンロードしたzipファイルを削除
        if ZIP_PATH.exists():
            ZIP_PATH.unlink() # pathlib.Path.unlink(): ファイルを削除

if __name__ == '__main__':
    download_movielens_100k()