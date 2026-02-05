import os
import argparse
import logging
import tempfile
from typing import List
import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

# ロギング設定
# 処理の状況を分かりやすく表示するための設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_timestamp(seconds: float) -> str:
    """
    秒数をSRT形式のタイムスタンプ（HH:MM:SS,ms）に変換するヘルパー関数
    Args:
        seconds (float): 変換したい秒数
    Returns:
        str: SRT形式のタイムスタンプ文字列 (例: "00:01:23,456")
    """
    # 整数部と小数部に分ける
    integer_part = int(seconds)
    millisecond_part = int((seconds - integer_part) * 1000)
    
    # 時、分、秒に変換
    hours = integer_part // 3600
    minutes = (integer_part % 3600) // 60
    secs = integer_part % 60
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecond_part:03d}"

def transcribe_with_whisper(args: argparse.Namespace) -> None:
    """
    コマンドライン引数に基づき、音声ファイルの文字起こしを実行する関数
    音声ファイルを読み込み、無音区間で分割後、Whisperモデルを使用してチャンクごとに文字起こしを行う
    結果はテキストファイル、SRTファイル、または標準出力に出力
    Args:
        args (argparse.Namespace): コマンドラインから受け取った引数が格納されたオブジェクト
    Returns:
        None
    """
    logging.info(f"Whisperモデル '{args.model}' をロード中...")
    # 指定されたモデルサイズでWhisperモデルをロード
    # CPU環境のため、fp16はFalseに設定
    model = whisper.load_model(args.model) # プログレスバーも表示される
    logging.info("モデルのロードが完了しました")

    logging.info(f"音声ファイル '{args.input_file}' を読み込んでいます...")
    # pydubを使用して音声ファイルを読み込む
    audio: AudioSegment = AudioSegment.from_file(args.input_file)
    logging.info("音声ファイルの読み込みが完了しました")

    # --- 長尺ファイルの分割処理 ---
    # 無音区間を検出して音声を分割するためのパラメータ
    # silence_thresh: このdB未満を無音とみなす
    # min_silence_len: このミリ秒以上の無音区間を探す
    # keep_silence: 分割後のチャンクの最初と最後に残す無音の長さ
    logging.info("無音区間を検出し、音声を分割しています...")
    chunks: List[AudioSegment] = split_on_silence(
        audio,
        min_silence_len=1000,
        silence_thresh=audio.dBFS - 14,
        keep_silence=500
    )
    
    if not chunks:
        # 分割されなかった場合は、ファイル全体を一つのチャンクとして扱う
        logging.warning("無音区間が検出されませんでした。ファイル全体を処理します。")
        chunks = [audio]
    else:
        logging.info(f"音声を {len(chunks)} 個のチャンクに分割しました。")

    # 一時ディレクトリを作成し、分割した音声ファイルを保存
    with tempfile.TemporaryDirectory() as temp_dir:
        transcribed_texts: List[str] = []
        full_srt_content: str = ""
        segment_counter: int = 1
        
        logging.info("各チャンクの文字起こしを開始します...")
        # tqdmを使用してプログレスバーを表示
        for i, chunk in enumerate(tqdm(chunks, desc="文字起こし進捗")):
            # 一時ファイルとしてWAV形式で保存
            chunk_path: str = os.path.join(temp_dir, f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            
            # Whisperで文字起こしを実行
            # verbose=Trueにするとタイムスタンプ付きの結果が得られる
            result: dict = model.transcribe(chunk_path, language=args.language, fp16=False, verbose=True)
            
            # 通常のテキスト結果を保存
            transcribed_texts.append(result["text"])
            
            # SRT形式のコンテンツを生成
            # 例：
            # 1
            # 00:00:01,234 --> 00:00:03,456
            # こんにちは、皆さん
            if args.srt:
                for segment in result["segments"]:
                    full_srt_content += f"{segment_counter}\n"
                    full_srt_content += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
                    full_srt_content += f"{segment['text'].strip()}\n\n"
                    segment_counter += 1

        logging.info("すべてのチャンクの文字起こしが完了しました。")
        
        # 結果の出力
        final_text: str = "".join(transcribed_texts)

        if args.output_file:
            # SRT形式での保存が指定されている場合
            if args.srt:
                output_path = args.output_file if args.output_file.endswith(".srt") else args.output_file + ".srt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(full_srt_content)
                logging.info(f"SRT形式の文字起こし結果を '{output_path}' に保存しました。")
            # 通常のテキスト形式で保存
            else:
                output_path = args.output_file if args.output_file.endswith(".txt") else args.output_file + ".txt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(final_text)
                logging.info(f"文字起こし結果を '{output_path}' に保存しました。")
        else:
            # 出力ファイルが指定されていない場合はコンソールに表示
            print("\n--- 文字起こし結果 ---")
            print(final_text)


def main() -> None:
    """
    音声ファイルからの文字起こしプログラムのメイン処理関数
    Args:
      None
    Returns:
      None
    """
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(
        description="Whisperを使用して音声ファイルを高精度に文字起こしするツール。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "input_file", 
        type=str, 
        help="文字起こし対象の音声ファイルパス (例: audio.mp3)"
    )
    parser.add_argument(
        "-o", "--output_file", 
        type=str, 
        default=None, 
        help="結果を保存するファイルパス。\n指定しない場合はコンソールに結果を出力します。"
    )
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        default="base", 
        choices=["tiny", "base", "small", "medium", "large"],
        help="使用するWhisperモデルのサイズ。\nCPU環境では 'tiny' または 'base' を推奨します。(デフォルト: base)"
    )
    parser.add_argument(
        "-l", "--language", 
        type=str, 
        default="ja", 
        help="音声ファイルの言語を指定します。(デフォルト: ja)"
    )
    parser.add_argument(
        "--srt",
        action="store_true",
        help="このフラグを立てると、出力がタイムスタンプ付きのSRT形式になります。\n--output_file の指定が必須です。"
    )

    args: argparse.Namespace = parser.parse_args()
    
    if args.srt and not args.output_file:
        parser.error("--srt オプションを使用するには、--output_file で出力先を指定する必要があります。")

    transcribe_with_whisper(args)

if __name__ == "__main__":
    main()