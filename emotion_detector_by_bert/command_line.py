# sentiment_analyzer.py ファイルから、SentimentAnalyzer というクラスをインポートします
from emotion_detector_by_bert import SentimentAnalyzer

def main() -> None:
    """
    コマンドラインインターフェース（CLI）のメイン処理
    Args:
        None
    Returns:
        None
    """
    try:
        # --- 分析器のインスタンスを作成 ---
        # ここでSentimentAnalyzerクラスが初期化され、モデルが読み込まれます
        analyzer = SentimentAnalyzer()

    except Exception as e:
        # モデルの読み込み中にエラーが発生した場合、ここで処理を中断します
        print(f"初期化中にエラーが発生しました。プログラムを終了します。: {e}")
        return # main関数を抜ける

    # --- 対話インターフェース部分 ---
    print("\n--- テキスト感情分析ツール (CLI版) ---")
    print("分析したい日本語の文章を入力してください。（終了する場合は 'q' と入力）")

    # ユーザーが 'q' を入力するまで無限にループします
    while True:
        # ユーザーからの入力を受け取ります
        user_input = input("> ")

        # 入力された文字列を小文字に変換し、'q'と一致するかチェックします
        if user_input.lower() == 'q':
            print("ご利用ありがとうございました。")
            break # whileループを抜けてプログラムを終了します

        # 何か文字が入力されている場合（空のままエンターが押されなかった場合）
        if user_input:
            # analyzerインスタンスが持つ analyze メソッドを呼び出して、感情分析を実行します
            result = analyzer.analyze(user_input)
            # 結果を画面に表示します
            print(f"分析結果: {result}\n")
        
        # 空のままエンターが押された場合
        else:
            print("文章が入力されていません。\n")


# このスクリプトが直接実行された場合にのみ、main()関数を呼び出します
# (他のファイルからインポートされた場合には実行されません)
if __name__ == '__main__':
    main()