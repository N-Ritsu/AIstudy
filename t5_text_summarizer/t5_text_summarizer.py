import argparse
from collections import Counter
from dataclasses import dataclass
from typing import List

from janome.tokenizer import Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# --- 設定値の集約 ---
@dataclass(frozen=True)
class SummarizationConfig:
    """
    要約処理に関する設定値を保持するデータクラス
    """
    # 使用する事前学習済みモデルの名前
    MODEL_NAME: str = "tsmatz/mt5_summarize_japanese"
    # モデルが一度に処理できる最大のトークン数
    MAX_INPUT_LENGTH: int = 350
    
    # 【第一段階：部分要約】用の設定
    STAGE1_MAX_RATIO: float = 0.5 # 入力に対する最大要約長比率
    STAGE1_MIN_RATIO: float = 0.4 # 入力に対する最小要約長比率
    
    # 【第二段階：全体要約】用の設定
    STAGE2_MAX_RATIO: float = 0.5 # 入力に対する最大要約長比率
    STAGE2_MIN_RATIO: float = 0.4 # 入力に対する最小要約長比率
    
    # 短い単体記事用の設定
    SHORT_TEXT_MAX_RATIO: float = 0.3 # 入力に対する最大要約長比率
    SHORT_TEXT_MIN_RATIO: float = 0.15 # 入力に対する最小要約長比率
    
    MAX_SUMMARY_CEILING: int = 150 # 最終的に生成される要約の絶対的な最大トークン長
    MIN_SUMMARY_FLOOR: int = 80 # 最終的に生成される要約の絶対的な最小トークン長


class TextSummarizer:
    """
    テキストの要約処理を行うクラス
    モデルとトークナイザを内部に保持し、テキストの長さに応じて適切な要約戦略を実行する
    """
    def __init__(self, config: SummarizationConfig) -> None:
        """
        TextSummarizerのインスタンスを初期化
        Args:
            config (SummarizationConfig): 要約処理に関する設定オブジェクト
        Returns:
            None
        """
        self.config = config
        print("モデルとトークナイザを読み込んでいます...")
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)
        print("読み込みが完了しました。")

    def summarize(self, text: str) -> str:
        """
        入力されたテキストを要約する
        テキストのトークン数に応じて、短いテキスト用の要約処理か、長いテキスト用の二段階要約処理かを自動で判断して実行
        Args:
            text (str): 要約対象のテキスト
        Returns:
            str: 生成された要約文
        """
        source_token_length = len(self.tokenizer.encode(text))
        print(f"元のテキストのトークン数: {source_token_length}")

        if source_token_length > self.config.MAX_INPUT_LENGTH:
            return self._summarize_long_text(text)
        else:
            return self._summarize_short_text(text, max_ratio=self.config.SHORT_TEXT_MAX_RATIO, min_ratio=self.config.SHORT_TEXT_MIN_RATIO)

    def _summarize_short_text(self, text: str, max_ratio: float, min_ratio: float) -> str:
        """
        単一のテキストチャンクを要約する内部メソッド
        指定された最大・最小比率に基づいて動的に要約長を計算し、要約を生成
        Args:
            text (str): 要約対象のテキスト（MAX_INPUT_LENGTH以下であること）
            max_ratio (float): 入力トークン長に対する最大要約長の比率
            min_ratio (float): 入力トークン長に対する最小要約長の比率
        Returns:
            str: 生成された要約文
        """
        # トークン化
        # truncation=True: max_lengthを超える部分は切り捨てる
        # return_tensors="pt": PyTorchのテンソル形式で返す
        inputs = self.tokenizer(text, max_length=self.config.MAX_INPUT_LENGTH, truncation=True, return_tensors="pt")
        input_token_length = len(inputs["input_ids"][0])
        
        # 引数で渡されたmax/min比率を使って、動的に長さを計算
        dynamic_max_length = int(input_token_length * max_ratio)
        dynamic_min_length = int(input_token_length * min_ratio)
        
        # 安全装置：上限(CEILING)と下限(FLOOR)の範囲内に収める
        final_max_length = min(self.config.MAX_SUMMARY_CEILING, max(self.config.MIN_SUMMARY_FLOOR, dynamic_max_length))
        final_min_length = min(final_max_length, max(self.config.MIN_SUMMARY_FLOOR, dynamic_min_length))

        print(f"入力長: {input_token_length}トークン -> 要約長 (min/max): {final_min_length}/{final_max_length} トークンに設定します。")
        
        # モデルを使って要約を生成
        summary_ids = self.model.generate(
            **inputs, # トークン化された入力を展開
            max_length=final_max_length, 
            min_length=final_min_length,
            num_beams=5, # ビームサーチ。次の単語の候補を複数保持して最適な要約を生成するための設定。
            repetition_penalty=1.5, # 同じ単語の繰り返しを防ぐ
            early_stopping=True # 生成が完了したら、max_lengthに達していなくても終了
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def _summarize_long_text(self, text: str) -> str:
        """
        長いテキストを二段階アプローチで要約する内部メソッド
        1. テキストをモデルが処理できる長さのチャンクに分割
        2. 各チャンクを個別に要約（第一段階）
        3. 全てのチャンクの要約文を結合し、再度要約して最終的な要約文を生成（第二段階）
        Args:
            text (str): 要約対象の長いテキスト
        Returns:
            str: 最終的に生成された要約文
        """
        print("入力テキストが長いため、二段階の分割要約処理を開始します...")
        
        # --- チャンク分割ロジック ---
        sentences = [s.strip() for s in text.replace('\n', '').split('。') if s.strip()]
        chunks: List[str] = []
        current_chunk_sentences: List[str] = []
        current_chunk_length = 0
        # チャンクの最大長を少し短めに設定して、特殊トークン分の余裕を持たせる
        CHUNK_MAX_LENGTH = self.config.MAX_INPUT_LENGTH - 32  

        for sentence in sentences:
            sentence_with_period = sentence + "。" # splitで失われた句点を再び文末に追加
            sentence_length = len(self.tokenizer.encode(sentence_with_period)) # その文のトークン長を計算
            if current_chunk_length + sentence_length > CHUNK_MAX_LENGTH:
                if current_chunk_sentences:
                    chunks.append("。".join(current_chunk_sentences) + "。") # 各文を句点で連結して１つのチャンクにする
                # 現在処理を行っている文を、新しいチャンク作成の最初の文とする
                current_chunk_sentences = [sentence]
                current_chunk_length = sentence_length
            else:
                current_chunk_sentences.append(sentence) # 現在のチャンクに文を追加
                current_chunk_length += sentence_length # 現在のチャンクのトークン長を更新
        if current_chunk_sentences:
            # ループが終了した時点でcurrent_chunk_sentencesに残っている最後の作成中チャンクを追加
            chunks.append("。".join(current_chunk_sentences) + "。")
        print(f"{len(chunks)}個のチャンクに分割しました。")

        # --- 第一段階：部分要約 ---
        print("\n--- 第一段階：部分要約 ---")
        partial_summaries = []
        for i, chunk in enumerate(chunks):
            # 第一段階用の設定値を渡す
            partial_summary = self._summarize_short_text(
                chunk,
                max_ratio=self.config.STAGE1_MAX_RATIO,
                min_ratio=self.config.STAGE1_MIN_RATIO
            )
            partial_summaries.append(partial_summary)

        # --- 第二段階：全体要約 ---
        print("\n--- 第二段階：全体要約 ---")
        combined_summary_text = " ".join(partial_summaries)
        
        if len(self.tokenizer.encode(combined_summary_text)) > self.config.MAX_INPUT_LENGTH:
            print("警告：部分要約の結合結果が長すぎるため、切り捨てが発生する可能性があります。")

        # 第二段階用の設定値を渡す
        final_summary = self._summarize_short_text(
            combined_summary_text,
            max_ratio=self.config.STAGE2_MAX_RATIO,
            min_ratio=self.config.STAGE2_MIN_RATIO
        )
        return final_summary


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Janomeを使い、テキストからキーワード（名詞）を抽出して頻度順に返す
    Args:
        text (str): キーワード抽出対象のテキスト
        top_n (int, optional): 抽出するキーワードの最大数。デフォルトは5
    Returns:
        List[str]: 抽出されたキーワードのリスト
    """
    # Janomeの形態素解析器を初期化
    janome_tokenizer = Tokenizer()
    
    # 品詞が「名詞」であり、かつ「一般名詞」または「固有名詞」である単語のみを抽出
    nouns = [
        token.surface for token in janome_tokenizer.tokenize(text) 
        if token.part_of_speech.startswith(('名詞,一般', '名詞,固有名詞'))
    ]
    
    # 単語の出現頻度をカウント
    word_counts = Counter(nouns)
    
    # 頻度が高い順に上位n個のキーワード（単語）を返す
    return [word for word, count in word_counts.most_common(top_n)]


def main() -> None:
    """
    コマンドライン引数から入力ファイルを受け取り、要約とキーワード抽出を実行するメイン処理
    Args:
        None
    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="テキストファイルを読み込んで、内容を要約するCLIツール")
    parser.add_argument("input_file", type=str, help="要約したいテキストファイルのパス")
    args = parser.parse_args()

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            source_text = f.read()
    except FileNotFoundError:
        print(f"エラー: ファイル '{args.input_file}' が見つかりません。")
        return
    except Exception as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {e}")
        return
    
    # 設定をインスタンス化
    config = SummarizationConfig()
    
    # 要約器をインスタンス化（ここでモデル等が読み込まれる）
    summarizer = TextSummarizer(config)
    
    # 要約処理の実行
    summary_result = summarizer.summarize(source_text)

    # 要約文からキーワードを抽出
    keywords = extract_keywords(summary_result)

    print("\n" + "="*40)
    print("--- 最終的に生成された要約文 ---")
    print(summary_result)
    print("\n--- 抽出されたキーワード ---")
    print(", ".join(keywords))
    print("="*40)


if __name__ == "__main__":
    main()