# LDA: 教師なし学習で、大量の文書群を読み込んで、その中に隠れているトピックを自動的に発見するための統計モデル
import os
import re
import random
from openai import OpenAI
from glob import glob
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter
from janome.tokenfilter import POSKeepFilter, LowerCaseFilter
from gensim import corpora
from gensim.models import LdaModel
from typing import List, Tuple

# データ関連
DATA_DIR_PATH = os.path.join('data', 'text', '**', '*.txt')
NUM_SAMPLES = 2000  # 処理対象とする記事の数

# Gensim関連
# 辞書フィルタリング用:
# NO_BELOW: 5つの記事未満にしか出現しない低頻度語は無視
# NO_ABOVE: 全記事の50%以上にまたがって出現する高頻度語は無視
FILTER_NO_BELOW = 5
FILTER_NO_ABOVE = 0.5

# LDAモデル学習用:
# NUM_TOPICS: 作成するトピックの数 (livedoorニュースの9カテゴリに合わせる)
# PASSES: 学習の繰り返し回数（エポック数）
NUM_TOPICS = 9
PASSES = 15

# 再現性確保のための乱数シード
RANDOM_STATE = 42

# OpenAI API関連
OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"
TOPIC_LABEL_MAX_TOKENS = 20
TOPIC_LABEL_TEMPERATURE = 0.2


def extract_text(file_path: str) -> str:
    """
    livedoorニュースコーパスのファイルから本文テキストを抽出する
    ファイル形式は1行目がURL、2行目がタイムスタンプ、3行目以降が本文となっている
    この関数は3行目以降を読み込み、結合して一つの文字列として返す
    Args:
        file_path (str): 読み込むファイルのパス
    Returns:
        str: 抽出された本文テキスト
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[2:] # 3行目から最後までを取得
        text = "".join(lines)
        return text.strip() # 先頭と末尾の空白を削除して返す

def tokenize(text: str) -> List[str]:
    """
    Janomeを使い、テキストを単語のリストに分割する
    URLや数字の除去、Unicode正規化などの前処理を行った後、形態素解析を実行する
    品詞フィルタリング（名詞, 動詞, 形容詞のみ）と1文字の単語の除外を行い、意味のある単語だけを抽出する
    Args:
        text (str): 形態素解析を行う日本語テキスト
    Returns:
        List[str]: 抽出された単語のリスト
    """
    char_filters = [
        UnicodeNormalizeCharFilter(), # 全角文字を半角にしたり、特殊文字を通常の文字に変換したりする
        RegexReplaceCharFilter(r'https://?[-_a_zA-Z0-9./]+', ''), # URLを空文字に置換
        RegexReplaceCharFilter(r'[0-9]+', '0') # トピックを拾う上で正確な数値は不必要なため全て0に変換
    ]
    tokenizer = Tokenizer() # Janomeの形態素解析器(トークンに分割)
    token_filters = [
        POSKeepFilter(['名詞', '動詞', '形容詞']), # 残す単語の種類(助詞などは捨てる)
        LowerCaseFilter(), # 英字を全て小文字に変換(Appleとappleを同じ単語として処理したいため)
    ]
    analyzer = Analyzer(char_filters=char_filters, tokenizer=tokenizer, token_filters=token_filters) # analyzerの作成
    # for token in analyzer.analyze(text): analyzerを実行、tokenに代入
    # .base_form: 単語の基本形を取得(走った→走る)
    # if len(token.base_form) > 1: 1文字より長い単語のみ抽出
    tokens = [token.base_form for token in analyzer.analyze(text) if len(token.base_form) > 1]
    return tokens

def generate_topic_label(topic_words: List[str]) -> str:
    """
    OpenAI APIを呼び出し、トピックの単語リストからラベル名を生成する
    API呼び出しに失敗した場合は、エラーメッセージを表示し、固定の文字列を返す
    Args:
        topic_words (List[str]): トピックを表す単語のリスト
    Returns:
        str: 生成されたトピックラベル or エラーメッセージ
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("エラー: 環境変数 OPENAI_API_KEY が設定されていません。")
            return "（APIキー未設定）"

        client = OpenAI(api_key=api_key)
        word_list_str = ", ".join(topic_words)
        prompt = f"""
        以下の単語リストは、あるテーマを持つ文書群を代表する単語です。
        このリストから最も的確なトピック名を一つ、10文字以内の非常に簡潔な日本語で生成してください。

        単語リスト: [{word_list_str}]
        """

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "あなたは、与えられた単語群からその内容を要約し、最適なトピック名を考えるAIアシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=TOPIC_LABEL_TEMPERATURE,
            max_tokens=TOPIC_LABEL_MAX_TOKENS
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"OpenAI APIの呼び出し中にエラーが発生しました: {e}")
        return "（ラベル生成失敗）"


def load_and_sample_files(path_pattern: str, num_samples: int, seed: int) -> List[str]:
    """
    指定されたパスパターンからファイルリストを取得し、ランダムにサンプリングする
    Args:
        path_pattern (str): ファイルパスのパターン（glob形式）
        num_samples (int): サンプリングするファイル数
        seed (int): 乱数シード
    Returns:
        List[str]: サンプリングされたファイルパスのリスト
    """
    print("--- ファイルパスの取得とサンプリングを開始します ---")
    all_files = glob(path_pattern, recursive=True) # 指定したファイルパス(例: data/text/**/*.txt)に一致するものを全てリストに格納
    all_files = [path for path in all_files if 'README.txt' not in path] # README.txtは除外

    random.seed(seed) # 乱数の結果を固定するためのシード値を設定
    random.shuffle(all_files)
    sampled_files = all_files[:num_samples] # シャッフルされた記事から、2000件を抽出
    print(f"{len(all_files)}件中、{len(sampled_files)}件のファイルをサンプリングしました。\n")
    return sampled_files


def preprocess_documents(file_paths: List[str]) -> List[List[str]]:
    """
    ファイルのリストを受け取り、各ファイルの前処理（テキスト抽出、トークン化）を実行する
    Args:
        file_paths (List[str]): 前処理を行うファイルパスのリスト
    Returns:
        List[List[str]]: 各記事ごとのトークン化された単語リストのリスト
    """
    print("--- 全記事の前処理を開始します ---")
    all_documents = []
    for i, file_path in enumerate(file_paths):
        text = extract_text(file_path) #3行目以降の本文テキストを抽出
        tokens = tokenize(text) # 抽出したテキストをトークン化
        all_documents.append(tokens)
        if (i + 1) % 500 == 0:
            print(f"進捗: {i + 1} / {len(file_paths)} 件完了")
    print("--- 記事の前処理が完了しました ---\n")
    return all_documents


def create_dictionary_and_corpus(documents: List[List[str]]) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:
    """
    トークン化された文書リストから、Gensim用の辞書とコーパスを作成する
    Args:
        documents (List[List[str]]): 各記事ごとにトークン化された、計全部の単語リストのリスト
    Returns:
        Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]: 作成された辞書とコーパス
    """
    print("--- 辞書とコーパスの作成を開始します ---")
    # 辞書の作成(例: {0: '猫', 1: '犬', 2: '走る'})なお、重複は除かれる
    dictionary = corpora.Dictionary(documents)

    # 辞書のフィルタリング: 不要な単語を辞書から取り除く
    # NO_BELOW: 5つの記事未満にしか出現しない低頻度語は無視
    # NO_ABOVE: 全記事の50%以上にまたがって出現する高頻度語は無視
    dictionary.filter_extremes(no_below=FILTER_NO_BELOW, no_above=FILTER_NO_ABOVE)
    print(f"辞書に含まれる単語数（フィルタリング後）: {len(dictionary)} 語")

    # コーパスの作成: 各記事を、辞書に基づいてBoW(Bag-of-Words)ベクトルに変換する
    # 1. 記事1つ分の単語リストdocを受け取る
    # 2. docの中の各単語が、dictionaryのどのIDに対応するかを調べる
    # 3. 記事内で、各IDの単語が何回出現したかを数える
    # 4. 最終的に [(単語ID, 出現回数), (単語ID, 出現回数), ...] という形式のリストを返す
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    print("--- 辞書とコーパスの作成が完了しました ---\n")
    return dictionary, corpus


def train_lda_model(corpus: List[List[Tuple[int, int]]], dictionary: corpora.Dictionary) -> LdaModel:
    """
    コーパスと辞書を使ってLDAモデルを学習させる
    Args:
        corpus (List[List[Tuple[int, int]]]): BoW形式のコーパス
        dictionary (corpora.Dictionary): Gensim用の辞書
    Returns:
        LdaModel: 学習済みのLDAモデル
    """
    print(f"--- LDAモデルの学習を開始します (トピック数={NUM_TOPICS}) ---")
    print("データ量に応じて数分かかる場合があります...")

    # モデルの学習を実行
    # random_stateを固定すると、誰が実行しても同じ結果になり、再現性が得られる
    lda_model = LdaModel(
        corpus=corpus, # BoW形式のコーパス(学習の主要データ)
        id2word=dictionary, # 辞書(コーパスから導き出されたIDと単語の対応表)
        num_topics=NUM_TOPICS, # 作成するトピックの数
        passes=PASSES, # 学習の繰り返し回数（エポック数）
        random_state=RANDOM_STATE # 乱数シード(再現性確保のため)
    )
    print("--- LDAモデルの学習が完了しました！ ---\n")
    return lda_model


def display_topics_with_labels(lda_model: LdaModel) -> None:
    """
    学習済みLDAモデルの各トピックについて、構成単語とLLMによる推定ラベルを表示する
    Args:
        lda_model (LdaModel): 学習済みのLDAモデル
    Returns:
        None
    """
    print("--- 学習結果（各トピックの構成単語と推定ラベル）の確認 ---")
    # lda_model.print_topics() は (ID, 文字列) のリストを返す
    # 例: (0, '0.015*"発売" + 0.013*"搭載" + ...')
    for topic_id, topic_str in lda_model.print_topics(num_words=10):
        # トピックの構成単語リストを文字列から抽出する
        # 正規表現を使って""で囲まれた単語を全て抜き出す
        topic_words = re.findall(r'"(.*?)"', topic_str)

        # LLMを呼び出してトピック名を生成する
        print(f"トピック {topic_id} のラベルを生成中...")
        generated_label = generate_topic_label(topic_words)

        print(f"【トピック {topic_id}】 << 推定ラベル: {generated_label} >>")
        print(f"   構成単語: {', '.join(topic_words)}")
        print("-" * 50)


def classify_test_document(lda_model: LdaModel, corpus: List[List[Tuple[int, int]]], file_paths: List[str]) -> None:
    """
    コーパスの最初の記事が、学習済みモデルによってどのトピックに分類されるかを表示する
    Args:
        lda_model (LdaModel): 学習済みのLDAモデル
        corpus (List[List[Tuple[int, int]]]): BoW形式のコーパス
        file_paths (List[str]): コーパスに対応するファイルパスのリスト
    Returns:
        None
    """
    print(f"\n--- テスト記事のトピック分類結果 ---")
    # 最初の記事のBoW表現を取得
    test_doc_bow = corpus[0]
    test_doc_path = file_paths[0]

    # モデルに記事を入力し、トピック分類結果を取得
    topic_distribution = lda_model[test_doc_bow]

    # 結果を確率の高い順に並び替える
    sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)

    print(f"対象ファイル: {test_doc_path}")
    print("--------------------------------------------------")
    for topic_id, probability in sorted_topics:
        # f-stringのフォーマット: .3fは小数点以下3桁まで表示
        print(f"トピック {topic_id}: 割り当て確率 {probability:.3f}")
    print("--------------------------------------------------")
    most_likely_topic = sorted_topics[0][0]
    print(f"→ この記事は、最も確率の高い トピック {most_likely_topic} に分類されました。")


def main() -> None:
    """
    トピックモデルによる文書集合の自動分類システムのメイン処理
    Args:
        None
    Returns:
        None
    """
    # ファイルパスの取得とサンプリング
    sampled_files = load_and_sample_files(DATA_DIR_PATH, NUM_SAMPLES, RANDOM_STATE)

    # 全記事の前処理を実行
    documents = preprocess_documents(sampled_files)

    # 辞書とコーパスの作成
    dictionary, corpus = create_dictionary_and_corpus(documents)

    # LDAモデルの学習
    lda_model = train_lda_model(corpus, dictionary)

    # 学習結果の表示
    display_topics_with_labels(lda_model)
    
    # テスト記事の分類
    classify_test_document(lda_model, corpus, sampled_files)


if __name__ == '__main__':
    main()