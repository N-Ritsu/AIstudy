# torch,npは直接は参照されないが、計算エンジン・NumPy配列の処理に必要
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import Tuple

KNOWLEDGE_PASS = "knowledge.txt"
QUESTIONS = [
  "日本の首都はどこですか？",
  "日本にはいくつの都道府県がありますか？",
  "今日の天気はなんですか？"
]
# 回答の参考にする知識の数
NUMBER_OF_KNOWLEDGE = 3
# 文章をベクトルに変換するためのAIモデルをロード
# "intfloat/multilingual-e5-large" は、多言語に対応した高性能なモデル
VECTORIZER = SentenceTransformer("intfloat/multilingual-e5-large")
# LLMの準備
# Mistralの比較的小さなモデルをロード
LLM = pipeline("text-generation", model="google/gemma-2b-it") # Gemmaの方が軽いので推奨

def load_knowledge(knowledge_pass: str) -> list[str]:
  """ 
  LLMの知識となるテキストファイルを読み込む
  Args:
    knowledge_pass (str): 読み込むテキストファイルのパス
  Returns: 
    list[str]: ファイルの各段落の内容を格納した文字列のリスト
  """
  with open(knowledge_pass, "r", encoding = "utf-8") as f:
    # テキストを段落ごとに分割する
    # f.read(): ファイルの中身を読み取り、一つの巨大な文字列に変換("段落Aです。\n\n段落Bです。\n\n\n段落Cです。")
    # .split("\n\n"): 2回の改行(空行)を見つけ次第、そこで文字列を分割してリストへ格納(["段落Aです。", "段落Bです。", ", "段落Cです。"])
    # for p in ... if p.strip: リストのそれぞれをpとし、その両端の空白や改行を削除。なお、空白や改行のみで構成されていた場合、if Falseとなり消える。
    # p.strip() ...: 最後の整形
    knowledge_texts = [p.strip() for p in f.read().split("\n\n") if p.strip()]
    return knowledge_texts

def make_vector_database(knowledge_texts: list[str], vectorizer: SentenceTransformer) -> faiss.Index:
  """
  テキストのリストを受け取り、ベクトルに変換する
  Args: 
    knowledge_texts (list[str]): ベクトル化するテキストのリスト
    vectorizer (SentenceTransformer): テキストをベクトルに変換するための、初期化済みのSentenceTransformerモデル。
  Returns:
    faiss.Index: テキストベクトルが追加された、FAISSのインデックスオブジェクト
  """
  knowledge_vectors = vectorizer.encode(knowledge_texts)

  # 索引（ベクトルデータベース）の作成
  # faiss: 翻訳されたベクトルの束をいつでも一瞬で検索できる索引を作成
  # faiss.IndexFlatL2: 最も基本的なタイプの索引を作成
  # knowledge_vectors.shape[1]は、ベクトルの次元数(例: 1024)
  index = faiss.IndexFlatL2(knowledge_vectors.shape[1])
  # 作成した索引に、知識ベクトルを追加
  index.add(knowledge_vectors)
  return index

def search_for_knowledge(question: str, knowledge_texts: list[str], vectorizer: SentenceTransformer, index: faiss.Index, number_of_knowledge: int) -> list[str]:
	"""
  受け取った質問に最も関連性の高い知識を、ベクトル検索によって特定する
  Args:
    question (str): 質問の文字列
    knowledge_texts (list[str]): 全知識テキストのリスト
    vectorizer (SentenceTransformer): 文章をベクトルに変換するためのAIモデル
    index (faiss.Index): 検索対象のFAISSインデックス
    number_of_knowledge (int): 回答の参考にする知識の数
  Returns:
    list[str]: 質問との関連性が高いと判断された、上位number_of_knowledge個のテキストのリスト
  """

	# 質問と関連性の高い知識を検索
	# 質問文もベクトルに変換
	question_vector = vectorizer.encode([question])
	# faissインデックスで、最も意味が近い知識を検索
	# 最も意味が近い知識の書かれた行の要素番号を3つ返す(本来複数質問にも対応できるよう、２重配列となっていることに注意)
	distances, indices = index.search(question_vector, number_of_knowledge)

	# 検索結果のテキストを取得
	# indices[0]: ２重配列のため、普通のリストとして扱いたいから[0]と指定
	# [knowledge_texts[i] ... ]: indicesに保存された３つの要素を、knowledge_textsのリストから抜粋、それだけのリストを作成
	retrieved_texts = [knowledge_texts[i] for i in indices[0]]
	return retrieved_texts

def generate_answer_by_llm(prompt: str) -> str:
  """
  プロンプトを元に、LLMで回答を生成する
  Args:
    prompt (str): LLMに渡すプロンプト文字列
  Returns:
    str: LLMによって生成され、整形された最終的な回答文字列
  """
	# max_new_tokens=150: 生成する文章の文字数の制限
	# num_return_sequences=1: 回答の候補を１つだけ生成する
  response = LLM(prompt, max_new_tokens = 150, num_return_sequences = 1)
	# response: [{"generated_text": "...【質問】日本の首都はどこですか？\n【回答】\n日本の首都は東京です"}]
	# response[0]: {"generated_text": "...【質問】日本の首都はどこですか？\n【回答】\n日本の首都は東京です"}
	# response[0]["generated_text"]: "...【質問】日本の首都はどこですか？\n【回答】\n日本の首都は東京です"
	# .split("【回答】"): 【回答】前後で分割してリストにする(["...【質問】日本の首都はどこですか？\n", "\n日本の首都は東京です"])
	# [-1]: 最後の要素のみを取り出す("\n日本の首都は東京です")
	# .strip: 整形("日本の首都は東京です")
  answer = response[0]["generated_text"].split("【回答】")[-1].strip()
  return answer

def main() -> None:
  """
  RAGチャットボットのメイン処理
  Args:
    None
  Returns:
    None
  """
  knowledge_texts = load_knowledge(KNOWLEDGE_PASS)
  print(f"{len(knowledge_texts)}個の知識（段落）を読み込みました。")

  print("知識をベクトルに変換中...")
  index = make_vector_database(knowledge_texts, VECTORIZER)
  print("ベクトルデータベースの準備が完了しました。")

  for question in QUESTIONS:
    print(f"\n質問: {question}")
    # 実行
    retrieved_texts = search_for_knowledge(question, knowledge_texts, VECTORIZER, index, NUMBER_OF_KNOWLEDGE)
    if retrieved_texts:
      print("\n--- 関連知識を検索しました ---")
      for text in retrieved_texts:
        print(f"- {text[:100]}...") # 最初の100文字だけ表示
      # LLMに渡すためのプロンプトを作成
      # "\n- ".join(retrieved_texts): retrieved_textsを区切り文字\n- で区切って連結
      context_string = "\n- ".join(retrieved_texts)
      prompt = f"""以下の「コンテキスト情報」だけを元にして、ユーザーからの「質問」に、簡潔に日本語で回答してください。
      コンテキスト情報に答えがない場合は、「分かりません」と回答してください。
      【コンテキスト情報】
      - {context_string}
      【質問】
      {question}
      【回答】
      """
      answer = generate_answer_by_llm(prompt)
      print("\n--- AIによる最終回答 ---")
      print(answer)
    else:
      print("知識を検索できませんでした")

if __name__ == "__main__":
  main()