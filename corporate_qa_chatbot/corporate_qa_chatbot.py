from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from transformers import (
  AutoTokenizer, 
  AutoModelForCausalLM, 
  BitsAndBytesConfig, 
  StoppingCriteria, 
  StoppingCriteriaList,
)
import gradio as gr
from typing import List, Dict, Tuple, Set, Union


# モデル関連
VECTOR_MODEL_NAME = "all-MiniLM-L6-v2" #ベクトル化モデル
LLM_MODEL_NAME = "llm-jp/llm-jp-3-440m"
LLM_MAX_INPUT_LENGTH = 512  # llm-jp/llm-jp-3-440mのコンテキスト長
# RAG関連
NUM_RELEVANT_CHUNKS = 3  # 検索するチャンク数
MAX_NEW_TOKENS = 64      # LLMが生成する最大トークン数
# Gradio関連
GRADIO_TITLE = "社内文書QAチャットボット"
GRADIO_DESCRIPTION = "documentsフォルダ内のPDFに関する質問をしてください。"
GRADIO_MAX_FLAG_TIME = 300 # Gradioのタイムアウトを5分に設定
# 停止ワード
STOP_KEYWORDS = ["質問", "\n\n"] # モデルが回答の後に余分なものを生成しやすいので、停止ワードを調整


print("--- モデルのロードを開始します ---")
# ベクトル化モデルのロード
try:
  VECTOR_MODEL = SentenceTransformer(VECTOR_MODEL_NAME)
  print(f"ベクトル化モデル '{VECTOR_MODEL_NAME}' のロード完了。")
except Exception as e:
  print(f"エラー: ベクトル化モデル '{VECTOR_MODEL_NAME}' のロードに失敗しました: {e}")
  exit(1)

# LLMのロード
try:
  print(f"大規模言語モデル '{LLM_MODEL_NAME}' のロードを開始します...")
  quantization_config = BitsAndBytesConfig(load_in_4bit=True) # 4bit量子化設定
  
  # テキストをトークンにする、トークナイザーをロード
  LLM_TOKENIZER = AutoTokenizer.from_pretrained(
    LLM_MODEL_NAME,
    trust_remote_code=True # このモデルではTrue推奨だが、危険性もあるため注意
  )
  LLM_MODEL = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    quantization_config=quantization_config,
    device_map="cpu", # モデルをCPUに明示的にロード
    trust_remote_code=True
  )
  print(f"大規模言語モデル '{LLM_MODEL_NAME}' のロード完了。")
except Exception as e:
  print(f"エラー: 大規模言語モデル '{LLM_MODEL_NAME}' のロード中に致命的なエラーが発生しました: {e}")
  print("ヒント: メモリ不足の可能性があります。PCを再起動するか、不要なアプリケーションを終了してください。")
  exit(1)


# 停止条件に必要なデータをselfとして保持したいためクラス化
class StopOnKeywords(StoppingCriteria):
  """
  指定されたキーワードが生成されたテキストに含まれた場合、生成を停止するカスタムStoppingCriteria
  """
  def __init__(self, keywords: List[str], tokenizer, input_length: int) -> None:
    """
    StopOnKeywordsのインスタンスの初期化
    Args:
      keywords (list[str]): 生成テキスト内で検出された場合に停止する指定文字列のリスト
      tokenizer: 生成テキストをデコードするためのHugging Faceトークナイザーオブジェクト
      input_length (int): LLMへの入力プロンプトのトークン長
    """
    self.keywords = keywords
    self.tokenizer = tokenizer
    self.input_length = input_length

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    """
    生成された部分のテキストをデコードし、指定したキーワードのいずれかが含まれていた場合に出力を停止する
    Args:
      input_ids (torch.LongTensor): 現在までに生成されたすべてのトークンID
      scores (torch.FloatTensor): 各トークンにおける生成確率のスコア
      **kwargs: その他のgenerateメソッドから渡される引数(使用しない)
    Returns: 
      bool: いずれかのキーワードが検出された場合はTrue(停止)、それ以外はFalse(続行)
    """
    generated_token_ids = input_ids[0, self.input_length:] # 入力プロンプト部分を除外し、純粋な生成部分のみを取得
    generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True) # 入力プロンプト部分を除いた、生成したトークンを人間が読めるようにデコードし、文字列に

    for keyword in self.keywords:
      if keyword in generated_text:
        print(f"DEBUG: 停止ワード '{keyword}' が検出されました。生成を停止します。")
        return True # 生成を停止
    return False # 生成を続行


def extract_text_from_pdfs(pdf_folder_path: Path) -> List[Dict[str, str]]:
  """
  指定されたフォルダ内のすべてのPDFファイルからテキストを抽出し、出典情報と共にチャンクのリストとして返す
  Args:
    pdf_folder_path (Path): PDFファイルが格納されているフォルダのパス
  Returns:
    List[Dict[str, str]]: 各チャンクが {'text': '抽出テキスト', 'source': '出典情報'} 形式の辞書であるリスト
    (フォルダが見つからない、またはPDFが存在しない場合は空のリストを返す)
  """
  if not pdf_folder_path.is_dir():
    print(f"エラー: '{pdf_folder_path}' フォルダが見つかりません。")
    return []

  pdf_files = list(pdf_folder_path.glob("*.pdf"))
  if not pdf_files:
    print(f"警告: '{pdf_folder_path}' にPDFファイルが見つかりませんでした。")
    return []

  print(f"{len(pdf_files)}個のPDFファイルからテキストを抽出します...")
  
  text_chunks = []
  for pdf_path in pdf_files:
    try:
      reader = PdfReader(pdf_path)
      for page_num, page in enumerate(reader.pages): # pageにPDFの各ページ内容を、それに合わせて順にpage_numにページ番号を取得
        text = page.extract_text() # ページからテキストを読み込んで抽出、文字列に
        if text:
          source_info = f"（出典: {pdf_path.name}, ページ: {page_num + 1}）"
          text_chunks.append({"text": text, "source": source_info})
    except Exception as e:
      print(f"警告: '{pdf_path.name}' の処理中にエラーが発生しました: {e}")

  print("テキストの抽出とチャンク分割が完了しました。")
  return text_chunks


def create_vector_db(text_chunks: List[Dict[str, str]]) -> Tuple[Union[faiss.Index, None], List[Dict[str, str]]]:
  """
  テキストチャンクのリストからベクトルデータベース(Faiss)を構築する
  Args:
    text_chunks (List[Dict[str, str]]): テキストチャンクと出典情報のリスト
  Returns:
    Tuple[Union[faiss.Index, None], List[Dict[str, str]]]: 構築されたFaissインデックスと、元のチャンク情報のリストのタプル
    (テキストチャンクが空の場合、(None, None)を返す)
  """
  if not text_chunks:
    print("警告: テキストチャンクが空のため、ベクトルDBは作成されません。")
    return None, None

  model = SentenceTransformer(VECTOR_MODEL_NAME) # ベクトル化モデルのロード
  print(f"\n'{VECTOR_MODEL_NAME}' モデルをロードしました。")

  texts_to_embed = [chunk["text"] for chunk in text_chunks] # text_chunks({"text": str, "source": str})から、"text"部分のみを抽出してリスト化
  
  print(f"{len(texts_to_embed)}個のテキストチャンクをベクトル化します...")
  embeddings = model.encode(texts_to_embed, show_progress_bar=True)

  d = embeddings.shape[1] # テキストのベクトル次元数を取得
  index = faiss.IndexFlatL2(d) # FAISSインデックスを作成
  index.add(embeddings.astype('float32')) # 生成された埋め込みベクトルをfloat32型に変換(互換性のため)した後、初期化されたFAISSインデックスに追加し、検索可能な状態にする

  print(f"\nベクトルDB (Faissインデックス) の構築が完了しました。")
  print(f"インデックスされたベクトル数: {index.ntotal}")

  return index, text_chunks


def search_relevant_chunks(query: str, index: faiss.Index, text_chunks: List[Dict[str, str]], k: int) -> List[Dict[str, str]]:
  """
  ベクトルDBから質問に関連するチャンクを検索する
  Args:
    query (str): ユーザーからの質問文字列
    index (faiss.Index): 構築済みのFaissインデックス
    text_chunks (List[Dict[str, str]]): 元のテキストチャンクのリスト
    k (int): 検索する関連チャンクの数
  Returns:
    List[Dict[str, str]]: 関連性の高いチャンクのテキストと出典のリスト
  """
  # ユーザーの質問文字列をall-MiniLM-L6-v2モデルでベクトル化し、FAISSが要求するfloat32型に変換してquery_vectorに代入
  query_vector = VECTOR_MODEL.encode([query]).astype('float32')
  # ユーザーの質問ベクトルに対して、FAISSインデックスに格納されているすべてのチャンクベクトルの中から、ユークリッド距離が最も近いk個のチャンクを見つけ出し、その距離と元のインデックスを返している
  # 実際のチャンクの中から、関係性の高い要素の番号だけ入手するイメージ
  distances, indices = index.search(query_vector, k)
  # 関係性の高いk個の要素の文字列(文章)を実際のチャンクから取得
  relevant_chunks = [text_chunks[i] for i in indices[0]]
  return relevant_chunks


def generate_answer_with_rag(query: str, relevant_chunks: List[Dict[str, str]]) -> str:
  """
  検索されたチャンクを元に、LLMで回答を生成する
  Args:
    query (str): ユーザーからの質問文字列
    relevant_chunks (List[Dict[str, str]]): 検索によって取得された関連チャンクのリスト
  Returns:
    str: LLMによって生成された回答と出典情報を含む文字列
  """
  context_parts = []
  source_info_set = set() # 重複する要素は自動的に排除
  for chunk in relevant_chunks:
    context_parts.append(chunk['text'])
    source_info_set.add(chunk['source'])
  context = "\n\n".join(context_parts)
  
  # プロンプトの調整：質問と回答の間に改行を追加して視認性向上
  prompt_template = f"""
  質問に簡潔に日本語で答えてください。

  参考情報:
  {context}

  質問: {query}

  回答:
  """
  prompt_template = prompt_template.strip() # 先頭と末尾の不要な空白を削除

  print(f"DEBUG: prompt_templateの文字数: {len(prompt_template)}")
  # LLMの最大入力トークン長を使用
  print(f"DEBUG: モデルの最大入力トークン長（設定値）: {LLM_MAX_INPUT_LENGTH}")

  input_ids = LLM_TOKENIZER(
    prompt_template, # トークン化する対象の文字列
    return_tensors="pt", # 上記のトークン化の結果をPyTorchのtorch.Tensor型で返す(PyTorchモデルのLLM_MODELに渡すため)
    truncation=True, # max_lengthを超える内容のとき、超過部分を切り捨てる
    max_length=LLM_MAX_INPUT_LENGTH # トークン化後の最大長さ
  ).input_ids.to(LLM_MODEL.device) # LLM_MODEL.device(CPU)に、トークン化の結果のうち、モデルへの入力となるトークンIDのテンソルを移動
  
  print(f"DEBUG: トークン化後のinput_idsの長さ: {input_ids.shape[1]}")

  # StoppingCriteriaListを作成
  # 停止条件のオブジェクトを、transformersライブラリがgenerateメソッドで利用できる形式のリストにまとめる
  stopping_criteria = StoppingCriteriaList([
    StopOnKeywords(STOP_KEYWORDS, LLM_TOKENIZER, input_ids.shape[1])
  ])

  outputs = LLM_MODEL.generate(
    input_ids,
    max_new_tokens=MAX_NEW_TOKENS, # 生成する文章の最大トークン量
    pad_token_id=LLM_TOKENIZER.pad_token_id, # パディングトークンIDを指定
    repetition_penalty=1.1, # 同じフレーズの繰り返し生成を抑制するペナルティ
    do_sample=True, # サンプリングを有効化して多様な出力を生成
    temperature=0.7,
    top_k=50, # 最も確率の高い上位k個のトークンの中からのみサンプリングを行う
    top_p=0.95, # 累積確率がtop_pを超えるまでのトークンからサンプリングを行う
    stopping_criteria=stopping_criteria, # 生成停止条件を指定
  )
  
  # LLMが数値として出力した結果を、トークナイザーを使って人間が読めるテキスト形式に変換し、その際に不要な特殊トークンを除去する
  response_text = LLM_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
  
  # プロンプト部分(prompt_template)を除去する処理
  # 出力の先頭がprompt_templateから始まる場合、それを除去(基本)
  # 出力の先頭がprompt_templateから始まらない場合、前後の余分な空白を削除(例外)
  if response_text.startswith(prompt_template):
    answer = response_text[len(prompt_template):].strip()
  else:
    answer = response_text.strip()
  
  # 停止ワードで生成が止まった場合、そのワード自体がanswerに含まれていることがあるため除去
  for keyword in STOP_KEYWORDS:
    if keyword in answer:
      answer = answer.split(keyword)[0].strip()
      # print(f"DEBUG: 後処理で停止ワード '{keyword}' を除去しました。") # デバッグ用

  # 出典情報を回答の末尾に追加する
  if source_info_set:
    answer += "\n\n" + "\n".join(sorted(list(source_info_set)))

  return answer


def chatbot_interface(question: str, history: List[List[str]]) -> str:
  """
  GradioのUIからの入力を処理し、回答を返す
  Args:
    question (str): ユーザーからの質問文字列
    history (List[List[str]]): Gradioチャットインターフェースでの会話履歴
  Returns:
    str: 処理結果としてユーザーに表示される回答文字列
  """
  if not question:
    return "質問を入力してください。"
  print(f"\n質問を受け付けました: {question}")
  
  # FAISS_INDEXとCHUNKS_WITH_SOURCEがグローバル変数として初期化されていることを確認
  if FAISS_INDEX is None or CHUNKS_WITH_SOURCE is None:
    return "エラー: ベクトルDBが初期化されていません。プログラムを再実行してください。"

  # 関連チャンク検索
  relevant_chunks = search_relevant_chunks(question, FAISS_INDEX, CHUNKS_WITH_SOURCE, NUM_RELEVANT_CHUNKS)
  
  # 回答生成
  answer = generate_answer_with_rag(question, relevant_chunks)
  
  print(f"回答を生成しました: {answer}")
  return answer


# グローバルリソースをWebUIのコールバックで参照する必要があるため、メイン関数としない
if __name__ == "__main__":
  PDF_FOLDER = Path("./documents") # パスを定数として定義

  # 関数を呼び出してテキストチャンクを取得
  chunks = extract_text_from_pdfs(PDF_FOLDER)
  FAISS_INDEX, CHUNKS_WITH_SOURCE = create_vector_db(chunks)
  
  if FAISS_INDEX is None: # create_vector_dbがNoneを返す可能性を考慮
    print("エラー: ベクトルDBの作成に失敗したため、プログラムを終了します。")
    exit(1)

  # GradioのUIを起動
  # Gradioライブラリを使って、作成したチャットボットをWebアプリケーションとして起動し、ユーザーがブラウザで操作できるインターフェースを提供
  print(f"\n--- {GRADIO_TITLE} UIを起動します ---")
  print("Webブラウザで表示されるURLにアクセスしてください。終了するにはCtrl+Cを押してください。")
  iface = gr.ChatInterface( # 対話型のチャットインターフェースを作成
    chatbot_interface, # ユーザーがUI上でメッセージを送信する度に呼び出し
    title=GRADIO_TITLE, # Webページの上部に表示されるタイトル名
    description=GRADIO_DESCRIPTION # Webページのタイトル下に表示される説明文
  ).launch(max_flag_time=GRADIO_MAX_FLAG_TIME) # Webブラウザからアクセス可能な形で公開し、Gradioサーバーを起動(URLをコンソールに出力)＋長時間かかる処理でも接続が切れないようにするためのタイムアウト設定