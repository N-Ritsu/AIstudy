import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# --- ファイルパスとカラム定義 ---
DATA_DIR = Path('ml-100k') # ml-100kフォルダがスクリプトと同じ階層にあると仮定
U_DATA_PATH = DATA_DIR / 'u.data'
U_ITEM_PATH = DATA_DIR / 'u.item'
U1_BASE_PATH = DATA_DIR / 'u1.base'
U1_TEST_PATH = DATA_DIR / 'u1.test'

# カラム名
RATING_COLUMNS = ['user_id', 'item_id', 'rating', 'timestamp']
ITEM_COLUMNS = ['item_id', 'title'] # u.itemから読み込む主要なカラム

# --- デフォルトのハイパーパラメータ ---
# コマンドライン引数で指定されない場合のデフォルト値として使用
DEFAULT_K_NEIGHBORS = 1
DEFAULT_NUM_RECOMMENDATIONS = 1

def load_movielens_data(data_path: Path, items_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  MovieLens 100kデータセットを読み込む
  Args:
    data_path (Path): u.data (評価データ) のパス
    items_path (Path): u.item (アイテム情報) のパス
  Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: 評価データフレームとアイテム情報データフレーム
  """
  try:
    df = pd.read_csv(data_path, sep='\t', names=RATING_COLUMNS)
    # u.itemファイルは、item id | movie title | release date | video release date | IMDb URL | genres...といった形式で、|で区切られている
    # 最初の2列だけを読み込み、それ以降は無視
    items_df = pd.read_csv(
      items_path, sep='|', names=ITEM_COLUMNS + [f'col_{i}' for i in range(2, 24)],
      usecols=[0, 1], encoding='latin-1'
    )
    return df, items_df
  except FileNotFoundError as e:
    print(f"エラー: ファイルが見つかりません - {e.filename}")
    raise
  except Exception as e:
    print(f"データの読み込み中にエラーが発生しました: {e}")
    raise

def create_rating_matrix(df: pd.DataFrame) -> pd.DataFrame:
  """
  評価データフレームからユーザー-アイテム評価行列を作成する
  Args:
    df (pd.DataFrame): 'user_id', 'item_id', 'rating' を含むデータフレーム
  Returns:
    pd.DataFrame: ユーザーを行、アイテムを列、評価を値とする評価行列
  """
  # ピボットテーブルを使って評価行列を作成
  # 推薦アルゴリズムが直接利用できるような、ユーザーとアイテムの関係性を一目でわかる表形式のデータ構造に生データを変換
  rating_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating')
  return rating_matrix

def calculate_user_similarity(rating_matrix: pd.DataFrame) -> pd.DataFrame:
  """
  ユーザー間のコサイン類似度を計算する関数
  Args:
    rating_matrix (pd.DataFrame): ユーザー-アイテム評価行列
  Returns:
    pd.DataFrame: ユーザー間の類似度行列
  """
  # 欠損値を0で埋める
  # ゼロで埋めることで、評価していないアイテムが類似度に寄与しないようにする（簡略化されたアプローチ）
  filled_matrix = rating_matrix.fillna(0)

  # コサイン類似度を計算
  # sklearn.metrics.pairwise.cosine_similarity を使うと簡単
  user_similarity = cosine_similarity(filled_matrix)

  # 結果はNumPy配列なので、DataFrameに変換して見やすくする
  user_similarity_df = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)
  return user_similarity_df

def _get_base_recommendations(
  user_id: int,
  rating_matrix: pd.DataFrame,
  user_similarity_df: pd.DataFrame,
  k_neighbors: int,
  num_recommendations: int
) -> List[Tuple[int, float]]:
  """
  指定されたユーザーへのおすすめアイテムのIDと予測評価を生成するコア関数。
  ターゲットユーザーと類似度が高い他ユーザーを調べ、類似度上位k人の、ターゲットユーザーが評価していない映画に対する評価を調べ、類似度と評価を掛け合わせることでおすすめ度を計算
  Args:
    user_id (int): 推薦を生成するターゲットユーザーのID
    rating_matrix (pd.DataFrame): ユーザー-アイテム評価行列 (訓練データに基づく)
    user_similarity_df (pd.DataFrame): ユーザー間の類似度行列
    k_neighbors (int): 類似ユーザーの上位K人
    num_recommendations (int): 返すおすすめの数
  Returns:
    List[Tuple[int, float]]: 推薦されたアイテムIDと予測評価のリスト
  """
  # user_idがuser_similarity_dfのインデックスに存在するか確認
  # ユーザーが類似度行列に存在しない場合は推薦なし（コールドスタート対応）
  if user_id not in user_similarity_df.index:
    return []

  # ターゲットユーザーの評価を取得
  target_user_ratings = rating_matrix.loc[user_id]

  # ターゲットユーザーがまだ評価していないアイテムを特定
  unrated_items = target_user_ratings[target_user_ratings.isnull()].index
  # ターゲットユーザーがすべてのアイテムを評価していた場合、推薦できるものがないためreturn
  if len(unrated_items) == 0:
    return []

  # ターゲットユーザーと他のユーザーの類似度を取得(ターゲットユーザー自身との類似度は除く)
  # .drop: user_idのデータを排除、.loc: 類似度スコアを返す
  similar_users = user_similarity_df.loc[user_id].drop(user_id, errors='ignore') # errors='ignore'でuser_idがなくてもエラーにならない
  # ソート
  similar_users = similar_users.sort_values(ascending=False)

  # 上位k_neighbors人の近傍ユーザーを選択
  k_similar_users = similar_users.head(k_neighbors)

  predicted_ratings: Dict[int, float] = {}
  for item_id in unrated_items:
    # このアイテムを評価している近傍ユーザーの評価を取得
    # 類似度で重み付けするために、NaNは無視する
    item_ratings_by_similar_users = rating_matrix.loc[k_similar_users.index, item_id].dropna()

    if len(item_ratings_by_similar_users) > 0:
      # 類似度と評価を掛け合わせて、重み付け平均を計算
      # 類似度の合計で割ることで正規化
      numerator = 0.0
      denominator = 0.0
      for neighbor_user_id, neighbor_rating in item_ratings_by_similar_users.items():
        similarity_score = k_similar_users.loc[neighbor_user_id]
        numerator += similarity_score * neighbor_rating
        denominator += similarity_score

      if denominator > 0:
        predicted_ratings[item_id] = numerator / denominator
      else:
        # 類似度0のユーザーしか評価していないか、または類似度が非常に低いユーザーによる評価
        predicted_ratings[item_id] = 0.0
    else:
      # 近傍ユーザーが誰も評価していない場合
      predicted_ratings[item_id] = 0.0

  # 予測評価が高い順にソートし、上位num_recommendations個を返す
  return sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]

def convert_item_ids_to_titles(recommendations_by_id: List[Tuple[int, float]], items_df: pd.DataFrame) -> List[Tuple[str, float]]:
  """
  アイテムIDと予測評価のリストを、映画タイトルと予測評価のリストに変換する
  Args:
    recommendations_by_id (List[Tuple[int, float]]): アイテムIDと予測評価のリスト
    items_df (pd.DataFrame): アイテムIDとタイトルを含むデータフレーム
  Returns:
    List[Tuple[str, float]]: 映画タイトルと予測評価のリスト
  """
  final_recommendations_with_titles: List[Tuple[str, float]] = []
  for item_id, predicted_rating in recommendations_by_id:
    # items_dfから映画タイトルを取得
    movie_title = items_df[items_df['item_id'] == item_id]['title'].iloc[0]
    final_recommendations_with_titles.append((movie_title, predicted_rating))
  return final_recommendations_with_titles

def get_user_recommendations_with_titles(
  user_id: int,
  rating_matrix: pd.DataFrame,
  user_similarity_df: pd.DataFrame,
  items_df: pd.DataFrame,
  k_neighbors: int,
  num_recommendations: int
) -> List[Tuple[str, float]]:
  """
  指定されたユーザーへのおすすめアイテムを生成する（映画タイトル付き、ユーザー表示用）。
  Args:
    user_id (int): 推薦を生成するターゲットユーザーのID
    rating_matrix (pd.DataFrame): ユーザー-アイテム評価行列 (訓練データに基づく)
    user_similarity_df (pd.DataFrame): ユーザー間の類似度行列
    items_df (pd.DataFrame): アイテムIDとタイトルを含むデータフレーム
    k_neighbors (int): 類似ユーザーの上位K人
    num_recommendations (int): 返すおすすめの数
  Returns:
    List[Tuple[str, float]]: 推薦された映画のタイトルと予測評価のリスト
  """
  # コア推薦ロジックを呼び出し、アイテムIDと予測評価を取得
  recommendations_by_id = _get_base_recommendations(
    user_id, rating_matrix, user_similarity_df, k_neighbors, num_recommendations
  )
  
  # アイテムIDをタイトルに変換
  return convert_item_ids_to_titles(recommendations_by_id, items_df)

def evaluate_recommendations(
  test_matrix: pd.DataFrame,
  train_matrix: pd.DataFrame,
  user_similarity_df: pd.DataFrame,
  k_neighbors: int,
  num_recommendations: int
) -> float:
  """
  推薦モデルのPrecision@Kを評価する。
  テストデータでユーザーが高評価（3点以上）したアイテムを正解とする。
  Args:
    test_matrix (pd.DataFrame): テスト評価行列 (ユーザーの実際の評価)
    train_matrix (pd.DataFrame): 訓練評価行列 (推薦生成に使用)
    user_similarity_df (pd.DataFrame): ユーザー間の類似度行列 (訓練データに基づく)
    k_neighbors (int): 類似ユーザーの上位K人
    num_recommendations (int): 推薦するアイテムの数
  Returns:
    float: 全ユーザーの平均Precision@K
  """
  total_precision = 0.0
  num_evaluated_users = 0

  # テストデータに存在する全ユーザーに対して評価
  for user_id in test_matrix.index:
    # 訓練データに存在するユーザー、かつ、類似度行列に存在するユーザーのみを対象とする
    if user_id not in train_matrix.index or user_id not in user_similarity_df.index:
      continue

    # 評価が付いたアイテム全てを取得(NaNを除外)
    actual_high_rated_items_in_test = test_matrix.loc[user_id].dropna()
    # [actual_high_rated_items_in_test >= 3]により、評価値が3以上のアイテムを抽出
    actual_relevant_items_in_test = actual_high_rated_items_in_test[actual_high_rated_items_in_test >= 3].index.tolist()
    
    # テストデータで関連性の高い評価がないユーザーは評価からスキップ
    if not actual_relevant_items_in_test:
      continue 

    # コア推薦ロジックを呼び出し、アイテムIDと予測評価を取得（タイトル変換は不要）
    recommended_items_with_pred_rating = _get_base_recommendations(
      user_id, train_matrix, user_similarity_df, k_neighbors, num_recommendations
    )
    recommended_item_ids = [item_id for item_id, _ in recommended_items_with_pred_rating]

    # Precision@K の計算
    # 推薦されたアイテムのうち、実際にユーザーがテストデータで評価したアイテムの割合
    hits = 0
    for item_id in recommended_item_ids:
      if item_id in actual_relevant_items_in_test:
        hits += 1

    precision_at_k = hits / num_recommendations if num_recommendations > 0 else 0
    total_precision += precision_at_k
    num_evaluated_users += 1

  # 全ユーザーの平均Precision@Kを計算
  if num_evaluated_users > 0:
    avg_precision_at_k = total_precision / num_evaluated_users
    return avg_precision_at_k
  else:
    return 0.0

def main(args: argparse.Namespace) -> None:
  """
  Movielensコサイン類似度レコメンダーのメイン実行関数
  指定されたパラメータで推薦を生成し、評価を行う
  Args:
    args: コマンドライン引数
  Returns:
    None
  """

  # 1. データ読み込み
  print("\n[1] データの読み込みを開始...")
  try:
    full_df, items_df = load_movielens_data(U_DATA_PATH, U_ITEM_PATH)
    print(f"データフレームの最初の5行:\n{full_df.head()}")
    print(f"\nアイテム情報の最初の5行:\n{items_df.head()}")

    print("\nデータフレームの情報:")
    full_df.info()

    n_users_full = full_df['user_id'].nunique()
    n_items_full = full_df['item_id'].nunique()
    print(f"\nユニークなユーザー数 (全データ): {n_users_full}")
    print(f"ユニークなアイテム数 (全データ): {n_items_full}")

    train_df = pd.read_csv(U1_BASE_PATH, sep='\t', names=RATING_COLUMNS)
    test_df = pd.read_csv(U1_TEST_PATH, sep='\t', names=RATING_COLUMNS)
    print(f"\n訓練データとテストデータを読み込みました。")

  except Exception as e:
    print(f"初期データの読み込み中に致命的なエラーが発生しました: {e}")
    return

  # 2. 評価行列の作成
  print("\n[2] 評価行列の作成...")
  full_rating_matrix = create_rating_matrix(full_df) # 全データからの評価行列 (参考用)
  train_rating_matrix = create_rating_matrix(train_df)
  test_rating_matrix = create_rating_matrix(test_df)

  print(f"全評価行列の形状: {full_rating_matrix.shape}")
  print(f"訓練評価行列の形状: {train_rating_matrix.shape}")
  print(f"テスト評価行列の形状: {test_rating_matrix.shape}")

  # 3. ユーザー類似度行列の計算 (訓練データから)
  print("\n[3] 訓練データからユーザー類似度行列を計算...")
  train_user_similarity_df = calculate_user_similarity(train_rating_matrix)
  print(f"訓練ユーザー類似度行列の最初の5行5列:\n{train_user_similarity_df.iloc[:5, :5]}")

  # パラメータの設定
  k_neighbors_val = args.k_neighbors # コマンドライン引数から取得
  num_recommendations_val = args.num_recommendations # コマンドライン引数から取得

  print(f"\n--- 設定された推薦パラメータ ---")
  print(f"考慮する近傍ユーザー数 (k_neighbors): {k_neighbors_val}")
  print(f"推薦するアイテム数 (num_recommendations): {num_recommendations_val}")
  print(f"-------------------------------")

  # 4. 特定のユーザーへのおすすめ生成の例
  print("\n[4] 特定ユーザーへのおすすめを生成 (例: ユーザーID 100)...")
  user_id_to_recommend = args.user_id # コマンドライン引数

  if user_id_to_recommend in train_rating_matrix.index:
    recommendations_for_user = get_user_recommendations_with_titles(
      user_id_to_recommend,
      train_rating_matrix,
      train_user_similarity_df,
      items_df,
      k_neighbors=k_neighbors_val,
      num_recommendations=num_recommendations_val
    )

    print(f"ユーザーID {user_id_to_recommend} へのおすすめアイテム:")
    if recommendations_for_user:
      for title, predicted_rating in recommendations_for_user:
        print(f"映画タイトル: {title}, 予測評価: {predicted_rating:.2f}")
    else:
      print("推薦するアイテムが見つかりませんでした。")
  else:
    print(f"ユーザーID {user_id_to_recommend} は訓練データに存在しないか、類似度を計算できません。")

  # 5. モデルの評価
  print(f"\n[5] モデルの評価を開始します (Precision@{num_recommendations_val})...")
  
  avg_precision = evaluate_recommendations(
    test_rating_matrix,
    train_rating_matrix,
    train_user_similarity_df,
    k_neighbors=k_neighbors_val,
    num_recommendations=num_recommendations_val
  )
  print(f"平均 Precision@{num_recommendations_val}: {avg_precision:.4f}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Movielens recommender using cosine similarity.")
  parser.add_argument(
    '--user_id',
    type=int,
    help='Specify a user ID for recommendations (default: 100).',
    default=100
  )
  parser.add_argument(
    '--k_neighbors',
    type=int,
    default=DEFAULT_K_NEIGHBORS,
    help=f'Number of similar neighbors to consider (default: {DEFAULT_K_NEIGHBORS}).'
  )
  parser.add_argument(
    '--num_recommendations',
    type=int,
    default=DEFAULT_NUM_RECOMMENDATIONS,
    help=f'Number of recommendations to generate (default: {DEFAULT_NUM_RECOMMENDATIONS}).'
  )
  args = parser.parse_args()

  main(args)