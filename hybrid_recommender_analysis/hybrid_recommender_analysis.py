from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# データセットが格納されているディレクトリへのパス
DATA_DIR = Path('ml-100k-data/ml-100k')
# 評価のためにあらかじめ分割されている訓練・テストデータセットのパス
U1_BASE_PATH = DATA_DIR / 'u1.base' # 訓練データ
U1_TEST_PATH = DATA_DIR / 'u1.test' # テストデータ
U_ITEM_PATH = DATA_DIR / 'u.item' # アイテム（映画）情報
U_GENRE_PATH = DATA_DIR / 'u.genre' # ジャンル情報

# 評価データのカラム名定義
RATING_COLUMNS = ['user_id', 'item_id', 'rating', 'timestamp']

# 協調フィルタリングで考慮する近傍ユーザーの数
K_NEIGHBORS = 40
# 各ユーザーに対して生成する推薦リストのアイテム数
N_RECOMMENDATIONS = 30
# ハイブリッド推薦における協調フィルタリング(CF)とコンテンツベース(CB)の重み
# HYBRID_ALPHA = 1.0 のときCFと同一、0.0のときCBと同一になる
HYBRID_ALPHA = 0.5


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    MovieLens 100kデータセットを読み込み、前処理を行う。
    Args:
        None
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
            - train_df (pd.DataFrame): 訓練用の評価データ。
            - test_df (pd.DataFrame): テスト用の評価データ。
            - items_df (pd.DataFrame): 映画のID、タイトル、ジャンル情報。
            - genres (List[str]): 映画のジャンル名のリスト。
    Raises:
        FileNotFoundError: データセットのファイルが見つからない場合に発生。
    """
    print("--- [1/5] データの読み込みと前処理 ---")
    try:
        # --- 評価データの読み込み ---
        # u1.base と u1.test は、ユーザーごとに評価の一部が分けられたファイル
        train_df = pd.read_csv(U1_BASE_PATH, sep='\t', names=RATING_COLUMNS)
        test_df = pd.read_csv(U1_TEST_PATH, sep='\t', names=RATING_COLUMNS)

        # --- ジャンル情報の読み込み ---
        # u.genre ファイルからジャンル名のリストを生成する (全19ジャンル)
        with open(U_GENRE_PATH, 'r') as f:
            genres = [line.strip().split('|')[0] for line in f if line.strip()]

        # --- アイテム（映画）情報の読み込み ---
        # u.itemファイルは '|' 区切りで、合計24列の情報を持つ。
        # 今回必要なのはitem_id・titleと19個のジャンル列(数値)のみ。

        # 必要な要素のみのリストを作成
        #   - 0列目: item_id
        #   - 1列目: title
        #   - 5列目から23列目: 19個のジャンル情報 (0 or 1)
        item_cols_to_use = [0, 1] + list(range(5, 24))

        # 上記のリストは、全て数値。それぞれに対応する列名を定義する。
        item_col_names = ['item_id', 'title'] + genres

        # read_csv実行時に、読み込む列(usecols)と列名(names)を指定する。
        #   encoding='latin-1'は、このデータセット特有の文字コードに対応するため。
        items_df = pd.read_csv(
            U_ITEM_PATH,
            sep='|',
            names=item_col_names,
            usecols=item_cols_to_use,
            encoding='latin-1'
        )

        # --- データクレンジング ---
        # 'unknown'ジャンルは分析に寄与しないため、DataFrameとジャンルリストの両方から除外する。
        # errors='ignore' は 'unknown' 列が存在しない場合にエラーを出さないためのオプション。
        if 'unknown' in items_df.columns:
            items_df = items_df.drop('unknown', axis=1)
        if 'unknown' in genres:
            genres.remove('unknown')

        print("データの読み込みと前処理が完了しました。")
        return train_df, test_df, items_df, genres
    
    except FileNotFoundError as e:
        print(f"エラー: データファイルが見つかりません: {e.filename}")
        print(f"'{DATA_DIR}' のパスが正しいか確認してください。")
        raise


# create_user_item_matrixで、計算用のデータフレームを作って、
# それを使ってcalculate_user_similarityでユーザー間の類似度を計算して、
# そのデータフレームでアイテム間の類似度も計算したいけど、現在は0か1でしかジャンルを表せていなくて、”深み”が分からず精度が低い
# → create_item_tfidf_matrixでアイテムのジャンルに重みづけを行ってからcalculate_item_similarityでアイテム間の類似度を計算している

def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    評価データフレームからユーザー-アイテム評価行列を作成する。
    類似度計算やスコア予測の基本となる。
    Args:
        df (pd.DataFrame): 'user_id', 'item_id', 'rating' を含む評価データ。
    Returns:
        pd.DataFrame: 行にuser_id、列にitem_id、値にratingを持つ評価行列。
                      評価がない箇所はNaNとなる。
    """
    # pivot_tableを使用して、縦長の評価データをユーザー×アイテムの横長な行列に変換する
    return df.pivot_table(index='user_id', columns='item_id', values='rating')


def calculate_user_similarity(rating_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    ユーザー-アイテム評価行列から、ユーザー間のコサイン類似度を計算する。
    Args:
        rating_matrix (pd.DataFrame): create_user_item_matrixで作成した評価行列。
    Returns:
        pd.DataFrame: ユーザー間の類似度スコアを格納した行列 (ユーザー×ユーザー)。
    """
    # コサイン類似度を計算する際、NaNは0として扱う（評価していない＝関心がないとみなす）
    filled_matrix = rating_matrix.fillna(0)
    # scikit-learnのcosine_similarityを用いて、全ユーザー間の類似度を一括で計算
    user_similarity = cosine_similarity(filled_matrix)
    # 計算結果(numpy配列)を、インデックスとカラムにuser_idを持つDataFrameに変換
    return pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)


def create_item_tfidf_matrix(items_df: pd.DataFrame, genres: List[str]) -> np.ndarray:
    """
    映画のジャンル情報から、TF-IDFに基づいたアイテムの特徴量行列を作成する。
    Args:
        items_df (pd.DataFrame): 映画のメタデータ（ジャンル情報を含む）。
        genres (List[str]): ジャンル名のリスト。
    Returns:
        np.ndarray: 各映画をジャンルで特徴量ベクトル化したTfidf行列。
    """
    # 各映画のジャンル情報を、機械学習モデルが扱えるテキスト形式に変換する。
    # 例: Action=1, Comedy=1 の映画 -> "Action Comedy" という文字列を生成
    # 生成した文字列を、genre_strという新しい列に格納。
    items_df['genre_str'] = items_df[genres].apply(
        lambda row: ' '.join(row.index[row == 1]), axis=1
    )
    
    # TfidfVectorizerを初期化
    # TF-IDFは、ジャンルの重要度や珍しさを考慮し、そのアイテムの核となるジャンルに高い数値を割り当てる特徴量数値化手法。
    tfidf_vectorizer = TfidfVectorizer()
    # ジャンル文字列をTF-IDFベクトルに変換
    return tfidf_vectorizer.fit_transform(items_df['genre_str'])


def calculate_item_similarity(tfidf_matrix: np.ndarray) -> np.ndarray:
    """
    アイテムのTF-IDF特徴量行列から、アイテム間のコサイン類似度を計算する。
    Args:
        tfidf_matrix (np.ndarray): create_item_tfidf_matrixで作成した特徴量行列。
    Returns:
        np.ndarray: アイテム間の類似度スコアを格納した行列 (アイテム×アイテム)。
    """
    # TF-IDFベクトル間のコサイン類似度を計算することで、アイテム間の内容的な近さを算出
    return cosine_similarity(tfidf_matrix)


def predict_scores(
    user_id: int,
    train_matrix: pd.DataFrame,
    user_sim_matrix: pd.DataFrame,
    item_sim_matrix: np.ndarray,
    items_df: pd.DataFrame
) -> Dict[str, pd.Series]:
    """
    一人のユーザーに対して、各推薦モデルの予測評価スコアを計算する。
    Args:
        user_id (int): 予測対象のユーザーID。
        train_matrix (pd.DataFrame): 訓練データのユーザー-アイテム評価行列。
        user_sim_matrix (pd.DataFrame): ユーザー間の類似度行列。
        item_sim_matrix (np.ndarray): アイテム間の類似度行列。
        items_df (pd.DataFrame): アイテムのメタデータ。
    Returns:
        Dict[str, pd.Series]: モデル名('CF', 'CB', 'Hybrid')をキー、予測スコア(pd.Series)を値とする辞書。Seriesのインデックスはitem_id。
    """
    # ユーザーがまだ評価していないアイテム（推薦候補）を特定
    user_ratings = train_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings.isnull()].index

    # --- 協調フィルタリング (CF) のスコア予測 ---
    # 対象ユーザーと類似度の高い上位K人のユーザー（近傍ユーザー）を選択
    similar_users = user_sim_matrix.loc[user_id].drop(user_id).nlargest(K_NEIGHBORS)
    
    cf_preds = {}
    for item_id in unrated_items:
        # 推薦候補アイテムを、近傍ユーザーがどのように評価しているかを取得
        neighbor_ratings = train_matrix.loc[similar_users.index, item_id].dropna()
        
        if not neighbor_ratings.empty:
            # 予測スコアは近傍ユーザーの評価値をユーザー類似度で重み付けした加重平均
            numerator = (neighbor_ratings * similar_users.loc[neighbor_ratings.index]).sum()
            denominator = similar_users.loc[neighbor_ratings.index].sum()
            if denominator > 0:
                cf_preds[item_id] = numerator / denominator
    cf_scores = pd.Series(cf_preds, name="CF_Score")

    # --- コンテンツベース (CB) のスコア予測 ---
    # ユーザーが過去に高く評価したアイテムを取得
    rated_items = user_ratings.dropna()
    
    cb_preds = {}
    for item_id in unrated_items:
        # 推薦候補アイテムの内部インデックスを取得
        # .locは低速なため、より高速なインデックス参照を行う
        unrated_idx = items_df.index[items_df['item_id'] == item_id][0]
        # ユーザーが評価済みアイテムの内部インデックスを取得
        rated_indices = items_df.index[items_df['item_id'].isin(rated_items.index)]
        
        # 推薦候補アイテムと、ユーザーが過去に評価した全アイテムとの類似度を取得
        similarities_to_rated = pd.Series(item_sim_matrix[unrated_idx, rated_indices], index=rated_items.index)

        # 予測スコアはアイテム間類似度をユーザーの過去の評価値で重み付けした加重平均
        numerator = (similarities_to_rated * rated_items).sum()
        denominator = similarities_to_rated.sum()
        if denominator > 0:
            cb_preds[item_id] = numerator / denominator
    cb_scores = pd.Series(cb_preds, name="CB_Score")

    # --- ハイブリッドのスコア予測 ---
    # CFとCBの両方でスコアが計算できたアイテムのみを対象とする
    common_items = cf_scores.index.intersection(cb_scores.index)
    
    # スコアの範囲が異なるモデルを組み合わせるため、スコアを正規化(0-1スケール)する
    # 稀にスコアが全て同じ値になりmax-min=0となる場合を避けるため、分母に小さな値(epsilon)を加える
    epsilon = 1e-9
    cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + epsilon)
    cb_norm = (cb_scores - cb_scores.min()) / (cb_scores.max() - cb_scores.min() + epsilon)
    
    # 各モデルの正規化済みスコアを、定数HYBRID_ALPHAで重み付けして結合する
    hybrid_scores = (
        HYBRID_ALPHA * cf_norm.loc[common_items] +
        (1 - HYBRID_ALPHA) * cb_norm.loc[common_items]
    )
    hybrid_scores.name = "Hybrid_Score"
    
    return {"CF": cf_scores, "CB": cb_scores, "Hybrid": hybrid_scores}


def evaluate_models(
    test_df: pd.DataFrame,
    all_predictions: Dict[int, Dict[str, pd.Series]]
) -> pd.DataFrame:
    """
    全テストユーザーの予測結果から、各モデルのPrecision@Nを計算する。
    Precision@N: 推薦リストの上位N件のうち、実際にユーザーが好んだアイテムがどれだけ含まれているかを示す精度指標。
    Args:
        test_df (pd.DataFrame): テスト用の評価データ。
        all_predictions (Dict[int, Dict[str, pd.Series]]):
            全テストユーザーに対する各モデルの予測スコア。
    Returns:
        pd.DataFrame: 各モデルの平均Precision@Nをまとめた結果。
    """
    print("\n--- [4/5] モデルの性能評価 ---")
    results: Dict[str, List[float]] = {"CF": [], "CB": [], "Hybrid": []}
    
    # テストデータに存在する全ユーザーに対してループ
    for user_id in test_df['user_id'].unique():
        # 訓練データに存在しない等の理由で予測が生成されなかったユーザーはスキップ
        if user_id not in all_predictions:
            continue
            
        # ユーザーがテストデータで高評価(ここでは4以上)を付けたアイテムが「正解」集合となる
        actual_positives = test_df[(test_df['user_id'] == user_id) & (test_df['rating'] >= 4)]['item_id'].tolist()
        
        # 正解がなければ、そのユーザーの精度は計算できないためスキップ
        if not actual_positives:
            continue

        # 各モデルについてPrecisionを計算
        for model_name in results.keys():
            user_preds = all_predictions[user_id].get(model_name)
            
            # 予測が存在しない、または空の場合はPrecision=0.0
            if user_preds is None or user_preds.empty:
                precision = 0.0
            else:
                # 予測スコアが高い上位N件を推薦リストとする
                recommended_items = user_preds.nlargest(N_RECOMMENDATIONS).index
                # 推薦リストと正解集合の積集合（ヒットしたアイテム）の数を数える
                hits = len(set(recommended_items) & set(actual_positives))
                # Precision = (ヒットした数) / (推薦した数)
                precision = hits / N_RECOMMENDATIONS
            
            results[model_name].append(precision)

    # 全ユーザーのPrecisionを平均して、モデルごとの最終的な性能スコアとする
    summary = {name: np.mean(scores) for name, scores in results.items() if scores}
    return pd.DataFrame([summary], index=[f"Precision@{N_RECOMMENDATIONS}"])


def main():
    """
    3つの推薦モデル（協調フィルタリング、コンテンツベース、ハイブリッド）を構築し、性能を比較するメイン処理。
    Args:
        None
    Returns:
        None
    Raises:
        FileNotFoundError: データセットのファイルが見つからない場合に発生。
        Exception: その他の予期せぬエラーが発生した場合に発生。
    """
    try:
        # --- データ読み込みと準備 ---
        train_df, test_df, items_df, genres = load_data()
        # 訓練データをユーザー-アイテム評価行列に変換
        train_matrix = create_user_item_matrix(train_df)

        # --- 各モデルの学習（主に類似度行列の事前計算）---
        print("--- [2/5] 各モデルの学習（類似度行列の計算） ---")
        user_sim_matrix = calculate_user_similarity(train_matrix)
        print("ユーザー類似度行列の計算が完了しました。")
        
        item_tfidf_matrix = create_item_tfidf_matrix(items_df, genres)
        item_sim_matrix = calculate_item_similarity(item_tfidf_matrix)
        print("アイテム類似度行列の計算が完了しました。")

        # --- 全テストユーザーに対する予測スコアの計算 ---
        # この処理はユーザー数とアイテム数が多いため、時間がかかることがある
        print("--- [3/5] 全テストユーザーの予測スコアを計算（時間がかかります） ---")
        all_predictions: Dict[int, Dict[str, pd.Series]] = {}
        test_users = test_df['user_id'].unique()
        
        for i, user_id in enumerate(test_users):
            # 訓練データに評価履歴が存在するユーザーのみ予測可能
            if user_id in train_matrix.index:
                all_predictions[user_id] = predict_scores(
                    user_id, train_matrix, user_sim_matrix, item_sim_matrix, items_df
                )
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(test_users)} 人のユーザーの処理が完了...")
        print("全予測スコアの計算が完了しました。")

        # --- モデルの性能評価 ---
        evaluation_results = evaluate_models(test_df, all_predictions)
        print("\n【性能比較結果】")
        print(evaluation_results.round(4))

        # --- 特定ユーザーへの推薦結果の表示---
        print("\n--- [5/5] 特定ユーザーへの推薦結果（例: ユーザーID 1） ---")
        target_user_id = 1
        if target_user_id in all_predictions:
            predictions_for_user = all_predictions[target_user_id]
            
            for model_name, scores in predictions_for_user.items():
                print(f"\n--- {model_name} による推薦トップ5 ---")
                if scores is None or scores.empty:
                    print("  推薦アイテムが見つかりませんでした。")
                    continue
                
                # 予測スコアが高い上位5件のアイテム情報を表示
                top_5_items = scores.nlargest(5)
                for item_id, score in top_5_items.items():
                    title = items_df.loc[items_df['item_id'] == item_id, 'title'].iloc[0]
                    print(f"  - {title} (予測スコア: {score:.3f})")
        else:
            print(f"ユーザーID {target_user_id} の予測結果は見つかりませんでした。")

    except FileNotFoundError:
        print("\n処理を中断しました。データセットのパスを確認してください。")
    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {e}")


if __name__ == '__main__':
    main()