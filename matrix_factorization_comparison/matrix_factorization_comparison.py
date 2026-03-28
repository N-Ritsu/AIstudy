import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from surprise import NMF, SVD as FunkSVD, SVDpp, Dataset, Reader
from surprise.model_selection import KFold
from surprise.trainset import Trainset

# データファイルのパス
DATA_DIR = Path("./ml-100k-data/ml-100k")
RATING_DATA_PATH = DATA_DIR / "u.data"
ITEM_DATA_PATH = DATA_DIR / "u.item"

# 乱数シード
RANDOM_STATE = 42
# 交差検証の分割数
N_SPLITS = 3
# 疎なデータセットを生成する際のデータ削除率
SPARSITY_LEVEL = 0.99
# Naive SVDで使用する次元数
SVD_N_COMPONENTS = 20
# NMFの解釈性可視化で使用する次元数
NMF_N_FACTORS = 15
# 解釈性可視化の対象ユーザーID
TARGET_USER_ID = '1'

# --- 型エイリアス ---
# 複雑な型定義に別名を付与し、コードの可読性を向上させる。
# 例: {"RMSE": 0.95, "MAE": 0.75, "Time": 10.2}
MetricsDict = Dict[str, float]
# 例: {"FunkSVD": {"RMSE": ..., "MAE": ...}, "NMF": {...}}
ResultsDict = Dict[str, MetricsDict]


def get_movielens_data() -> pd.DataFrame:
    """
    MovieLens 100kデータセットの評価データをローカルファイルから読み込む。
    ユーザーに対する各映画の評価を入手する。
    Args:
        None
    Returns:
        pd.DataFrame: ユーザーID(uid), アイテムID(iid), 評価(rating)を含むDataFrame。
    Raises:
        FileNotFoundError: データファイル(u.data)が見つからない場合に送出。
    """
    print(f"ローカルファイル '{RATING_DATA_PATH}' を読み込み中...")
    if not RATING_DATA_PATH.exists():
        print("\nエラー: データファイルが見つかりません。")
        print("まずpython download_movielens.pyを実行して、データをダウンロードしてください。")
        raise FileNotFoundError(f"{RATING_DATA_PATH} not found.")

    # SurpriseはIDを内部で文字列として扱うため、読み込み時にstr型を指定。
    # これにより後続処理との整合性を保つ。
    df = pd.read_csv(
        RATING_DATA_PATH,
        sep='\t', # seq: タブ区切り
        names=['uid', 'iid', 'rating', 'timestamp'], # 列名の指定
        dtype={'uid': str, 'iid': str, 'rating': float} # IDを文字列、評価を浮動小数点数として読み込む
    )
    df = df.drop('timestamp', axis=1) # タイムスタンプは分析に使用しないため削除
    print("評価データの読み込み完了。\n")
    return df


def load_movie_titles() -> Dict[str, str]:
    """
    u.itemファイルから映画IDとタイトルの対応辞書を生成する。
    Args:
        None
    Returns:
        Dict[str, str]: 映画ID(iid)をキー、映画タイトルをバリューとする辞書。
    """
    if not ITEM_DATA_PATH.exists():
        print("エラー: 映画タイトルファイルが見つかりません。")
        return {}

    # アイテムID(iid)を文字列として読み込み、辞書を生成。
    df_items = pd.read_csv(
        ITEM_DATA_PATH,
        sep='|',
        encoding='latin-1', # ファイルのエンコーディング指定
        usecols=[0, 1], # 必要な列のみ読み込む（iidとtitle）
        names=['iid', 'title'],
        dtype={'iid': str}
    )
    # pd.Seriesのindexをキー、値をバリューとして効率的に辞書へ変換。
    return pd.Series(df_items.title.values, index=df_items.iid).to_dict()


def create_sparse_dataset(df: pd.DataFrame, sparsity_level: float) -> pd.DataFrame:
    """
    既存のDataFrameから、指定された割合のデータをランダムに削除し、疎なデータセットを生成する。
    Args:
        df (pd.DataFrame): 元となる評価データ。
        sparsity_level (float): 削除するデータの割合 (0.0から1.0)。
    Returns:
        pd.DataFrame: データが間引かれた疎なDataFrame。
    """
    print(f"元のデータから{sparsity_level*100:.0f}%をランダムに削除し、極端に疎なデータセットを作成中...")
    # frac: 元のDataFrameからサンプリングする割合を指定
    # 1.0 - sparsity_level: 残すデータの割合
    sparse_df = df.sample(frac=1.0 - sparsity_level, random_state=RANDOM_STATE)
    print(f"作成完了。データ数: {len(df)} -> {len(sparse_df)}\n")
    return sparse_df


def evaluate_surprise_models(df: pd.DataFrame) -> ResultsDict:
    """
    Surpriseライブラリに実装された複数のモデルを交差検証により評価する。
    Args:
        df (pd.DataFrame): 評価対象のデータセット。
    Returns:
        ResultsDict: 各モデルの評価指標(RMSE, MAE)と計算時間を格納した辞書。
    """
    print("--- Surpriseライブラリのモデル群を評価開始 ---")
    reader = Reader(rating_scale=(1, 5)) # 評価値の範囲を指定。MovieLens 100kは1から5の整数評価。
    data = Dataset.load_from_df(df[['uid', 'iid', 'rating']], reader)
    kf = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE) # データをK分割し、交差検証を行うためのクラス。random_stateで分割の再現性を確保。

    # 評価対象モデルの辞書定義。ループによる一括評価を可能にする。
    models = {
        "FunkSVD": FunkSVD(random_state=RANDOM_STATE),
        "NMF": NMF(random_state=RANDOM_STATE),
        "SVD++": SVDpp(random_state=RANDOM_STATE),
    }

    results: ResultsDict = {}
    # 各モデルの評価ループ
    for name, model in models.items():
        print(f"モデル: {name} の学習・評価中...")
        start_time = time.time()

        rmses, maes = [], []

        # 交差検証のループ
        # `KFold.split()` は、学習/テスト用のデータ分割インデックスを生成するジェネレータを返す。
        for trainset, testset in kf.split(data):
            # 学習データでモデルを訓練
            model.fit(trainset)
            # テストデータで予測を生成
            predictions = model.test(testset)
            
            # 予測結果から真の値と予測値のリストを作成
            true_ratings = [p.r_ui for p in predictions]
            estimated_ratings = [p.est for p in predictions]
            
            # 評価指標を計算
            rmses.append(np.sqrt(mean_squared_error(true_ratings, estimated_ratings)))
            maes.append(np.mean(np.abs(np.array(true_ratings) - np.array(estimated_ratings))))

        # 全分割における指標の平均値を最終的な評価値として採用
        avg_rmse = np.mean(rmses) if rmses else float('inf')
        avg_mae = np.mean(maes) if maes else float('inf')

        duration = time.time() - start_time
        results[name] = {"RMSE": avg_rmse, "MAE": avg_mae, "Time": duration}
        print(f"  完了 (RMSE: {avg_rmse:.4f}, Time: {duration:.2f}s)")

    print("--- Surpriseモデル群の評価完了 ---\n")
    return results


def evaluate_naive_svd(train_df: pd.DataFrame, test_df: pd.DataFrame) -> ResultsDict:
    """
    欠損値を全体の平均評価値で補完する単純なSVDを評価する。
    Args:
        train_df (pd.DataFrame): 学習用データ。
        test_df (pd.DataFrame): テスト用データ。
    Returns:
        ResultsDict: Naive SVDモデルの評価指標と計算時間を格納した辞書。
    """
    print("--- Naive SVDモデルを評価開始 ---")
    start_time = time.time()

    if train_df.empty or test_df.empty:
        print("データが不足しているため、評価をスキップします。")
        return {"Naive SVD": {"RMSE": float('inf'), "MAE": float('inf'), "Time": 0}}

    # 1. ユーザー×アイテムの評価行列を生成
    rating_matrix = train_df.pivot(index='uid', columns='iid', values='rating')
    
    # 2. 欠損値(NaN)を学習データ全体の平均評価値で補完
    mean_rating = train_df['rating'].mean()
    rating_matrix_filled = rating_matrix.fillna(mean_rating)

    # 3. TruncatedSVDを適用し、行列を分解
    # ユーザーの好みや映画の特徴のグループを、20個の因子として圧縮する
    svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=RANDOM_STATE) # SVDの次元数を指定
    user_factors = svd.fit_transform(rating_matrix_filled) # 20個の因子で、各ユーザーを学習、個々人の傾向を20次元の因子で表す(ユーザー因子行列)
    item_factors = svd.components_.T # 各映画を20次元の因子で表す(アイテム因子行列)。

    # 4. 因子行列の積から、全評価値を予測した行列を再構成
    # 個々のユーザーを表した因子行列と、個々の映画を表した因子行列の積を取ることで、そのユーザーがその映画に対してどのような評価をするかを予測する行列を得る。
    pred_matrix = np.dot(user_factors, item_factors.T)
    # 再構成した行列のインデックスとカラムを元の評価行列に合わせる
    pred_df = pd.DataFrame(pred_matrix, index=rating_matrix_filled.index, columns=rating_matrix_filled.columns)

    # 5. テストデータに含まれる評価の予測値を取得し、精度を算出
    preds, actuals = [], []
    for _, row in test_df.iterrows():
        user_id, item_id, rating = row['uid'], row['iid'], row['rating']
        # テストデータのユーザー/アイテムが学習データに存在する場合のみ評価対象とする
        if user_id in pred_df.index and item_id in pred_df.columns:
            preds.append(pred_df.loc[user_id, item_id])
            actuals.append(rating)

    rmse = np.sqrt(mean_squared_error(actuals, preds)) if actuals else float('inf')
    mae = np.mean(np.abs(np.array(actuals) - np.array(preds))) if actuals else float('inf')
    duration = time.time() - start_time

    print(f"完了 (RMSE: {rmse:.4f}, Time: {duration:.2f}s)\n")
    return {"Naive SVD": {"RMSE": rmse, "MAE": mae, "Time": duration}}


def run_comparison(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    指定のデータセットで全モデルの比較を実行し、結果をDataFrameで返す。
    Args:
        dataset_name (str): データセットの名称（例: "標準", "疎"）。
        df (pd.DataFrame): 評価対象のデータセット。
    Returns:
        pd.DataFrame: 全モデルの評価結果をまとめたDataFrame。
    """
    print(f"===== {dataset_name} データセットでの比較を開始 =====")

    if len(df) < 10: # データが極端に少ない場合は評価をスキップ
        print("データが少なすぎて評価できません。")
        return pd.DataFrame()

    # Naive SVD用にデータを分割。Surpriseモデルは内部で交差検証を行う。
    train_df = df.sample(frac=0.8, random_state=RANDOM_STATE)
    test_df = df.drop(train_df.index)

    # 各評価関数を呼び出し、結果を統合
    results = evaluate_surprise_models(df)
    results.update(evaluate_naive_svd(train_df, test_df))

    # 結果を整形してDataFrameへ変換
    results_df = pd.DataFrame(results).T
    results_df['Dataset'] = dataset_name

    print(f"===== {dataset_name} データセットでの比較が完了 =====\n")
    return results_df


def plot_results(full_results_df: pd.DataFrame) -> None:
    """
    全データセットの評価結果を棒グラフで可視化し、画像として保存する。
    Args:
        full_results_df (pd.DataFrame): `run_comparison`から得られた結果を結合したDataFrame。
    Returns:
        None
    """
    print("\n--- 性能比較結果を可視化しています ---")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("行列因子分解モデルの性能比較", fontsize=20, fontweight='bold')
    
    # 評価失敗時の無限大の値をNaNに置換し、プロットエラーを防止。
    plot_df = full_results_df.replace([np.inf, -np.inf], np.nan)

    # グラフ1: 予測精度(RMSE)の比較
    sns.barplot(data=plot_df, x='RMSE', y=plot_df.index, hue='Dataset', ax=axes[0])
    axes[0].set_title("予測精度 (RMSE) の比較 (低いほど良い)", fontsize=15)
    axes[0].set_xlabel("RMSE")
    axes[0].set_ylabel("モデル")
    axes[0].grid(axis='x')

    # グラフ2: 学習時間の比較
    sns.barplot(data=plot_df, x='Time', y=plot_df.index, hue='Dataset', ax=axes[1])
    axes[1].set_title("学習時間の比較 (秒)", fontsize=15)
    axes[1].set_xlabel("学習時間 (秒)")
    axes[1].set_ylabel("") # 左のグラフとラベルが重複するため空にする
    axes[1].grid(axis='x')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # タイトルとの重なりを防止
    plt.savefig("mf_comparison_results.png")
    plt.show()
    print("性能比較グラフを 'mf_comparison_results.png' として保存しました。")


def visualize_nmf_interpretation(model: NMF, trainset: Trainset, item_titles: Dict[str, str], user_id: str) -> None:
    """
    NMFモデルの学習結果を解釈し、可視化する。
    Args:
        model (NMF): 学習済みのNMFモデル。
        trainset (Trainset): 学習に使用したデータセット。
        item_titles (Dict[str, str]): アイテムIDとタイトルの対応辞書。
        user_id (str): 可視化対象のユーザーID。
    Returns:
        None
    """
    print("--- NMFモデルの解釈性 可視化 ---")

    # --- 1. アイテム因子（潜在的なジャンルや特徴）の可視化 ---
    print("\n--- 各因子（潜在的ジャンル）を代表する映画トップ10 ---")
    # model.qi は (アイテム数 x 因子数) のアイテム因子行列
    item_factors = model.qi 
    
    for factor_idx in range(model.n_factors):
        # 各因子について、その因子の値が大きいアイテムを上位から取得
        top_item_inner_ids = np.argsort(item_factors[:, factor_idx])[::-1][:10]
        # Surpriseの内部IDを元のアイテムIDに変換し、タイトルを取得
        top_item_titles = [item_titles.get(trainset.to_raw_iid(iid), "N/A") for iid in top_item_inner_ids]
        
        print(f"\n【因子 {factor_idx}】")
        for i, title in enumerate(top_item_titles):
            print(f"  {i+1:2d}. {title}")

    # --- 2. ユーザー因子（個人の好み）の可視化 ---
    print(f"\n--- ユーザーID {user_id} の好みプロファイル (レーダーチャート) ---")
    try:
        # Surpriseの内部ID（0からの連番）への変換が必要
        user_inner_id = trainset.to_inner_uid(user_id)
        # model.pu は (ユーザー数 x 因子数) のユーザー因子行列
        user_profile = model.pu[user_inner_id]

        # レーダーチャートの描画
        labels = [f"因子 {i}" for i in range(model.n_factors)]
        angles = np.linspace(0, 2 * np.pi, model.n_factors, endpoint=False).tolist()
        angles += angles[:1] # 円を閉じるために始点を末尾に追加

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        values = user_profile.tolist()
        values += values[:1] # 同様に、値を閉じるために始点の値を追加
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"ユーザー {user_id}")
        ax.fill(angles, values, alpha=0.25)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title(f"ユーザーID {user_id} の好みプロファイル", size=15)
        
        plt.savefig("nmf_user_profile_radar.png")
        plt.show()
        print("レーダーチャートを 'nmf_user_profile_radar.png' として保存しました。")

    except ValueError:
        # 学習データに存在しないユーザーIDが指定された場合のエラーハンドリング
        print(f"エラー: ユーザーID '{user_id}' は学習データセットに存在しません。")


def main() -> None:
    """
    4つの行列因子分解モデルの性能比較とNMFの解釈性可視化を実行するメイン処理。
    Args:
        None
    Returns:
        None
    """
    # --- 1. データ準備 ---
    try:
        movielens_df = get_movielens_data()
        item_titles = load_movie_titles()
    except FileNotFoundError:
        return # データがなければ処理を終了

    # 比較のため、意図的に疎にしたデータセットも生成
    sparse_movielens_df = create_sparse_dataset(movielens_df, sparsity_level=SPARSITY_LEVEL)
    
    # --- 2. モデル性能比較の実行 ---
    results_standard = run_comparison("MovieLens 100k (標準)", movielens_df)
    results_sparse = run_comparison("MovieLens 100k (極端に疎)", sparse_movielens_df)
    
    full_results = pd.concat([results_standard, results_sparse])
    
    # --- 3. 性能比較結果の表示と可視化 ---
    print("\n\n" + "="*50)
    print("           最終的な性能評価まとめ")
    print("="*50)
    print(full_results.to_string(float_format="%.4f"))
    print("="*50)
    
    plot_results(full_results)
    
    # --- 4. NMFモデルの解釈性可視化 ---
    # 解釈性分析は、より多くの情報を含む元のデータセット全体で実行
    print("\n--- NMFモデルの解釈性分析のために全データで再学習します ---")
    reader = Reader(rating_scale=(1, 5))
    full_data = Dataset.load_from_df(movielens_df[['uid', 'iid', 'rating']], reader)
    full_trainset = full_data.build_full_trainset()
    
    nmf_model = NMF(n_factors=NMF_N_FACTORS, random_state=RANDOM_STATE)
    nmf_model.fit(full_trainset)
    
    visualize_nmf_interpretation(
        model=nmf_model,
        trainset=full_trainset,
        item_titles=item_titles,
        user_id=TARGET_USER_ID
    )

if __name__ == '__main__':
    main()