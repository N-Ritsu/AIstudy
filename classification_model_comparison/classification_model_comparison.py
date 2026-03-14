import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from wcwidth import wcswidth

RANDOM_STATE = 42  # 乱数シード
TEST_SIZE = 0.3    # テストデータの割合


def create_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    3種類の機械学習用データセットを生成し、辞書形式で返す。
    - Linearly Separable: 線形分離が可能な基本的なデータセット。
    - Non-linear (Moons): 線形分離が不可能な三日月形状のデータセット。
    - More Complex Synthetic: 特徴量が多く、より複雑にクラスが混ざり合った高次元データセット。
    Args:
        None
    Returns:
        Dict[str, Tuple[np.ndarray, np.ndarray]]: キーがデータセット名、バリューが(特徴量データ, ラベルデータ)のタプル。
    """
    # データセット1: 理想的な線形分離可能データ
    # make_classification: 分類問題の練習用データを自動で生成する関数
    # n_samples: データ点の総数, n_features: 各データの特徴量の数(2: 2次元データとする), n_redundant: 特長量のうち冗長な特徴量の数, n_informative: 特長量のうち有用な特徴量の数
    # n_clusters_per_class: クラスごとのクラスタ(塊)数, class_sep: クラス間の距離, flip_y: ラベルのノイズ割合
    X1, y1 = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2,
        random_state=1, n_clusters_per_class=1, class_sep=2.0, flip_y=0
    )

    # データセット2: Moonsデータセット (非線形)
    n_samples_per_class = 100 # 1クラスあたりのデータ数
    t = np.linspace(0, np.pi, n_samples_per_class) # 0からπまでの範囲を100等分した数値の配列を生成(三日月のカーブを描くための角度として使用)
    X_class0 = np.c_[np.cos(t), np.sin(t)] + np.random.randn(n_samples_per_class, 2) * 0.1 # cos(t)とsin(t)を組み合わせて三日月の形状を作り、ランダムノイズを加える
    y_class0 = np.zeros(n_samples_per_class, dtype=int) # クラス0のラベル(正解データ)をすべて0に設定
    X_class1 = np.c_[1 - np.cos(t), 0.5 - np.sin(t)] + np.random.randn(n_samples_per_class, 2) * 0.3 # 反転した三日月の形状を作り、より大きなランダムノイズを加える
    y_class1 = np.ones(n_samples_per_class, dtype=int) # クラス1のラベルをすべて1に設定
    X2 = np.vstack([X_class0, X_class1]) # クラス0(X_class0)とクラス1(X_class1)の特徴量データを縦に結合して1つの特徴量データセットを作成
    y2 = np.concatenate([y_class0, y_class1]) # クラス0のラベル(y_class0)とクラス1のラベル(y_class1)を結合して1つのラベルデータセットを作成

    # データセット3: より複雑な高次元データ
    n_features = 20
    X3, y3 = make_classification(
        n_samples=500, n_features=n_features, n_informative=8, n_redundant=5,
        n_classes=2, n_clusters_per_class=2, class_sep=0.8, flip_y=0.05,
        random_state=RANDOM_STATE
    )

    datasets = {
        "Linearly Separable": (X1, y1),
        "Non-linear (Moons)": (X2, y2),
        "More Complex Synthetic": (X3, y3)
    }
    return datasets


def get_models_and_display_names() -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    比較対象となる機械学習モデルのインスタンスと、表示用のモデル名を定義。
    Args:
        None
    Returns:
        Tuple[Dict[str, Any], Dict[str, str]]:
            - Dict[str, Any]: キー: モデルの内部名  バリュー: モデルのインスタンス
            - Dict[str, str]: キー: モデルの内部名  バリュー: グラフや表で表示するための日本語併記の名前
    """
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=RANDOM_STATE)
    }

    model_display_names = {
        "Logistic Regression": "Logistic Regression (ロジスティック回帰)",
        "Decision Tree": "Decision Tree (決定木)",
        "Naive Bayes": "Naive Bayes (ナイーブベイズ)",
        "SVM": "SVM (サポートベクターマシン)",
        "Random Forest": "Random Forest (ランダムフォレスト)"
    }
    return models, model_display_names


def _plot_decision_boundary(ax: plt.Axes, model: Any, scaler: StandardScaler, X: np.ndarray, y: np.ndarray) -> None:
    """
    指定されたAxesオブジェクトに、学習済みモデルの決定境界とデータ点をプロットする。
    Args:
        ax (plt.Axes): 描画対象のサブプロット。
        model (Any): 学習済みの分類モデル。
        scaler (StandardScaler): 学習データにフィット済みのStandardScalerオブジェクト。
        X (np.ndarray): 描画する特徴量データ (学習データ)。
        y (np.ndarray): 描画するラベルデータ (学習データ)。
    Returns:
        None
    """
    # 描画範囲をデータから計算
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # 描画範囲を細かいメッシュに分割(後に背景色で塗りつぶすためにグラフ全体を細かなブロック状に分割)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # メッシュの各点も、学習時と同じscalerで標準化してから予測を実行
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    # 決定境界を背景色で塗りつぶす
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    # 学習データを散布図としてプロット
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu, s=20)

def _plot_pca_decision_boundary(
    ax: plt.Axes,
    model: Any,
    X_scaled: np.ndarray,
    y: np.ndarray,
    pca: PCA
) -> None:
    """
    高次元データをPCAで2次元に削減し、その主成分空間上での決定境界とデータ点をプロットする。
    Args:
        ax (plt.Axes): 描画対象のサブプロット。
        model (Any): 学習済みの分類モデル (元の高次元データで学習)。
        X_scaled (np.ndarray): 標準化済みの高次元データ。
        y (np.ndarray): ラベルデータ。
        pca (PCA): データ全体にfit済みのPCAオブジェクト。
    Returns:
        None
    """
    # PCAでデータを2次元に削減
    X_pca = pca.transform(X_scaled)

    # 描画範囲をPCA後のデータから計算
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    # 描画範囲を細かいメッシュに分割
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # 2次元のメッシュ上の各点を、PCAの逆変換で元の高次元空間に戻してから予測
    # これにより、高次元空間での決定境界を2次元に射影して可視化できる
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    # 決定境界とデータ点をプロット
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu, s=15)
    ax.set_xlabel("第1主成分 (PC 1)")
    ax.set_ylabel("第2主成分 (PC 2)")


def evaluate_models_and_visualize_boundaries(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    models: Dict[str, Any],
    model_display_names: Dict[str, str]
) -> pd.DataFrame:
    """
    全てのデータセットとモデルの組み合わせで性能評価（Accuracy）を行い、決定境界をグリッド状にプロットして可視化。
    Args:
        datasets (Dict[str, Tuple[np.ndarray, np.ndarray]]): 評価対象のデータセット辞書。
        models (Dict[str, Any]): 評価対象のモデル辞書。
        model_display_names (Dict[str, str]): 表示用のモデル名辞書。
    Returns:
        pd.DataFrame: データセット、モデル、Accuracyを列に持つ性能評価結果のデータフレーム。
    """
    results = []
    
    # 描画エリアを データセット数 × モデル数 のグリッドで作成
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(25, 15))
    plt.suptitle("モデルの決定境界 (Model Decision Boundaries)", fontsize=20)

    # データセットごとにループ
    for i, (d_name, (X, y)) in enumerate(datasets.items()):
        # データの準備
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        # データを標準化（平均0, 分散1に変換）
        # これにより、距離ベースのモデル(SVMなど)や勾配を使うモデル(ロジスティック回帰など)の性能が安定する
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # モデルごとにループ
        for j, (m_name, model) in enumerate(models.items()):
            ax = axes[i, j]

            # モデルの学習
            # 標準化された学習データでモデルをフィットさせる
            model.fit(X_train_scaled, y_train)

            # 性能評価
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            results.append([d_name, model_display_names[m_name], accuracy])

            # 可視化
            # データが2次元か高次元かでプロット方法を分岐
            if X.shape[1] > 2:
                # 高次元データの場合: PCA(主成分分析)という方法で2次元に落とし込んで可視化
                # 可視化のため、PCAはデータ全体(X, y)にフィットさせる
                full_scaler = StandardScaler()
                X_scaled_full = full_scaler.fit_transform(X)
                pca = PCA(n_components=2)
                pca.fit(X_scaled_full) # 全データでPCAの軸を学習
                
                # プロット関数を呼び出す
                _plot_pca_decision_boundary(ax, model, X_scaled_full, y, pca)
            else:
                # 2次元データの場合: 決定境界を直接プロット
                # ここでは、標準化されたメッシュを生成して予測する必要がある
                _plot_decision_boundary(ax, model, scaler, X_train, y_train)

            # グラフのタイトルとラベルを設定
            if i == 0:  # 最初の行にのみモデル名を表示
                ax.set_title(model_display_names[m_name].replace(" ", "\n"), fontsize=10)
            if j == 0:  # 最初の列にのみデータセット名を表示
                ax.set_ylabel(d_name, fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return pd.DataFrame(results, columns=["Dataset", "Model", "Accuracy"])


def display_performance_table(
    results_df: pd.DataFrame,
    datasets: Dict[str, Any],
    models: Dict[str, Any],
    model_display_names: Dict[str, str]
) -> None:
    """
    性能評価結果のDataFrameを整形し、日本語・英語の2段ヘッダーを持つ表形式でコンソールに出力。
    Args:
        results_df (pd.DataFrame): 評価結果のデータフレーム。
        datasets (Dict[str, Any]): データセット名の順序を定義するために使用。
        models (Dict[str, Any]): モデル名の順序を定義するために使用。
        model_display_names (Dict[str, str]): モデルの表示名を取得するために使用。
    Returns:
        None
    """
    print("\n--- モデル性能比較 (Accuracy) ---")

    # --- データの整形 ---
    # 縦長のDataFrameを、行がデータセット名、列がモデル名の表形式に変換（ピボット）
    pivoted_df = results_df.pivot(index="Dataset", columns="Model", values="Accuracy")
    
    # 表示順を定義
    row_order = list(datasets.keys())
    column_order = [model_display_names[model_name] for model_name in models.keys()]
    
    # 定義した順序にDataFrameの行と列を並び替え
    ordered_df = pivoted_df.reindex(index=row_order)[column_order]

    # --- 2段ヘッダーの作成 (MultiIndex) ---
    # "Model Name (モデル名)" の形式から ( "Model Name", "(モデル名)" ) のタプルリストを作成
    new_columns_tuples = []
    for col_name in ordered_df.columns:
        parts = col_name.replace(")", "").split(" (")
        new_columns_tuples.append((parts[0], f"({parts[1]})"))
    
    # タプルのリストからMultiIndex（階層型インデックス）を作成
    ordered_df.columns = pd.MultiIndex.from_tuples(new_columns_tuples)

    # --- 表の各列の幅を計算 ---
    # 全角文字と半角文字の幅を正しく扱うために `wcswidth` を使用
    english_headers = [col[0] for col in ordered_df.columns]
    japanese_headers = [col[1] for col in ordered_df.columns]
    df_str = ordered_df.applymap(lambda x: f'{x:.3f}') # 全ての値を小数点以下3桁の文字列に

    # ヘッダーの幅を計算
    col_widths = [max(wcswidth(eng), wcswidth(jap)) for eng, jap in zip(english_headers, japanese_headers)]
    
    # データ部分の最大幅と比較して、列幅を更新
    for i, col_name in enumerate(df_str.columns):
        max_data_width = df_str[col_name].apply(wcswidth).max()
        col_widths[i] = max(col_widths[i], max_data_width)

    # インデックス列（データセット名）の幅も計算
    index_width = max(df_str.index.to_series().apply(wcswidth).max(), wcswidth("Dataset"))

    # --- 計算した幅に基づいて表を手動で構築・出力 ---
    def print_row(items: List[str], widths: List[int], index_item: str = "", index_width: int = 0, align: str = '<') -> None:
        """表の1行をフォーマットして出力するヘルパー関数"""
        line = f'{index_item:<{index_width}}'
        for i, item in enumerate(items):
            # パディング（埋めるスペース）を計算
            padding = " " * (widths[i] - wcswidth(item))
            if align == '<': # 左寄せ
                line += f' | {item}{padding}'
            else: # 右寄せ
                line += f' | {padding}{item}'
        print(line)

    # ヘッダー行1 (英語)
    print_row(english_headers, col_widths, "Dataset", index_width)
    # ヘッダー行2 (日本語)
    print_row(japanese_headers, col_widths, "", index_width)

    # 区切り線
    separator = f'{"-" * index_width}-+-{"-+-".join(["-" * w for w in col_widths])}'
    print(separator)

    # データ行
    for index, row in df_str.iterrows():
        print_row(list(row), col_widths, index, index_width, align='>')


def visualize_interpretability(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_display_names: Dict[str, str]
) -> None:
    """
    ロジスティック回帰、決定木、ランダムフォレストの3つのモデルについて、その予測根拠を解釈するための可視化を行う。
    - ロジスティック回帰: 各特徴量の係数（重み）を可視化。
    - 決定木: 学習された分岐ルールをツリー形式で可視化。
    - ランダムフォレスト: 各特徴量の重要度（予測への貢献度）を可視化。
    Args:
        X (np.ndarray): 特徴量データ。
        y (np.ndarray): ラベルデータ。
        feature_names (List[str]): 各特徴量の名前のリスト。
        model_display_names (Dict[str, str]): 表示用のモデル名辞書。
    Returns:
        None
    """
    print("\n--- 解釈性の可視化 (More Complex Synthetic Dataset) ---")

    # データの準備
    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # ロジスティック回帰のためにデータを標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 可視化プロットの準備
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.suptitle("解釈性の可視化 (Interpretability Visualization)", fontsize=16)

    # 各モデルの解釈性を可視化
    # これら３モデルは、内部的な解釈の可視化が可能なモデル(解釈を取り出すための関数が存在)
    # --- ロジスティック回帰: 特徴量の係数 ---
    lr_model = LogisticRegression().fit(X_train_scaled, y_train) # ここでも１から学習を行う
    coefs = pd.Series(lr_model.coef_[0], index=feature_names)
    # 係数の絶対値が大きいトップ10をプロット
    top_coefs = coefs.abs().nlargest(10)
    coefs[top_coefs.index].plot(kind='barh', ax=axes[0])
    axes[0].set_title(f"{model_display_names['Logistic Regression']}:\n特徴量の係数")
    axes[0].set_xlabel("係数 (Coefficient)")

    # --- 決定木: ルールの可視化 ---
    # 決定木やランダムフォレストは標準化が不要なため、元のデータ(X_train)で学習
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE).fit(X_train, y_train)
    plot_tree(dt_model, ax=axes[1], feature_names=feature_names,
              class_names=['class_0', 'class_1'], filled=True, rounded=True, fontsize=8)
    axes[1].set_title(f"{model_display_names['Decision Tree']}:\nルールの可視化")

    # --- ランダムフォレスト: 特徴量の重要度 ---
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    # 重要度が大きいトップ10をプロット
    top_importances = importances.nlargest(10)
    top_importances.sort_values().plot(kind='barh', ax=axes[2]) # 値の小さい順にソートして見やすくする
    axes[2].set_title(f"{model_display_names['Random Forest']}:\n特徴量の重要度")
    axes[2].set_xlabel("重要度 (Feature Importance)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main() -> None:
    """
    分類モデルの比較プログラムのメイン処理。
    Args:
        None
    Returns:
        None
    """
    # データセットの準備
    datasets = create_datasets()

    # モデルの準備
    models, model_display_names = get_models_and_display_names()

    # モデルの性能評価と決定境界の可視化
    results_df = evaluate_models_and_visualize_boundaries(datasets, models, model_display_names)

    # 性能比較表をコンソールに表示
    display_performance_table(results_df, datasets, models, model_display_names)

    # 特定のデータセットを用いたモデルの解釈性の可視化
    X_complex, y_complex = datasets["More Complex Synthetic"]
    feature_names_complex = [f'feature_{i}' for i in range(X_complex.shape[1])]
    visualize_interpretability(X_complex, y_complex, feature_names_complex, model_display_names)


if __name__ == '__main__':
    main()