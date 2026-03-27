import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import japanize_matplotlib
from typing import Union

# 乱数シード
RANDOM_STATE = 42


def create_imbalanced_dataset(n_samples: int = 1000, weights: list[float] = [0.95, 0.05]) -> tuple[np.ndarray, np.ndarray]:
    """
    指定されたクラス比率で不均衡なデータセットを生成する。
    Args:
        n_samples (int): 生成する総サンプル数。
        weights (list[float]): 各クラスの比率。合計が1.0になるように設定。
    Returns:
        Tuple[np.ndarray, np.ndarray]
          - X (np.ndarray): 生成された特徴量データ。
          - y (np.ndarray): 生成されたラベルデータ。
    """
    print("--- 不均衡データセットの生成 ---")
    # make_classification: 指定されたクラス比率でランダムなデータセットを生成する関数
    X, y = make_classification(
        n_samples=n_samples, # 生成するサンプル数
        n_features=2, # 各データの特徴量の数
        n_redundant=0, # 冗長な特徴量の数
        n_informative=2, # 有用な特徴量の数
        n_clusters_per_class=1, # 各クラスごとにいくつのクラスタ(塊)に分割するか
        weights=weights, # クラスの比率
        flip_y=0, # ラベルを反転させる割合(ノイズ)
        random_state=RANDOM_STATE # 乱数シード
    )
    print(f"データ生成完了。形状: {X.shape}")
    print(f"クラス分布 (0/1): {np.bincount(y)}")
    print("-" * 30 + "\n")
    return X, y


def plot_data_distributions(
    original_data: tuple[np.ndarray, np.ndarray], 
    smote_data: tuple[np.ndarray, np.ndarray], 
    rus_data: tuple[np.ndarray, np.ndarray]
) -> None:
    """
    サンプリング適用前後のデータ分布を可視化し、画像として保存する。
    Args:
        original_data (Tuple)
          - X_train (np.ndarray): サンプリング前の特徴量データ。
          - y_train (np.ndarray): サンプリング前のラベルデータ。
        smote_data (Tuple)
          - X_smote (np.ndarray): SMOTE適用後の特徴量データ。
          - y_smote (np.ndarray): SMOTE適用後のラベルデータ。
        rus_data (Tuple)
          - X_rus (np.ndarray): RandomUnderSampler適用後の特徴量データ。
          - y_rus (np.ndarray): RandomUnderSampler適用後のラベルデータ。
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("サンプリング手法による学習データの分布変化", fontsize=16)

    def plot_scatter(ax, X, y, title):
        ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="クラス 0 (多数派)", alpha=0.5)
        ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="クラス 1 (少数派)", alpha=0.7, marker='x')
        ax.set_title(f"{title}\nクラス分布 (0/1): {np.bincount(y)}")
        ax.legend()
        ax.grid(True)

    plot_scatter(axes[0], original_data[0], original_data[1], "1. サンプリングなし (Original)")
    plot_scatter(axes[1], smote_data[0], smote_data[1], "2. オーバーサンプリング (SMOTE)")
    plot_scatter(axes[2], rus_data[0], rus_data[1], "3. アンダーサンプリング (RandomUnderSampler)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # --- 変更点: グラフをファイルに保存 ---
    output_filename = "data_distribution_comparison.png"
    plt.savefig(output_filename)
    print(f"グラフを '{output_filename}' として保存しました。")


def train_and_evaluate(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    model_name: str
) -> dict[str, Union[str, np.ndarray]]:
    """
    指定された学習データでモデルを学習し、テストデータで評価する。
    Args:
        X_train (np.ndarray): 学習データの特徴量。
        y_train (np.ndarray): 学習データのラベル。
        X_test (np.ndarray): テストデータの特徴量。
        y_test (np.ndarray): テストデータのラベル。
        model_name (str): 評価結果の表示に使う名前。
    Returns:
        dict[str, Union[str, np.ndarray]]
          - str: データ名。
          - Union[str, np.ndarray]]
            - name (str): モデル名。
            - y_true (np.ndarray): テストデータの正解ラベル。
            - y_pred (np.ndarray): テストデータの予測ラベル。
            - y_pred_proba (np.ndarray): テストデータの陽性クラスの予測確率（P-R曲線描画用）。
    """
    print(f"--- モデル学習・評価: {model_name} ---")
    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # テストデータで予測
    y_pred = model.predict(X_test)
    # predict_proba: そのデータが各クラスに属する確率を返す関数(例: クラス0の確率: 90%, クラス1の確率: 10%)。[:, 1]で陽性クラスの確率を取得
    y_pred_proba = model.predict_proba(X_test)[:, 1] # P-R曲線描画用に陽性の確率を取得

    # 性能評価
    print("分類レポート:")
    print(classification_report(y_test, y_pred, target_names=["クラス 0", "クラス 1"]))

    return {
        "name": model_name,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    }


def visualize_evaluation_results(results: list[dict]) -> None:
    """
    混同行列と適合率-再現率曲線（P-R曲線）を可視化し、画像として保存する。
    Args:
        results (list[dict]): train_and_evaluateから返された評価結果の辞書リスト。
    Returns:
        None
    """
    # --- 混同行列の可視化 ---
    fig_cm, axes_cm = plt.subplots(1, len(results), figsize=(21, 6))
    fig_cm.suptitle("混同行列の比較", fontsize=16)

    for i, result in enumerate(results):
        # 正解ラベルと予測ラベルの2*2の行列を作成し、ヒートマップで表示する
        cm = confusion_matrix(result["y_true"], result["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes_cm[i],
                    xticklabels=["予測: 0", "予測: 1"], yticklabels=["正解: 0", "正解: 1"])
        axes_cm[i].set_title(result["name"])
        axes_cm[i].set_ylabel("正解ラベル")
        axes_cm[i].set_xlabel("予測ラベル")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # --- 混同行列のグラフをファイルに保存 ---
    cm_filename = "confusion_matrix_comparison.png"
    plt.savefig(cm_filename)
    print(f"グラフを '{cm_filename}' として保存しました。")

    # --- 適合率-再現率曲線（P-R曲線）の可視化 ---
    plt.figure(figsize=(10, 8))
    plt.title("適合率-再現率曲線 (Precision-Recall Curve) の比較", fontsize=16)

    for result in results:
        # 正解ラベルと陽性であると判断した確率から、適合率と再現率を取得し、P-R曲線を描画する
        precision, recall, _ = precision_recall_curve(result["y_true"], result["y_pred_proba"])
        auc_pr = auc(recall, precision) # AUC-PR(曲線の下の面積、１に近いほど性能が良い)を計算する
        plt.plot(recall, precision, lw=2, label=f'{result["name"]} (AUC-PR = {auc_pr:.2f})')

    plt.xlabel("再現率 (Recall)")
    plt.ylabel("適合率 (Precision)")
    plt.legend(loc="best")
    plt.grid(True)

    # --- P-R曲線のグラフをファイルに保存 ---
    pr_curve_filename = "precision_recall_curve_comparison.png"
    plt.savefig(pr_curve_filename)
    print(f"グラフを '{pr_curve_filename}' として保存しました。")


def main() -> None:
    """
    SMOTEを用いたオーバーサンプリングとアンダーサンプリングの比較プログラムのメイン処理
    Args:
        None
    Returns:
        None
    """
    # --- 不均衡データセットを生成 ---
    X, y = create_imbalanced_dataset()

    # --- データを学習用とテスト用に分割 ---
    # サンプリングは学習データにのみ適用するため、先に分割する
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    print("--- 学習データとテストデータに分割 ---")
    print(f"学習データ クラス分布 (0/1): {np.bincount(y_train)}")
    print(f"テストデータ クラス分布 (0/1): {np.bincount(y_test)}")
    print("-" * 30 + "\n")

    # --- 各サンプリング手法を学習データに適用 ---
    # オーバーサンプリング (SMOTE)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # アンダーサンプリング (RandomUnderSampler)
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    
    # --- データ分布の可視化 ---
    plot_data_distributions(
        (X_train, y_train),
        (X_train_smote, y_train_smote),
        (X_train_rus, y_train_rus)
    )

    # --- 各データセットでモデルを学習・評価 ---
    results = []
    
    # ケース1: サンプリングなし
    results.append(train_and_evaluate(X_train, y_train, X_test, y_test, "サンプリングなし (Original)"))
    
    # ケース2: オーバーサンプリング (SMOTE)
    results.append(train_and_evaluate(X_train_smote, y_train_smote, X_test, y_test, "オーバーサンプリング (SMOTE)"))

    # ケース3: アンダーサンプリング
    results.append(train_and_evaluate(X_train_rus, y_train_rus, X_test, y_test, "アンダーサンプリング (RUS)"))

    # --- 評価結果の可視化 ---
    visualize_evaluation_results(results)

    print("\n全ての処理が完了しました。")

if __name__ == "__main__":
    main()