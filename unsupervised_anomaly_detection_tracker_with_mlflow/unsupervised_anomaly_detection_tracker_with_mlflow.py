import json
import os
import warnings
from typing import Any, Dict, List
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

OPTIMAL_PARAMS_FILE: str = 'optimal_hyperparameters.json'
CREDITCARD_DATA_FILE: str = 'creditcard.csv'
MINIMUM_RECALL: float = 0.1
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
N_TRIALS: int = 15  # Optunaの試行回数
MODELS_TO_TUNE: List[str] = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    クレジットカードの取引データを読み込み、前処理を行う関数。
    Args:
        filepath (str): データセットのCSVファイルへのパス。
    Returns:
        pd.DataFrame: 前処理済みのデータフレーム。
    Raises:
        FileNotFoundError: 指定されたファイルパスにファイルが存在しない場合に発生。
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"エラー: '{filepath}' が見つかりません。")
        raise

    # 'Time'カラムは学習に直接関係ないため削除
    df = df.drop('Time', axis=1)

    # 金額('Amount')のスケールを他の特徴量と合わせるために標準化
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

    return df


def objective(
    trial: optuna.trial.Trial,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    """
    Optunaによるハイパーパラメータ最適化のための目的関数。
    指定されたモデルのハイパーパラメータを提案し、学習と評価を行う。
    Args:
        trial (optuna.trial.Trial): OptunaのTrialオブジェクト。
        model_name (str): 最適化対象のモデル名。
        X_train (pd.DataFrame): 学習用の特徴量データ。
        y_train (pd.Series): 学習用のラベルデータ。
        X_test (pd.DataFrame): テスト用の特徴量データ。
        y_test (pd.Series): テスト用のラベルデータ。
    Returns:
        float: 最適化の指標となるスコア（通常はF1スコア、ペナルティ時は負の値）。
    Raises:
        ValueError: サポートされていないモデル名が指定された場合に発生。
    """
    with mlflow.start_run(run_name=f"trial_{trial.number}"):

        # モデル名に応じて探索するハイパーパラメータを定義
        if model_name == "IsolationForest":
            params = {
                "contamination": trial.suggest_float("contamination", 0.001, 0.05, log=True), # 検知する異常データを全体の何％に設定するかの閾値の範囲
                "n_estimators": trial.suggest_int("n_estimators", 50, 200), # 決定木の数の範囲
                "random_state": RANDOM_STATE, # 乱数シードを固定して再現性を確保
                "n_jobs": -1, # 計算時に使用するCPUコアの数(-1: 全てのコアを使用)
            }
            model = IsolationForest(**params)
        elif model_name == "OneClassSVM":
            params = {
                "nu": trial.suggest_float("nu", 0.001, 0.05, log=True), # 境界線からのはみだしをどれだけ許容するかの閾値の範囲
                "gamma": trial.suggest_categorical("gamma", ['scale', 'auto']), # 個々の学習データが、境界線に与える影響の強さ
                "kernel": "rbf", # データの正常・異常の境界線の引き方(柔軟タイプ)
            }
            model = OneClassSVM(**params)
        elif model_name == "LocalOutlierFactor":
            params = {
                "contamination": trial.suggest_float("contamination", 0.001, 0.05, log=True),
                "n_neighbors": trial.suggest_int("n_neighbors", 10, 50), # どのぐらいの数の隣接するデータポイントを、局所的なデータと見なすかの範囲
                "novelty": True,  # predictメソッドを使うためにTrueに設定
                "n_jobs": -1,
            }
            model = LocalOutlierFactor(**params)
        else:
            raise ValueError(f"サポートされていないモデル名です: {model_name}")

        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params)

        # OneClassSVMとLocalOutlierFactorは正常系データ（Class=0）のみで学習
        if model_name in ["OneClassSVM", "LocalOutlierFactor"]:
            model.fit(X_train[y_train == 0]) # 正解ラベル(y_train)が0のデータのみで学習
        else:
            model.fit(X_train)

        # テストデータで予測
        y_pred_model = model.predict(X_test) # 学習済みのモデルで、未知のデータ(X_test)に対し予測を行う
        y_pred = [1 if pred == -1 else 0 for pred in y_pred_model] # 異常検知モデルは異常を-1, 正常を1と予測するため、ラベル（0, 1）に変換

        # y_test(正解)と、y_pred(予測)を比較して、classification_reportで性能評価を行い、F1スコアを最適化の指標とする
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # 1のキーが存在しない場合(不正利用が１件の見つけられなかった)でもエラーにならず、デフォルト値を返す
        fraud_metrics = report_dict.get('1', {})

        # fraud_metrics辞書から、再現率(Recall)、適合率(Precision)、F1スコアを取得。存在しない場合は0.0を返す
        reported_recall = fraud_metrics.get('recall', 0.0)
        reported_precision = fraud_metrics.get('precision', 0.0)
        reported_f1 = fraud_metrics.get('f1-score', 0.0)

        # OptunaのTrialオブジェクトにユーザー属性として各指標を保存し、後から参照できるようにする
        trial.set_user_attr("fraud_precision", reported_precision)
        trial.set_user_attr("fraud_recall", reported_recall)

        # MLflowに各評価指標を記録
        mlflow.log_metric("fraud_f1_score", reported_f1)
        mlflow.log_metric("fraud_precision", reported_precision)
        mlflow.log_metric("fraud_recall", reported_recall)

        # 不正検知の再現率（Recall）が最低基準を下回った場合、その試行は失敗と見なす
        if reported_recall < MINIMUM_RECALL:
            # F1スコアの代わりにペナルティ値を返すことで、Optunaに悪い結果だと学習させる
            return - (1.0 - reported_recall)

        return reported_f1


def tune_and_save_hyperparameters(
    models: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    指定されたモデルリストに対してハイパーパラメータチューニングを行い、結果をJSONファイルに保存する。
    Args:
        models (List[str]): チューニング対象のモデル名のリスト。
        X_train (pd.DataFrame): 学習用の特徴量データ。
        y_train (pd.Series): 学習用のラベルデータ。
        X_test (pd.DataFrame): テスト用の特徴量データ。
        y_test (pd.Series): テスト用のラベルデータ。
    Returns:
        None
    """
    print(f"'{OPTIMAL_PARAMS_FILE}' が見つかりません。ハイパーパラメータの探索を開始します...")

    final_results: Dict[str, Dict[str, Any]] = {}

    for model_name in models:
        print(f"\n{'='*20}\n[{model_name}] のチューニングを開始します...\n{'='*20}")

        # MLflowの実験を設定（モデルごとに実験を分ける）
        mlflow.set_experiment(f"{model_name}_HyperOpt_Tuning")

        # OptunaのStudyオブジェクトを作成し、F1スコアが最大になるように最適化を実行
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, model_name, X_train, y_train, X_test, y_test),
            n_trials=N_TRIALS
        )

        best_trial = study.best_trial
        print(f"\n--- [{model_name}] チューニング完了・最終結果 ---")
        print(f"最高のF1スコア: {best_trial.value:.4f}")
        print(f"  その時の検知成功率 (Recall): {best_trial.user_attrs['fraud_recall']:.4f} （不正利用の検知に成功した確率）")
        print(f"  その時の適合率 (Precision): {best_trial.user_attrs['fraud_precision']:.4f} （不正だと警告したうち、本当に不正だった確率）")
        print("最高のハイパーパラメータ:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")

        # 最終結果を辞書に格納
        final_results[model_name] = {
            'best_f1_score': best_trial.value,
            'best_recall': best_trial.user_attrs['fraud_recall'],
            'best_precision': best_trial.user_attrs['fraud_precision'],
            'params': best_trial.params
        }

    # 全モデルの結果をJSONファイルに書き出す
    with open(OPTIMAL_PARAMS_FILE, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\n全てのチューニングが完了しました。最適なパラメータを '{OPTIMAL_PARAMS_FILE}' に保存しました。")


def load_and_display_results(filepath: str) -> None:
    """
    保存された最適なハイパーパラメータを読み込み、結果を整形して表示する。
    Args:
        filepath (str): 最適なハイパーパラメータが保存されたJSONファイルへのパス。
    Returns:
        None
    """
    print(f"'{filepath}' を発見しました。保存済みの最適パラメータを読み込みます。")
    with open(filepath, 'r') as f:
        best_params_all_models = json.load(f)

    print("\n--- 各モデルの最適なハイパーパラメータ ---")
    for model_name, data in best_params_all_models.items():
        print(f"\nモデル: {model_name}")
        print(f"  最高のF1スコア: {data['best_f1_score']:.4f}")
        print(f"    その時の検知成功率 (Recall): {data['best_recall']:.4f} （誤検知を恐れず、どれだけ見つけられたか）")
        print(f"    その時の適合率 (Precision): {data['best_precision']:.4f}  （どれだけ警告の精度が高いか）")
        print("  最高のパラメータ:")
        for key, value in data['params'].items():
            print(f"    {key}: {value}")
    print("\nチューニングをスキップしました。")


def main() -> None:
    """
    複数アルゴリズムによるクレジットカード不正利用検知とMLflowでの性能追跡プログラムのメイン処理。
    Args:
        None
    Returns:
        None
    """
    # 既存の最適化済みファイルがあるか確認
    if os.path.exists(OPTIMAL_PARAMS_FILE):
        load_and_display_results(OPTIMAL_PARAMS_FILE)
    else:
        try:
            # データの読み込みと前処理
            full_df = load_and_preprocess_data(CREDITCARD_DATA_FILE)
        except FileNotFoundError:
            # データファイルがない場合は処理を中断
            return

        # 特徴量(X)とターゲット(y)に分割
        X_full = full_df.drop('Class', axis=1)
        y_full = full_df['Class']

        # データを学習用とテスト用に分割
        # stratify=y_fullで、分割後の各セットのクラス比率が元データと同じになるようにする
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_full
        )

        # ハイパーパラメータのチューニングと結果の保存を実行
        tune_and_save_hyperparameters(
            models=MODELS_TO_TUNE,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )


if __name__ == "__main__":
    # scikit-learnなどが出力する警告を非表示に設定
    warnings.filterwarnings('ignore')
    main()