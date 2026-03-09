import json
import os
import warnings
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from unsupervised_anomaly_detection_tracker_with_mlflow import (
    load_and_preprocess_data,
    CREDITCARD_DATA_FILE,
    RANDOM_STATE,
    TEST_SIZE
)

OPTIMAL_PARAMS_FILE: str = 'optimal_hyperparameters.json'
ENSEMBLE_THRESHOLDS: List[int] = [1, 2, 3] # アンサンブル評価で試す閾値
# モデル名をクラスにマッピングする辞書
MODEL_CLASSES: Dict[str, Any] = {
    "IsolationForest": IsolationForest,
    "OneClassSVM": OneClassSVM,
    "LocalOutlierFactor": LocalOutlierFactor
}

def load_optimal_params(filepath: str) -> Dict[str, Any]:
    """
    保存された最適なハイパーパラメータをJSONファイルから読み込む。
    Args:
        filepath (str): 最適なハイパーパラメータが保存されたJSONファイルへのパス。
    Returns:
        Dict[str, Any]: モデルごとの最適なハイパーパラメータを格納した辞書。
    Raises:
        FileNotFoundError: 指定されたファイルパスにファイルが存在しない場合に発生。
    """
    if not os.path.exists(filepath):
        print(f"エラー: '{filepath}' が見つかりません。")
        print("先に、ハイパーパラメータを探索するスクリプトを実行して、最適パラメータファイルを生成してください。")
        raise FileNotFoundError
        
    with open(filepath, 'r') as f:
        return json.load(f)

def train_and_predict_models(
    best_params: Dict[str, Any], 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame
) -> List[np.ndarray]:
    """
    最適なハイパーパラメータを使い、各モデルを再学習させて予測結果を返す。
    Args:
        best_params (Dict[str, Any]): モデルごとの最適なハイパーパラメータ。
        X_train (pd.DataFrame): 学習用の特徴量データ。
        y_train (pd.Series): 学習用のラベルデータ。
        X_test (pd.DataFrame): テスト用の特徴量データ。
    Returns:
        List[np.ndarray]: 各モデルの予測結果（0または1）を格納したリスト。
    """
    predictions: List[np.ndarray] = []
    
    print("--- 各モデルを最高のパラメータで再学習・予測します ---")
    for model_name, data in best_params.items():
        print(f"モデル: {model_name}...")
        
        current_params = data['params'].copy() # 元の辞書を変更しないようにコピー
        if model_name == "LocalOutlierFactor":
            current_params['novelty'] = True
            current_params['n_jobs'] = -1
            
        model = MODEL_CLASSES[model_name](**current_params)

        # OneClassSVMとLocalOutlierFactorは正常系データ（Class=0）のみで学習
        if model_name in ["OneClassSVM", "LocalOutlierFactor"]:
            model.fit(X_train[y_train == 0])
        else:
            model.fit(X_train)
            
        # 予測結果を-1/1から0/1に変換してリストに追加
        y_pred_model = model.predict(X_test)
        y_pred = np.array([1 if pred == -1 else 0 for pred in y_pred_model])
        predictions.append(y_pred)
        
    return predictions

def evaluate_ensemble(predictions: np.ndarray, y_test: pd.Series, thresholds: List[int]) -> None:
    """
    アンサンブル学習の結果を、複数の閾値で評価して表示する。
    Args:
        predictions (np.ndarray): 各モデルの予測結果を列に持つNumpy配列 (サンプル数 x モデル数)。
        y_test (pd.Series): テストデータの正解ラベル。
        thresholds (List[int]): 評価に使用する閾値のリスト。
    Returns:
        None
    """
    for threshold in thresholds:
        # 予測の合計が閾値以上なら不正と判断
        # (例: threshold=2なら、3つのモデルのうち2つ以上が「不正」と予測した場合に最終判断を「不正」とする)
        final_preds = (np.sum(predictions, axis=1) >= threshold).astype(int)
        
        report = classification_report(y_test, final_preds, output_dict=True, zero_division=0)
        fraud_metrics = report.get('1', {})
        
        print(f"\n--- 閾値: {threshold} (不正と判断するのに必要なモデルの数) ---")
        if np.sum(final_preds) == 0:
            print("不正利用を1件も検知できませんでした。")
        else:
            print(f"  検知成功率 (Recall): {fraud_metrics.get('recall', 0.0):.4f}")
            print(f"  適合率 (Precision): {fraud_metrics.get('precision', 0.0):.4f}")
            print(f"  総合評価 (F1-Score): {fraud_metrics.get('f1-score', 0.0):.4f}")

def display_individual_model_performance(best_params: Dict[str, Any]) -> None:
    """
    比較参考のために、個々のモデルのチューニング後の最高性能を表示する。
    Args:
        best_params (Dict[str, Any]): モデルごとの最適なハイパーパラメータと性能スコア。
    Returns:
        None
    """
    print("\n" + "="*50)
    print("参考：個々のモデルのチューニング後の最高性能")
    print("="*50)
    for model_name, data in best_params.items():
        print(f"\nモデル: {model_name}")
        print(f"  検知成功率 (Recall): {data['best_recall']:.4f}")
        print(f"  適合率 (Precision): {data['best_precision']:.4f}")
        print(f"  総合評価 (F1-Score): {data['best_f1_score']:.4f}")


def main() -> None:
    """
    各モデルを最適パラメータで再学習させ、アンサンブル評価を行うメイン関数。
    Args:
        None
    Returns: 
        None
    """
    try:
        # 最適パラメータの読み込み
        best_params_all_models = load_optimal_params(OPTIMAL_PARAMS_FILE)
        # データの読み込みと前処理
        X_full = load_and_preprocess_data(CREDITCARD_DATA_FILE)
    except FileNotFoundError:
        # 必要なファイルがない場合は処理を終了
        return

    # 特徴量(X)とターゲット(y)に分割
    y_full = X_full.pop('Class')
    
    # データを学習用とテスト用に分割
    # stratify=y_fullで、分割後の各セットのクラス比率が元データと同じになるようにする
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_full
    )

    # 各モデルを最高のパラメータで学習させ、予測結果を取得
    predictions_list = train_and_predict_models(
        best_params=best_params_all_models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test
    )
    
    # 予測結果リストをNumpy配列に変換し、アンサンブル評価に適した形 (サンプル数 x モデル数) に転置
    predictions_array = np.array(predictions_list).T
    
    # アンサンブル評価の実行
    evaluate_ensemble(predictions_array, y_test, ENSEMBLE_THRESHOLDS)
    
    # 参考として個々のモデルの性能を表示
    display_individual_model_performance(best_params_all_models)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()