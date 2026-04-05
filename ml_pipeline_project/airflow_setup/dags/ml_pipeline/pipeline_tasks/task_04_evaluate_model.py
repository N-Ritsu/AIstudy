# --- START OF FILE pipeline_tasks/task_04_evaluate_model.py ---
import pandas as pd
import joblib
from sklearn.metrics import classification_report
import json
from pipeline_tasks import settings

def evaluate_model() -> None:
    """
    テストデータでモデルを評価し、結果を保存する
    Args:
        None
    Returns:
        None
    """
    print("--- [Task 4] モデル評価を開始 ---")
    model = joblib.load(settings.MODEL_PATH)
    test_df = pd.read_csv(settings.TEST_DATA_PATH)
    
    X_test = test_df.drop(settings.TARGET_COLUMN, axis=1)
    y_test = test_df[settings.TARGET_COLUMN]
    
    y_pred = model.predict(X_test) # 教師ありモデルなので予測値の変換は不要
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print("分類レポート:")
    print(classification_report(y_test, y_pred, target_names=['悪性 (0)', '良性 (1)']))

    settings.EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.EVALUATION_PATH, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"評価結果を '{settings.EVALUATION_PATH}' に保存しました。")

if __name__ == '__main__':
    evaluate_model()