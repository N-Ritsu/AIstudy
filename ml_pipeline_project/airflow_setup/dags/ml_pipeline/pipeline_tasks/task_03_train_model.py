# --- START OF FILE pipeline_tasks/task_03_train_model.py ---
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from pipeline_tasks import settings

def train_model() -> None:
    """
    訓練データでモデルを学習し、保存する
    Args:
        None
    Returns:
        None
    """
    print("--- [Task 3] モデル学習を開始 ---")
    train_df = pd.read_csv(settings.TRAIN_DATA_PATH)
    
    X_train = train_df.drop(settings.TARGET_COLUMN, axis=1)
    y_train = train_df[settings.TARGET_COLUMN]
    
    model = RandomForestClassifier(**settings.MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    settings.TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.MODEL_PATH)
    
    print(f"学習済みモデルを '{settings.MODEL_PATH}' に保存しました。")

if __name__ == '__main__':
    train_model()