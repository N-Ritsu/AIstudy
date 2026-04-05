# --- START OF FILE pipeline_tasks/task_02_split_data.py ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pipeline_tasks import settings

def split_and_preprocess_data() -> None:
    """
    データを訓練用とテスト用に分割し、スケーリングして保存する
    Args:
        None
    Returns:
        None
    Raises:
        FileNotFoundError: 入力ファイルが見つからない場合
    """
    print("--- [Task 2] データ分割と前処理を開始 ---")
    df = pd.read_csv(settings.INPUT_DATA_PATH)
    
    X = df.drop(settings.TARGET_COLUMN, axis=1)
    y = df[settings.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE, stratify=y
    )
    
    # スケーリングを追加
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # DataFrameに戻して保存
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    train_df = pd.concat([X_train_df, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)

    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(settings.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(settings.TEST_DATA_PATH, index=False)
    
    print(f"前処理済みの訓練データを '{settings.TRAIN_DATA_PATH}' に保存しました。")
    print(f"前処理済みのテストデータを '{settings.TEST_DATA_PATH}' に保存しました。")

if __name__ == '__main__':
    split_and_preprocess_data()