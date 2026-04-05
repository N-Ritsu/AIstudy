# --- START OF FILE pipeline_tasks/task_01_validate_data.py ---
import pandas as pd
from pipeline_tasks import settings

def validate_data() -> None:
    """
    入力データの存在と基本構造を検証する
    Args:
        None
    Returns:
        None
    Raises:
        FileNotFoundError: 入力ファイルが見つからない場合
        ValueError: 必要なカラムが存在しない場合
    """
    print("--- [Task 1] データ検証を開始 ---")
    if not settings.INPUT_DATA_PATH.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {settings.INPUT_DATA_PATH}")

    df = pd.read_csv(settings.INPUT_DATA_PATH)
    # 最低限必要なカラムを 'target' に変更
    required_cols = {settings.TARGET_COLUMN}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"入力データに必要なカラム {required_cols} が含まれていません。")
    
    print("データ検証が完了しました。")

if __name__ == '__main__':
    validate_data()