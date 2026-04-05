# --- START OF FILE pipeline_tasks/settings.py ---
from pathlib import Path

# --- ベースパス ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# --- 入力データ ---
INPUT_DATA_PATH = DATA_DIR / "breast_cancer.csv"

# --- 中間成果物（Artifacts）のパス ---
PROCESSED_DATA_DIR = ARTIFACTS_DIR / "processed_data"
TRAINED_MODELS_DIR = ARTIFACTS_DIR / "trained_models"
EVALUATION_RESULTS_DIR = ARTIFACTS_DIR / "evaluation_results"

# 各ステップの出力ファイル
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
MODEL_PATH = TRAINED_MODELS_DIR / "model.joblib"
EVALUATION_PATH = EVALUATION_RESULTS_DIR / "evaluation.json"

# 本番用モデルの保存先（擬似的）
PRODUCTION_MODEL_PATH = TRAINED_MODELS_DIR / "production_model.joblib"

# --- モデルのパラメータ ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = 'target' # 予測対象のカラム名を変更

# RandomForestClassifier用のパラメータ
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}
# デプロイ判断の閾値（悪性(クラス0)のF1スコアがこの値以上ならデプロイ）
DEPLOYMENT_THRESHOLD = 0.90 # より高い目標を設定