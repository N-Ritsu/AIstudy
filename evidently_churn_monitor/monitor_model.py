# monitor_model.py
import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

REFERENCE_DATA_PATH = "monitoring_artifacts/reference_data.csv"
PRODUCTION_LOGS_PATH = "monitoring_artifacts/production_logs.csv"
DRIFT_REPORT_PATH = "monitoring_artifacts/model_drift_report.html"


def main() -> None:
    """
    本番ログとベースラインデータを比較し、データドリフトレポートを生成する
    Args:
        None
    Returns:
        None
    """
    print("--- データドリフトの監視とレポート生成 ---")

    try:
        reference_df = pd.read_csv(REFERENCE_DATA_PATH)
        production_df = pd.read_csv(PRODUCTION_LOGS_PATH)
    except FileNotFoundError as e:
        print(f"エラー: データファイルが見つかりません: {e.filename}")
        print("先に 'train_churn_model.py' と 'simulate_traffic.py' を実行してください。")
        return
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return

    print("ドリフトレポートを生成しています...")

    # 監視対象の特徴量列（prediction_proba / prediction は除外）
    feature_columns = ['tenure', 'monthly_charges', 'total_charges', 'contract']

    # DataDefinitionで列の型を指定
    data_definition = DataDefinition(
        numerical_columns=['tenure', 'monthly_charges', 'total_charges'],
        categorical_columns=['contract']
    )

    # Datasetオブジェクトを作成
    reference_dataset = Dataset.from_pandas(
        reference_df[feature_columns],
        data_definition=data_definition
    )

    # production_dfには予測結果も含まれているが、ドリフトレポートでは特徴量のドリフトを分析するため、特徴量列のみをDatasetに渡す
    production_dataset = Dataset.from_pandas(
        production_df[feature_columns],
        data_definition=data_definition
    )

    # レポート作成
    data_drift_report = Report([DataDriftPreset()])

    # 実行して結果オブジェクトを受け取る
    result = data_drift_report.run(
        reference_data=reference_dataset,
        current_data=production_dataset
    )

    # resultに対して save_html を呼ぶ
    result.save_html(DRIFT_REPORT_PATH)

    print("-" * 50)
    print("データドリフトレポートの生成が完了しました。")
    print(f"ブラウザで '{DRIFT_REPORT_PATH}' を開いて確認してください。")
    print("-" * 50)


if __name__ == '__main__':
    main()