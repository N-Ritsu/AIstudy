#!/bin/bash

# パイプラインをローカルで実行するためのスクリプト
set -e

echo "===== 乳がん診断MLパイプラインのローカル実行を開始します ====="

# 成果物ディレクトリをクリーンアップ
echo "\n--- 成果物ディレクトリをクリーンアップ ---"
rm -rf artifacts/
mkdir -p artifacts/processed_data artifacts/trained_models artifacts/evaluation_results

# 各タスクを順番に実行
python -m pipeline_tasks.task_01_validate_data
# スクリプト名を変更
python -m pipeline_tasks.task_02_split_data
python -m pipeline_tasks.task_03_train_model
python -m pipeline_tasks.task_04_evaluate_model
python -m pipeline_tasks.task_05_deploy_model

echo "\n===== MLパイプラインのローカル実行が正常に完了しました ====="