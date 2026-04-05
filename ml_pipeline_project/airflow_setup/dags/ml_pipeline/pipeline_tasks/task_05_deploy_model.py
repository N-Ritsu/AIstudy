# --- START OF FILE pipeline_tasks/task_05_deploy_model.py ---
import json
import shutil
from pipeline_tasks import settings

def deploy_model() -> None:
    """
    評価結果に基づき、モデルを本番用としてデプロイするか判断する
    Args:
        None
    Returns:
        None
    """
    print("--- [Task 5] モデルデプロイ判断を開始 ---")
    with open(settings.EVALUATION_PATH, 'r') as f:
        report = json.load(f)
        
    # 今回重要なのは「悪性(0)」の見逃しを防ぐことなので、クラス'0'のF1スコアを基準にする
    f1_score = report.get('0', {}).get('f1-score', 0.0)
    
    print(f"悪性クラス(0)のF1スコア: {f1_score:.4f}")
    print(f"デプロイ判断の閾値: {settings.DEPLOYMENT_THRESHOLD}")
    
    if f1_score >= settings.DEPLOYMENT_THRESHOLD:
        print("F1スコアが閾値を超えたため、モデルを本番用としてデプロイします。")
        shutil.copyfile(settings.MODEL_PATH, settings.PRODUCTION_MODEL_PATH)
        print(f"モデルを '{settings.PRODUCTION_MODEL_PATH}' にコピーしました。")
    else:
        print("F1スコアが閾値に達しなかったため、デプロイは見送ります。")

if __name__ == '__main__':
    deploy_model()