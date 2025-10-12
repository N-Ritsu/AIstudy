import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from typing import Tuple 

SCORE_THRESHOLD_20_MINUTES = 2000
SCORE_THRESHOLD_60_MINUTES = 2500
SCORE_THRESHOLD_90_MINUTES = 2750
SCORE_THRESHOLD_HIGH = 3000

def generate_game_logs(num_players=1000) -> pd.DataFrame:
  """
  ゲームログデータをランダムにより擬似的に生成
  Args:
    num_players (int): プレイヤーの合計人数
  Returns:
    pd.DataFrame: プレイヤーID, スコア, プレイ時間を含むデータフレーム
  """
  # スコアが低い～普通のプレイヤーのデータ(それぞれ990個)
  # loc = 平均値、scale = 標準偏差、size = 人数
  normal_scores = np.random.normal(loc=1500, scale=300, size=num_players - 10)
  normal_playtime = np.random.normal(loc=120, scale=40, size=num_players - 10)
  # スコアが異常に高いプレイヤーのデータ(それぞれ10個)
  cheater_scores = np.random.normal(loc=3500, scale=500, size=10)
  cheater_playtime = np.random.normal(loc=10, scale=5, size=10)
  # データ配列を結合
  # normal_scores, cheater_scoresの順番をあえて変えることで、スコアは普通なのにプレイ時間が異常に短い、プレイ時間は普通なのにスコアが異常に高いの２種のプレイヤーを10人づつ用意
  scores = np.concatenate([normal_scores, cheater_scores])
  playtime = np.concatenate([cheater_playtime, normal_playtime])
  # 負の値を0にする
  scores[scores < 0] = 0
  playtime[playtime < 0] = 0
  
  # データフレームを作成
  df = pd.DataFrame({
    "player_id": range(1, num_players + 1),
    "score": scores.astype(int),
    "playtime_minutes": playtime.astype(int)
  })
    
  # データをシャッフルして、チーターを散らばせる
  # .sample(frac=1):全てをランダムに並び替え、.reset_index:行番号を0からふり直し、drop=True:古い行番号のデータを破棄
  return df.sample(frac=1).reset_index(drop=True)


def rule_based_detector(df: pd.DataFrame) -> pd.DataFrame:
  """
  簡単なルールに基づいた閾値設定による不正プレイヤー検出
  Args:
    df (pd.DataFrame): プレイヤーID, スコア, プレイ時間を含むデータフレーム
  Returns:
    pd.DataFrame: 簡単なルールに基づいた閾値設定によって検出されたプレイヤーのデータフレーム
  """
  # ルール1: プレイ時間が20分未満で、スコアが2000点以上
  rule1 = (df["playtime_minutes"] < 20) & (df["score"] > SCORE_THRESHOLD_20_MINUTES)
  # ルール2: プレイ時間が60分未満で、スコアが2500点以上
  rule2 = (df["playtime_minutes"] < 60) & (df["score"] > SCORE_THRESHOLD_60_MINUTES)
  # ルール3: プレイ時間が90分未満で、スコアが2750点以上
  rule3 = (df["playtime_minutes"] < 90) & (df["score"] > SCORE_THRESHOLD_90_MINUTES)
  # ルール4: スコアが3000点以上
  rule4 = df["score"] > SCORE_THRESHOLD_HIGH
  # ルール5: 未プレイにも関わらずスコアが0点ではない
  rule5 = (df["playtime_minutes"] == 0) & (df["score"] != 0)
  
  # ルールのいずれかに当てはまるプレイヤーを抽出
  suspicious_players = df[rule1 | rule2 | rule3 | rule4 |rule5]
  indices = suspicious_players.index
  df.loc[indices, "anomaly_prediction"] = -1
  return suspicious_players


def ai_anomaly_detector(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  特徴量エンジニアリングによる異常検知
  Args:
    df (pd.DataFrame): プレイヤーID, スコア, プレイ時間を含むデータフレーム
  Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: 
      - 最終的に不正と判断されたプレイヤーのデータフレーム
      - 全プレイヤーデータに予測結果が付与されたデータフレーム
  """
  # 特徴量score_per_minuteを作成
  # 0で割るのを防ぐため、プレイ時間が0の場合は1に置き換える
  df["score_per_minute"] = df["score"] / df["playtime_minutes"].replace(0, 1)
  # AIモデルに学習させるデータを、合計3つの特徴量にする
  features = df[["score", "playtime_minutes", "score_per_minute"]]
  
  # AIモデルの初期化と学習
  # random_stateに整数を入れると、AIによる予測のランダム性が固定され、毎回結果が変わることを防ぐ
  model = IsolationForest(contamination=0.03, random_state=1) 
  # AIの予測結果をdfの新キーワード"anomaly_prediction"に代入
  df["anomaly_prediction"] = model.fit_predict(features)
  anomalies = df[df["anomaly_prediction"] == -1].copy()

  # 正常なプレイヤーをはじく最終判断
  # 正常プレイヤーの平均時間効率を計算
  mean_efficiency = df["score_per_minute"].mean()
  # 時間効率が平均以上のプレイヤーにのみ、チーターかどうか判別する
  rule1 = anomalies["score_per_minute"] > mean_efficiency
  # スコアが2000点以上のプレイヤーにのみ、チーターかどうか判別する
  rule2 = anomalies["score"]>2000
  suspicious_players = anomalies[rule1 & rule2]
  return suspicious_players, df


def make_fig(normal_player: pd.DataFrame, suspicious_players_by_rule: pd.DataFrame, suspicious_players_by_ai: pd.DataFrame) -> str:
  """
  全プレイヤーデータが、一般プレイヤーとチーターで色分けされた状態でプロットされたグラフを作成。
  チーターの中でも、ルールベースとAIのそれぞれの判別においてプロットを行う
  グラフはpngファイルに保存される
  Args:
    game_data_with_prediction, suspicious_players_by_rule, suspicious_players_by_ai(Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:): 
      - 全プレイヤーデータに予測結果が付与されたデータフレーム
      - ルールベースによって、チーターと判断されたプレイヤーのデータフレーム
      - AIによる判別によって、チーターと判断されたプレイヤーのデータフレーム
  Returns:
    str: グラフの保存先ファイル名
  """
  plt.figure(figsize=(12, 8))
  # 正常と判定されたプレイヤーをプロット (灰色)
  # game_data_with_prediction["anomaly_prediction"] == 1:正常なプレイヤー要素
  # game_data_with_prediction[game_data_with_prediction["anomaly_prediction"] == 1]["score"]:game_data_with_predictionの中で、正常なプレイヤー要素のscoreの値を入手
  plt.scatter(normal_player["score"], normal_player["playtime_minutes"],
              c="lightgray", label="Normal Players")
  # ルールベースで異常と判定されたプレイヤーをプロット(青色の×)
  plt.scatter(suspicious_players_by_rule["score"], suspicious_players_by_rule["playtime_minutes"],
              c='blue', marker='x', s=100, label="Suspicious Players (by Rule)")
  # AIで異常と判定されたプレイヤーをプロット(赤色の○)
  plt.scatter(suspicious_players_by_ai['score'], suspicious_players_by_ai['playtime_minutes'],
                c='red', label='Suspicious Players (by AI)')
  plt.title("Game Logs Anomaly Detection")
  plt.xlabel("Score")
  plt.ylabel("Playtime (minutes)")
  plt.legend() # 凡例の表示
  plt.grid(True) # グラフにグリッド線を表示
  output_filename = "anomaly_detection_result.png"
  plt.savefig(output_filename) #グラフを保存
  return output_filename


def main() -> None:
  """
  チート検出プログラムのメイン処理
  Args:
    None
  Returns:
    None
  """
  # ゲームログを1000人分生成
  game_data = generate_game_logs()
  # データをCSVファイルとして保存
  game_data.to_csv("game_logs.csv", index=False)
  print("\nデータを 'game_logs.csv' に保存しました。")

  print("\n--- AIによる異常検知結果 ---")
  suspicious_players_by_ai, game_data_with_prediction_by_ai = ai_anomaly_detector(game_data.copy())
  suspicious_players_by_ai = suspicious_players_by_ai.round(2)
  print(suspicious_players_by_ai)

  print("\n--- ルールベースでの不正検知結果 ---")
  suspicious_players_by_rule = rule_based_detector(game_data_with_prediction_by_ai.copy())
  print(suspicious_players_by_rule)

  # 不正検知結果の統合
  game_data_with_prediction = game_data_with_prediction_by_ai
  game_data_with_prediction.loc[suspicious_players_by_rule.index, "anomaly_prediction"] = -1
  # 一般プレイヤーの確定
  normal_players = game_data_with_prediction[game_data_with_prediction['anomaly_prediction'] == 1]

  output_filename = make_fig(normal_players, suspicious_players_by_rule, suspicious_players_by_ai)
  print(f"\n検知結果のグラフを '{output_filename}' に保存しました。")

if __name__ == "__main__":
  main()