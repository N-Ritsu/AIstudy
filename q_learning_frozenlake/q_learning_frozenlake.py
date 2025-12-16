import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque

class QLearningAgent:
  """
  FrozenLake-v1環境でQ学習を実行するエージェント
  """
  def __init__(self, env_name: str = "FrozenLake-v1", is_slippery: bool = False, render_mode: str = None, **kwargs) -> None:
    """
    Q学習エージェントを初期化する
    Args:
      env_name (str): gymnasium環境の名前
      is_slippery (bool): FrozenLake環境が滑るかどうか
      render_mode (str, optional): 環境のレンダリングモード。学習中はNone、評価時は"human"など。
      **kwargs: Q学習のハイパーパラメータを設定するためのキーワード引数
    """
    # 環境のセットアップ
    self.env = gym.make(env_name, is_slippery=is_slippery, render_mode=render_mode)
    self.n_states = self.env.observation_space.n # 状態の総数(4マス×4マスの計16)
    self.n_actions = self.env.action_space.n # 行動の総数(上下左右の計4)

    # Qテーブルを全て0で初期化
    # Qテーブルは [状態数, 行動数] の2次元配列で、各(状態,行動)ペアの価値を保持
    # 行が状態、列が行動に対応
    self.q_table = np.zeros((self.n_states, self.n_actions))

    print(f"環境: {env_name}, 滑り: {is_slippery}")
    print(f"状態数: {self.n_states}")
    print(f"行動数: {self.n_actions}")
    print(f"Qテーブルの形状: {self.q_table.shape}")

    # Q学習のハイパーパラメータ
    self.config = {
      "n_episodes": 20000,              # エピソード数(ゲームをプレイする回数)
      "max_steps_per_episode": 100,     # 1エピソードあたりの最大ステップ数。これを超えると強制的にエピソードが終了する。
      "initial_learning_rate": 0.14,     # α: 初期学習率。新しい情報をどれだけQ値に反映させるか(0.0 ~ 1.0)
      "discount_factor": 0.95,          # γ: 割引率。将来の報酬をどれだけ重視するか(0.0 ~ 1.0)。1.0に近いほど、将来的な報酬を重視するため最短経路の模索を後回しにすることから学習速度は低下。
      "epsilon": 1.0,                   # 初期探索率: 学習初期は探索を多くする(ランダムな行動を選ぶ確率)
      "max_epsilon": 1.0,               # epsilonの上限
      "min_epsilon": 0.0001,            # epsilonの下限: ある程度学習したらランダムな探索を減らし、学習内容の活用を増やす
      "epsilon_decay_rate": 0.0005,     # epsilonの減衰率: エピソードごとにepsilonをどれだけ減らすか
      "step_penalty": -0.01,            # 最短経路を学習させるためのステップごとのペナルティ
      "avg_reward_window_for_lr_adjustment": 500, # 直近の平均報酬を計算するために使用するエピソード数
      "lr_boost_threshold_multiplier": 1.1,      # 今回のエピソード報酬が直近の平均報酬の何倍以上であれば学習率を上げるか
      "high_learning_rate_multiplier": 1.1,       # 成功したエピソードの次に適用する高い学習率の倍率。initial_learning_rate にこの倍率を掛ける。
      "max_adjusted_learning_rate": 0.95,         # 調整後の学習率がこの値を超えないようにするための上限。学習率が1.0を超えると不安定になる可能性があるため注意。
    }
    # 引数で渡されたパラメータでデフォルト設定を上書き
    self.config.update(kwargs)

    # 現在のエピソードで実際に使われる学習率を保持する変数。初期値はinitial_learning_rate。
    self.current_learning_rate = self.config["initial_learning_rate"]
    # 学習率調整のための直近の報酬を保持(効率的に計算するためdequeを使用)
    self.recent_rewards = deque(maxlen=self.config["avg_reward_window_for_lr_adjustment"])

  def _choose_action(self, state: int, current_epsilon: float) -> int:
    """
    ε-greedy法に基づいて行動を選択する
    ある確率 ε でランダムな行動を選択し(探索)、残りの確率 1 - ε で現在の知識に基づいて最も価値が高いと予測される行動を選択する(活用)
    0から1までの範囲で乱数を生成し、生成された乱数が ε より小さければランダムな行動を、ε 以上であればQテーブルに基づく最適な行動を選択
    Args:
      state (int): 現在の状態
      current_epsilon (float): 現在の探索率(ε)
    Returns:
      int: 選択された行動
    """
    if np.random.uniform(0, 1) < current_epsilon:
      # 探索: ランダムな行動を選択
      return self.env.action_space.sample()
    else:
      # 活用: Qテーブルに基づいて最適な行動を選択
      # np.argmax(q_table[state, :])は現在の状態でQ値(報酬の期待値)が最大の行動のインデックスを返す
      return np.argmax(self.q_table[state, :])

  def _update_q_value(self, state: int, action: int, actual_reward: float, new_state: int) -> None:
    """
    Q値を更新する
    Args:
      state (int): 現在の状態
      action (int): 実行された行動
      actual_reward (float): 修正後の報酬
      new_state (int): 次の状態
    Returns:
      None
    """
    # Q値の更新式(ベルマン方程式)
    # Q(s,a) = Q(s,a) + current_learning_rate * [actual_reward + discount_factor * max(Q(s',a')) - Q(s,a)]
    # s: 現在の状態, a: 現在の行動, s': 次の状態, max(Q(s',a')): 次の状態での最適なQ値
    self.q_table[state, action] = self.q_table[state, action] + self.current_learning_rate * (actual_reward + self.config["discount_factor"] * np.max(self.q_table[new_state, :]) - self.q_table[state, action])

  def _adjust_learning_rate(self, episode: int, current_episode_rewards: float) -> None:
    """
    直近の平均報酬に基づいて学習率を動的に調整する
    Args:
      episode (int): 現在のエピソード数
      current_episode_rewards (float): 現在のエピソードで得られた報酬の合計
    Returns:
      None
    """
    self.recent_rewards.append(current_episode_rewards)

    # エピソード数がavg_reward_window_for_lr_adjustmentより少ない間は、十分なデータがないため調整を行わない
    if episode >= self.config["avg_reward_window_for_lr_adjustment"]:
      # 直近の avg_reward_window_for_lr_adjustment エピソードの平均報酬を計算(np.mean: 平均値を計算)
      recent_average_reward = np.mean(self.recent_rewards)
      
      # 今回のエピソード報酬が直近の平均報酬を上回っているかチェック
      # ただし、平均報酬が0の場合（特に学習初期段階でまだ報酬が得られていない場合など）のゼロ除算を防ぐ
      # 平均が負または0で、今回が正の報酬だった場合は、それ自体が"良い"ため学習率を上げる
      should_boost_lr = (current_episode_rewards > 0 and recent_average_reward <= 0) or \
                        (recent_average_reward > 0 and (current_episode_rewards / recent_average_reward) >= self.config["lr_boost_threshold_multiplier"])

      if should_boost_lr:
        # min(計算された学習率, 学習率上限)
        self.current_learning_rate = min(self.config["initial_learning_rate"] * self.config["high_learning_rate_multiplier"], self.config["max_adjusted_learning_rate"])
      else:
        # falseなら元の学習率に戻す
        self.current_learning_rate = self.config["initial_learning_rate"]
    else:
      # まだ十分なエピソードがない場合は、初期学習率を使用
      self.current_learning_rate = self.config["initial_learning_rate"]

  def train(self) -> list:
    """
    Q学習のトレーニングループを実行する
    Returns:
      list: 各エピソードで得られた報酬のリスト
    """
    rewards_per_episode = []
    epsilon = self.config["epsilon"] # 初期epsilon

    print("\n--- Q学習を開始します ---")
    for episode in range(self.config["n_episodes"]):
      # 各エピソードの開始時に環境をリセットし、初期状態を取得
      state, info = self.env.reset()
      terminated = False # エピソードが終了したか(ゴール到達または穴落ち)
      truncated = False  # 最大ステップ数に達したか
      current_episode_rewards = 0 # このエピソードでの合計報酬を初期化

      # 1エピソード内のステップループ
      for step in range(self.config["max_steps_per_episode"]):
        # 探索or活用の行動選択
        action = self._choose_action(state, epsilon)

        # 選択した行動を実行し、次の状態、環境からの報酬、終了情報を得る
        new_state, reward_from_env, terminated, truncated, info = self.env.step(action)

        # 報酬の修正ロジック
        # 最短経路学習のために、行動の度に負の報酬を与える
        if reward_from_env == 0: # 環境からの報酬が0の場合(ゴール以外の場合)
          actual_reward = self.config["step_penalty"] # 定義したステップペナルティを適用
        else: # 環境からの報酬が1.0の場合(ゴールに到達した場合)
          actual_reward = reward_from_env # ゴール報酬はそのまま(+1.0)

        # Q値の更新
        self._update_q_value(state, action, actual_reward, new_state)

        # テーブル上の現在の状態を更新
        state = new_state
        # 今回のステップで得られた修正後の報酬をエピソード合計報酬に加算
        current_episode_rewards += actual_reward

        # エピソードが終了(ゴール到達または穴落ち)または最大ステップ数に達したら、ステップループを抜ける
        if terminated or truncated:
          break

      # エピソード終了後にepsilonを減衰させる(探索率を徐々に下げ、活用を増やしていく)
      # 指数関数的にepsilonを減衰させる式(強化学習アルゴリズムにおいてεの減衰方法として定石)
      epsilon = self.config["min_epsilon"] + (self.config["max_epsilon"] - self.config["min_epsilon"]) * np.exp(-self.config["epsilon_decay_rate"] * episode)
      
      # このエピソードの合計報酬を記録
      rewards_per_episode.append(current_episode_rewards)

      # 直近平均報酬を用いた学習率調整
      self._adjust_learning_rate(episode, current_episode_rewards)

    # 全てのエピソードが終了したら環境を閉じる
    self.env.close()

    print("\n--- Q学習が完了しました ---")
    print(f"最終 epsilon: {epsilon:.4f}")
    print("学習後のQテーブルの先頭5行:\n", self.q_table[:5, :])
    return rewards_per_episode

  def evaluate(self, n_test_episodes: int = 10, render: bool = True) -> float:
    """
    学習済みAIを評価し、その動作を可視化する
    Args:
      n_test_episodes (int): テストするエピソード数
      render (bool): 動作を画面に表示するかどうか
    Returns:
      float: テストエピソードでの平均報酬
    """
    # 新しい環境インスタンスを作成し、今度はrender_mode="human"を設定することで画面表示
    eval_env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human" if render else None)

    total_rewards_test = 0 # 評価時の合計報酬（修正された報酬の合計）

    print(f"\n--- 学習済みAIの動作確認（{n_test_episodes} エピソード）---")

    for episode in range(n_test_episodes):
      # 各評価エピソードの開始時に環境をリセット
      state, info = eval_env.reset()
      terminated = False # エピソードが終了したか
      truncated = False # 最大ステップ数に達したか
      print(f"\n--- テストエピソード {episode+1} ---")

      # 1評価エピソード内のステップループ
      for step in range(self.config["max_steps_per_episode"]):
        # 評価時は探索を行わず、学習済みのQテーブルに基づいて常に最適な行動を選択する
        # state: 現在の状態、：:すべての行動、self.q_table[state, :]: 現在の状態における各行動それぞれ(上下左右)に対するQ値
        # np.argmax: 4つのQ値の配列の中から、max値のインデックス(行動)を返す
        action = np.argmax(self.q_table[state, :])

        # 行動を実行し、次の状態、環境からの報酬、終了情報を得る
        new_state, reward_from_env, terminated, truncated, info = eval_env.step(action)
        
        # 評価時にも学習時と同じ報酬修正ロジックを適用して、AIが獲得した報酬を記録
        if reward_from_env == 0: # 環境からの報酬が0の場合
          actual_reward = self.config["step_penalty"]
        else: # 環境からの報酬が1.0の場合
          actual_reward = reward_from_env

        if render:
          eval_env.render() # 環境を描画してAIの動きを可視化

        total_rewards_test += actual_reward # 評価時の合計報酬に修正された報酬を加算
        state = new_state

        # エピソードが終了または最大ステップ数に達したら、ステップループを抜ける
        if terminated or truncated:
          break
      
      # AIの動きを見やすくするためのsleep
      if render:
        time.sleep(0.1)

    # 評価環境を閉じる
    eval_env.close()

    average_test_reward = total_rewards_test / n_test_episodes
    print(f"\n合計 {n_test_episodes} 回のテストエピソードでの平均報酬: {average_test_reward:.4f}")

    # 成功判定の閾値は、ステップペナルティを考慮して調整する必要あり
    # FrozenLake-v1(4x4)の最短経路は6ステップ
    # ゴール+1.0、各ステップ-0.01の場合、最短経路での最終報酬は 1.0 + ((6 - 1) * -0.01) = 0.95 (ゴール時のペナルティはないため実質5ステップ分のペナルティ)
    if average_test_reward > 0.8: # 閾値は環境やハイパーパラメータで適宜調整
      print("AIはFrozenLakeを効率的に攻略できています")
    else:
      print("AIは改善の余地がありそうです")
    
    return average_test_reward

def plot_smoothed_rewards(rewards_per_episode: list, smoothing_window: int = 100) -> None:
  """
  エピソードごとの報酬を平滑化してプロットする
  Args:
    rewards_per_episode (list): 各エピソードで得られた報酬のリスト
    smoothing_window (int): 移動平均を計算するための窓幅
  Returns:
    None
  """
  # 報酬の推移をプロットして学習の進捗を確認
  # エピソードごとの報酬は変動が大きいため、移動平均で平滑化するとトレンドが見やすくなる
  if len(rewards_per_episode) < smoothing_window:
    print(f"報酬データが少なすぎるため、平滑化された報酬はプロットできません (データ数: {len(rewards_per_episode)}, ウィンドウ: {smoothing_window})")
    return
  # np.convolve: 畳み込み演算を行う関数
  # validモードは、結果が完全にオーバーラップする部分のみを返す
  smoothed_rewards = np.convolve(rewards_per_episode, np.ones(smoothing_window)/smoothing_window, mode='valid')

  plt.figure(figsize=(12, 6)) # グラフのサイズを設定
  plt.plot(smoothed_rewards) # 平滑化された報酬をプロット
  plt.title(f"Smoothed Rewards per Episode (Q-Learning) - Window: {smoothing_window}") # グラフタイトル
  plt.xlabel(f"Episode (x {smoothing_window})") # グラフの横軸ラベル
  plt.ylabel("Average Reward") # グラフの縦軸ラベル
  plt.grid(True) # グリッド線を表示
  plt.show() # グラフを表示

  print(f"\n合計 {len(rewards_per_episode)} エピソード中、最終 {smoothing_window} エピソードの平均報酬: {np.mean(rewards_per_episode[-smoothing_window:]):.4f}")

def main() -> None:
  """
  Q学習エージェントのトレーニングと評価を実行するメイン関数
  Args:
    None
  Returns:
    None
  """
  # エージェントの初期化(学習用はrender_mode=None)
  agent = QLearningAgent(
    env_name="FrozenLake-v1",
    is_slippery=False,
    render_mode=None
  )

  # トレーニングの実行
  rewards_during_training = agent.train()

  # 結果の分析と可視化
  plot_smoothed_rewards(rewards_during_training, smoothing_window=100)

  # 学習済みAIの動作確認(評価モード)
  agent.evaluate(n_test_episodes=10, render=True)

if __name__ == "__main__":
  main()