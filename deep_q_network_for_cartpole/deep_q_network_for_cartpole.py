import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import math
import numpy as np
from collections import namedtuple, deque

# --- ハイパーパラメータ ---
NUM_EPISODES = 200 # 学習するエピソード数
TARGET_UPDATE_FREQ = 12 # ターゲットネットワークを更新する頻度（エピソードごと）
LR = 0.0006 # 学習率

# 経験を保存するためのデータ構造を定義(Experience型を作成)
# [state, action, next_state, reward, done] 
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

# GPUが使える場合はGPUを、使えない場合はCPUを設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Qネットワークを定義するクラス
class QNetwork(nn.Module):
    """状態を入力とし、各行動のQ値を出力するニューラルネットワーク。"""
    def __init__(self, state_dim: int, action_dim: int) -> None:
        """
        QNetworkの層を初期化する
        Args:
          state_dim(int): 状態の次元数 (CartPoleでは4)
          action_dim(int): 行動の選択肢の数 (CartPoleでは2)
        Returns:
          None
        """
        # nn.Moduleを継承したクラスを作るため
        super(QNetwork, self).__init__()
        
        # ニューラルネットワークの層を定義
        # イメージ: 
        #・位置の重要度：高、その他普通の視点の箱
        #・速度の重要度：高、その他普通の視点の箱
        #・全部平均的に重要視する箱...
        # といったように、128個の「視点の箱」を作成し、実際に学習がはじまると、結果と相関が強い特徴量を重点的に参考にしながら学習を進めるイメージ
        # 入力層から中間層への線形変換
        self.layer1 = nn.Linear(state_dim, 128)
        # 中間層からさらなる中間層への線形変換→抽象的な特徴量を捉える
        self.layer2 = nn.Linear(128, 128)
        # 中間層から出力層への線形変換
        self.layer3 = nn.Linear(128, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        順伝播を定義する関数
        ネットワークにstateが入力された時に、どのように出力（Q値）が計算されるかを記述
        Args:
          state(totch.Tensor): 環境から得られる状態
        Returns:
          torch.Tensor: 各行動に対応するQ値を表すTensor
        """
        # layer1を通して、活性化関数ReLUを適用
        x = F.relu(self.layer1(state))
        # layer2を通して、活性化関数ReLUを適用
        x = F.relu(self.layer2(x))
        # 最後にlayer3を通して、出力（各行動のQ値）を計算
        return self.layer3(x)
    
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        """
        ReplayBufferを初期化する
        Args:
          capacity(int): バッファの最大容量
        Returns:
          None
        """
        # deque: リストに似ているが、先頭や末尾へのデータ追加・削除が非常に高速
        # 最大容量(maxlen)を超えると自動的に古いものから削除してくれる点で便利
        self.memory = deque([], maxlen=capacity)

    def push(self, state: torch.Tensor, action: int, next_state: torch.Tensor, reward: float, done: bool) -> None:
        """
        経験をメモリ（バッファ）に保存
        Args:
          state(torch.Tensor): 現在の状態
          action(int): 選択した行動
          next_state(torch.Tensor): 遷移後の状態
          reward(float): 得られた報酬
          done(bool): エピソードが終了したかどうか
        Returns:
          None
        """
        # Experience型のインスタンスを作成してメモリに追加
        self.memory.append(Experience(state, action, next_state, reward, done))

    def sample(self, batch_size: int) -> list[Experience]:
        """
        メモリから指定されたバッチサイズ分の経験をランダムに取得
        Args:
          batch_size(int): 取得する経験の数
        Returns:
          list[Experience]: ランダムに選ばれた経験のリスト
        """
        # random.sample: 重複なくランダムに要素を抽出
        # self.memoryからbatch_size個のExperienceをランダムに選択して返す
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """
        現在のメモリサイズを返す関数
        len(replay_buffer) のように呼び出せる
        Args:
          None
        Returns:
          int: メモリに保存されている経験の数
        """
        return len(self.memory)
    
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, replay_capacity: int = 10000, batch_size: int = 64, gamma: float = 0.99, lr: float = 1e-3) -> None:
        """
        DQNAgentを初期化
        Args:
          state_dim(int): 状態の次元数
          action_dim(int): 行動の選択肢の数
          replay_capacity(int): リプレイバッファの容量
          batch_size(int): 学習時のバッチサイズ
          gamma(float): 時間割引率
          lr(float): 学習率
        Returns:
          None
        """
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.steps_done = 0 # ε-greedyの計算に使用

        # Qネットワークとターゲットネットワークを作成 (q_network.py参照)
        # .to(device)で、モデルをGPUまたはCPUに転送 (以降、policy_net, target_netに関する計算を全てdevice上で行われる)
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        # ターゲットネットワークの重みをポリシーネットワークと同じにする
        # self.policy_net.state_dict(): policy_netの現在の"重み"を全て取り出す
        # self.target_net.load_state_dict(...): 取り出した重みをtarget_netに丸ごとコピーする(policy_net = target_netのイメージ)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # ターゲットネットワークは評価モード(学習はしない)にする

        # オプティマイザ（Adam）とリプレイバッファを設定
        # learnメソッドで計算された誤差（loss）に基づいて、policy_netの重みを具体的にどのように修正する（学習する）かを決めるアルゴリズム
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(replay_capacity)
        

    def act(self, state: np.ndarray) -> torch.Tensor:
        """
        ε-greedy法に基づいて行動を選択する
        (学習が進むにつれて、ランダムな行動（探索）の割合が減り、Q値が最大となる行動（活用）の割合が増える)
        Args:
          state(np.ndarray): 現在の状態
        Returns:
          torch.Tensor: 選択した行動
        """
        # εを計算（学習が進むほどεは小さくなる）
        eps_threshold = 0.05 + (0.9 - 0.05) * math.exp(-1. * self.steps_done / 200)
        self.steps_done += 1
        
        # 確率εでランダムに行動
        if random.random() < eps_threshold:
            # action_spaceからランダムに選択
            return torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)
        
        # 確率1-εでQ値が最大になる行動を選択
        else:
            with torch.no_grad(): # 勾配計算を無効化して高速化
                # stateをネットワーク入力用に整形し、deviceへ送る
                state_tensor = torch.tensor(np.array([state]), device=device, dtype=torch.float32)
                # policy_netでQ値を計算
                q_values = self.policy_net(state_tensor)
                # Q値が最大となる行動のインデックスを取得
                return q_values.max(1)[1].view(1, 1)

    def remember(self, state: np.ndarray, action: torch.Tensor, next_state: np.ndarray | None, reward: float, done: bool) -> None:
        """
        リプレイバッファに経験を保存する
        Args:
          state(np.ndarray): 現在の状態
          action(torch.Tensor): 選択した行動
          next_state(np.ndarray | None): 遷移後の状態
          reward(float): 得られた報酬
          done(bool): エピソードが終了したかどうか
        Returns:
          None"""
        # データをTensorに変換
        state = torch.tensor([state], device=device, dtype=torch.float32)
        action = action.to(device)
        # next_stateがNoneでない場合のみTensorに変換
        if next_state is not None:
            next_state = torch.tensor([next_state], device=device, dtype=torch.float32)
        reward = torch.tensor([reward], device=device, dtype=torch.float32)
        done = torch.tensor([done], device=device, dtype=torch.bool)
        
        # push(): 5つのデータをExperience型へ
        # memory: Experience型をリストのように保存する、dequeオブジェクトに保存
        self.memory.push(state, action, next_state, reward, done)

    def learn(self) -> None:
        """
        リプレイバッファからサンプリングしたデータでネットワークを学習する
        Args:
          None
        Returns:
          None
        """
        # バッファに十分なデータがなければ何もしない(memoryがbatch_size分溜まっていなければやらない)
        if len(self.memory) < self.batch_size:
            return

        # バッファから経験をサンプリング
        # experiences: [[state, action, next_state, reward, done],[state, action, next_state, reward, done],...]という状態(state, action, next_state, reward, doneはそれぞれ独立したテンソル)
        # テンソルとは、その要素について計算過程をひっくるめて記憶する、GPUにそれぞれの要素を並行して計算させられる型のイメージ
        experiences = self.memory.sample(self.batch_size)
        # Experience(*zip(*...)) は、ExperienceのリストをExperienceの各フィールドのタプルに変換するテクニック
        # : batch.stateに64個のすべての状態が、baych.actionに64個のすべての行動が格納されるイメージ
        # batch.state: [tensor([a,b,c,d]), tensor([a,b,c,d]),...]の状態([a,b,c,d]で１つのテンソル)
        batch = Experience(*zip(*experiences))

        # バッチ内のデータをそれぞれTensorにまとめる
        # state_batch: tensor([a,b,c,d],[a,b,c,d]...)(全体で1つのテンソル)
        # 全体で１つのテンソルにすることで、全体(64個)を並行してGPUに思考させられる
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # 終了状態でないnext_stateのみを抽出
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # 1. 現在のQ値（Q(s, a)）を計算: 「行動を選択したとき」
        # policy_netでQ値を計算し、実際に行動したactionのインデックスのQ値を取得
        # state_batch: (batch_size, state_dim): 過去に経験した、異なる64個の時点での状態を、一つの大きな行列（テンソル）にまとめたもの
        # action_batch: (batch_size, 1): state_batchの各状態において、その時実際に取った行動をまとめたもの
        # policy_net(state_batch).gather(1, action_batch): 現在の状態(state_batch)で、行動(action_batch)を行うことの価値(どれぐらい生き残れそうか)を現在の学習済み価値観で考えてみる
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 2. 目標となるQ値（r + γ * max Q(s', a')）を計算: 「その結果どうなったか」
        # target_netが計算する64個の価値を格納するための、空の配列を用意
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            # 終了状態でないものについて、target_netで遷移後の状態のQ値の最大値を取得
            # non_final_next_status: 現在の状態(state_batch)で、行動(action_batch)を行った結果の状態(next_state)のうち、まだ終わっていないもの
            # target_net(non_final_next_states).max(1)[0]: 敢えて最新の学習内容を含めない価値観によって、non_final_next_statusの状態の価値(どれぐらい生き残れそうか)を算出
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # 目標Q値の計算(Target = r + γ * Q_future_max = 1 + 0.99 * Q_future_maxにより、より安定した数値をターゲットとする)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # 3. 損失を計算「行動の選択とその結果の価値の乖離を計算」
        criterion = nn.SmoothL1Loss()
        # criterion: (Q_current - Target)²: "Sの状態ならaするべきだorしないべきだ"VS"その結果どうなったか"の乖離を計算
        # loss: 行動の選択とその結果の価値の乖離が大きい→"Sの状態ならaするべきだorしないべきだ"という思考によってもたらされたS'という状態の価値が、思っていた値(Q_current)と違う...→学習を反省するべき
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 4. ネットワークを更新(反省を元に、重みを修正(学習))
        self.optimizer.zero_grad() # 勾配(前回の思考)をリセット
        loss.backward() # 誤差逆伝播(Tensorの強み。どのような計算でその値になったのかを記憶しているため、さかのぼって原因を突き止められる)
        # 勾配が大きくなりすぎないようにクリッピング(修正量に上限を設ける)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step() # パラメータを更新

    # DQNAgentクラスの中に追加
    def update_target_net(self) -> None:
        """
        ポリシーネットワークの重みをコピーし、ターゲットネットワークの重みを更新する
        Args:
          None
        Returns:
          None"""
        self.target_net.load_state_dict(self.policy_net.state_dict())


def main() -> None:
    """
    メイン処理
    Args:
      None
    Returns:
      None
    """
    # 環境を作成
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # エージェントを作成
    agent = DQNAgent(state_dim, action_dim)

    # 各エピソードの合計報酬を記録するリスト
    episode_rewards = []

    print("学習を開始します...")
    # 指定したエピソード数だけ学習を繰り返す
    for episode in range(NUM_EPISODES):
        # 環境を初期化
        state, info = env.reset()
        done = False
        total_reward = 0

        # 1エピソードが終わるまでループ
        while not done:
            # 1. エージェントが行動を選択
            action = agent.act(state)

            # 2. 環境を1ステップ進める
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            # 終了状態になった場合、next_stateはNoneとする
            if done:
                next_state = None

            # 3. 経験をリプレイバッファに保存
            agent.remember(state, action, next_state, reward, done)

            # 4. 状態を更新
            state = next_state
            
            # 5. エージェントを学習させる
            agent.learn()

            # 合計報酬を更新
            total_reward += reward

        # エピソードの合計報酬を記録
        episode_rewards.append(total_reward)
        print(f"エピソード: {episode + 1}/{NUM_EPISODES}, 合計報酬: {total_reward}")

        # 一定のエピソードごとにターゲットネットワークを更新
        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_net()
            print(f"--- ターゲットネットワークを更新しました (エピソード {episode + 1}) ---")
    
    print("\n学習が完了しました。")
    env.close()

    # --- 学習結果のプロット ---
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

    # --- 学習済みモデルのデモンストレーション ---
    print("\n学習済みエージェントの動きを確認します。")
    env = gym.make("CartPole-v1", render_mode="human")
    state, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()
        with torch.no_grad(): # 学習しない設定
            # 今度はε-greedyを使わず、常に最善の行動を選択
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            q_values = agent.policy_net(state_tensor) # [[左に進む価値, 右に進む価値]]
            action = q_values.max(1)[1].view(1, 1)
        
        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        state = next_state
        total_reward += reward
    
    print(f"デモンストレーションの合計報酬: {total_reward}")
    env.close()

if __name__ == '__main__':
    main()