import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# ハイパーパラメータ
EPOCHS: int = 10
BATCH_SIZE: int = 128
LEARNING_RATE: float = 1e-3

# モデルの次元設定
INPUT_DIM: int = 784  # 28x28
HIDDEN_DIM: int = 400
LATENT_DIM: int = 20

# その他
RESULTS_DIR: str = 'results'


class VAE(nn.Module):
  """
  変分オートエンコーダのモデル。エンコーダとデコーダから構成される。
  エンコーダは入力を潜在空間の確率分布（平均と対数分散）に変換し、デコーダは潜在空間の点から元のデータを復元する。
  """

  def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
    """
    モデルの層を初期化する。
    Args:
      input_dim (int): 入力次元数。入力画像のピクセル数に値する(28*28=784)
      hidden_dim (int): 中間層の次元数。HIDDEN_DIM = 400個の特徴量を、入力画像から抽出(全体的に縦の線が多いか？、画像の中央に丸い空間があるか？等)
      latent_dim (int): 潜在変数の次元数
    Returns:
      None
    """
    super(VAE, self).__init__()

    # --- エンコーダの定義 ---
    # input_dim -> hidden_dim: 入力画像から特徴量を入手
    # hidden_dim -> latent_dim: 特徴量の特徴量を入手
    # このプログラムは上記の２段階だが、より多くの層を追加することでより複雑なデータ解析が可能(その分、過学習のリスクが向上)
    self.fc1 = nn.Linear(input_dim, hidden_dim) # 入力画像を中間層に変換する(784 -> 400)
    self.fc21 = nn.Linear(hidden_dim, latent_dim) # 中間層からさらに特徴量を抽出(400 -> 20)。後の学習時に平均(mu)を計算するために使われる層。平均的にこのような特徴があるということを示す
    self.fc22 = nn.Linear(hidden_dim, latent_dim) # 中間層からさらに特徴量を抽出(400 -> 20)。後の学習時に対数分散(logvar)を計算する層。特徴量のばらつきや自由度を示す

    # --- デコーダの定義 ---
    # 特徴量から元のデータを復元する
    # latent_dim -> hidden_dim -> input_dim
    self.fc3 = nn.Linear(latent_dim, hidden_dim)
    self.fc4 = nn.Linear(hidden_dim, input_dim)

  def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    エンコーダ部分。入力xから潜在空間のパラメータ（平均と対数分散）を計算する。
    Args:
      x (torch.Tensor): 入力テンソル (バッチサイズ, 入力次元数)
    Returns:
      Tuple[torch.Tensor, torch.Tensor]: 平均と対数分散のタプル
    """
    h1 = F.relu(self.fc1(x)) # 活性化関数ReLU: 負の値を0に変換し、正の値はそのまま通す(中間層に変換させる際に、負の値を0に)
    return self.fc21(h1), self.fc22(h1)

  def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    再パラメータ化トリックを用いて、潜在変数zをサンプリングする。
    これにより、勾配が計算可能になり、モデルの学習が可能になる。
    Args:
      mu (torch.Tensor): 平均
      logvar (torch.Tensor): 対数分散
    Returns:
      torch.Tensor: サンプリングされた潜在変数z
    """
    std = torch.exp(0.5 * logvar) # 標準偏差を計算
    # torch.randn: 標準正規分布からランダムに値を生成
    # _like(std): stdというテンソルと全く同じ形状・データ型・デバイスを持つテンソルで値を生成
    eps = torch.randn_like(std) # 標準正規分布からランダムに値を得ることでノイズを生成。
    # eps * std: epsは標準正規分布に従うが、これにstdを掛けることで、標準偏差がstdに従うように変更。それにより、ノイズのばらつき具合が変わる
    return mu + eps * std # z = 平均 + (ノイズ * 標準偏差)

  def decode(self, z: torch.Tensor) -> torch.Tensor:
    """
    デコーダ部分。潜在変数zからデータを復元する。
    Args:
      z (torch.Tensor): 潜在変数
    Returns:
      torch.Tensor: 復元されたデータ。出力はsigmoid関数で0〜1の範囲に収められる。
    """
    h3 = F.relu(self.fc3(z))
    return torch.sigmoid(self.fc4(h3))

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    モデルの順伝播を定義する。
    入力画像xが、どのようにして再構成画像recon_xに変換されるか。
    Args:
      x (torch.Tensor): 入力データ (バッチサイズ, 1, 28, 28)
    Returns:
      Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        - recon_x (torch.Tensor): 再構成されたデータ
        - mu (torch.Tensor): 潜在空間の平均
        - logvar (torch.Tensor): 潜在空間の対数分散
    """
    # view(-1, 784)で(バッチサイズ, 1, 28, 28)の画像を(バッチサイズ, 784)に平坦化(ただのベクトルに変換)
    mu, logvar = self.encode(x.view(-1, INPUT_DIM)) # エンコード
    z = self.reparameterize(mu, logvar) # 再パラメータ化
    recon_x = self.decode(z) # デコード
    return recon_x, mu, logvar


def loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
  """
  VAEの損失関数を計算する。
  損失は再構成誤差と正則化項(KLダイバージェンス)の和で構成される。
  Args:
    recon_x (torch.Tensor): モデルによって再構成されたデータ
    x (torch.Tensor): 元の入力データ
    mu (torch.Tensor): 潜在空間の平均
    logvar (torch.Tensor): 潜在空間の対数分散
  Returns:
    torch.Tensor: 計算された合計損失
  """
  # 再構成誤差
  # デコーダが生成した画像(recon_x)と元の画像(x)がどれだけ近いかを測る
  # recon_x（生成された画像）とx（元の画像）を、ピクセル単位で一つずつ比較し、自信を持って間違えた場合に非常に大きなペナルティを与える
  # - 例：元のピクセルが白(1.0)なのに、生成画像が黒(0.1)だと、誤差は非常に大きくなる
  # - 例：元のピクセルが白(1.0)で、生成画像が薄いグレー(0.8)なら、誤差は比較的小さくなる
  # reduction='sum': 画像内の全ピクセルの誤差と、バッチ内の全画像の誤差をすべて合計して、一つの数値にまとめる
  BCE = F.binary_cross_entropy(recon_x, x.view(-1, INPUT_DIM), reduction='sum')

  # KLダイバージェンス
  # エンコーダが作る潜在空間の分布と、標準正規分布がどれだけ近いかを測る
  # 沢山ある画像同士を、標準正規分布の図の内側に配置させるように促すことで、図のどこにでもプロットできてしまう状態を防ぎ、画像同士のつながりを生むイメージ
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

  # BCEとKLDはトレードオフの関係
  # BCEとKLDの両方、少ないほどよい
  return BCE + KLD


def train(epoch: int, model: VAE, train_loader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> None:
  """
  1エポック分のモデル学習を行う。
  Args:
    epoch (int): 現在のエポック数
    model (VAE): 学習対象のVAEモデル
    train_loader (DataLoader): 学習用データローダー
    optimizer (optim.Optimizer): オプティマイザ
    device (torch.device): 使用するデバイス
  Returns:
    None
  """
  model.train()  # モデルを学習モードに設定
  train_loss = 0.0
  # train_loader: 60,000枚のMNIST画像が入っている
  # バッチサイズが128(定数定義の部分): 60000枚の画像を128枚ずつに分割して、60000/128=約469回ループを回す
  # 1回のループで128枚の画像を使って学習を行うが、その塊をミニバッチという
  for batch_idx, (data, _) in enumerate(train_loader):
    data = data.to(device) # CPU or GPUにデータを転送
    optimizer.zero_grad() # 勾配を初期化

    recon_batch, mu, logvar = model(data) # 画像データを学習モデルに渡し、返り値を受け取る(順伝播)
    loss = loss_function(recon_batch, data, mu, logvar) # 損失を計算

    loss.backward() # 損失を元に、モデル内のパラメータをどのように動かせば損失を減らせるか計算(逆伝播)
    optimizer.step() # 計算された勾配(改善案)を使って、実際にモデルのパラメータを更新

    train_loss += loss.item() # バッチの損失を累積

    if batch_idx % 100 == 0:
      print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
            f' ({100. * batch_idx / len(train_loader):.0f}%)]\t'
            f'Loss: {loss.item() / len(data):.6f}')

  avg_loss = train_loss / len(train_loader.dataset)
  print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')


def test(epoch: int, model: VAE, test_loader: DataLoader, device: torch.device) -> None:
  """
  1エポック分(一周学習し終わること。何周もデータセットを学習する。)のモデル評価（テスト）を行う。
  Args:
    epoch (int): 現在のエポック数
    model (VAE): 評価対象のVAEモデル
    test_loader (DataLoader): テスト用データローダー
    device (torch.device): 使用するデバイス
  Returns:
    None
  """
  model.eval()  # モデルを評価モードに設定
  test_loss = 0.0
  with torch.no_grad():  # 勾配(改善案)計算をしない設定にする(メモリ効率の向上)。エポックごとのテストのため。
    for i, (data, _) in enumerate(test_loader): # test_loaderには、学習に使用していない初見のデータセットが10000枚存在
      data = data.to(device) # デバイスにデータを送信
      recon_batch, mu, logvar = model(data) # 順伝播
      test_loss += loss_function(recon_batch, data, mu, logvar).item() # 損失を計算し、累積

      # 最初のミニバッチの結果について、元画像と比較できる形で保存
      if i == 0:
        n = min(data.size(0), 8)
        # data[:n]: 元画像の最初のn枚
        # recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]: AIが再構成した画像の最初のn枚
        # comparison = torch.cat([...]): 元の画像8枚と再構成された画像8枚を、上下に連結して1枚の画像にする
        comparison = torch.cat([data[:n], recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
        save_image(comparison.cpu(), os.path.join(RESULTS_DIR, f'reconstruction_{epoch}.png'), nrow=n) # 画像を保存

  avg_loss = test_loss / len(test_loader.dataset)
  print(f'====> Test set loss: {avg_loss:.4f}')


def main() -> None:
  """
  MNISTデータセットをダウンロードし、VAEモデルの学習、評価、画像生成を行うメイン関数。
  Args:
    None
  Returns:
    None
  """
  # --- デバイスの確認 ---
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # --- データの準備 ---
  transform = transforms.ToTensor()
  train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
  test_dataset = datasets.MNIST('./data', train=False, transform=transform)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

  # --- モデル、オプティマイザの準備 ---
  model = VAE(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM
  ).to(device)
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

  # --- 結果を保存するフォルダを作成 ---
  os.makedirs(RESULTS_DIR, exist_ok=True)

  # --- 学習とテストのループ ---
  for epoch in range(1, EPOCHS + 1):
    train(epoch, model, train_loader, optimizer, device)
    test(epoch, model, test_loader, device)

    # 各エポック終了時に、ランダムな潜在変数から画像を生成して保存
    with torch.no_grad():
      sample = torch.randn(64, LATENT_DIM).to(device) # 学習した潜在空間の標準正規分布の中から、ランダムに64個の値をサンプリング
      generated_images = model.decode(sample).cpu()
      save_image(generated_images.view(64, 1, 28, 28), os.path.join(RESULTS_DIR, f'sample_{epoch}.png'))

  print(f"学習が完了しました。'{RESULTS_DIR}'フォルダを確認してください。")


if __name__ == '__main__':
    main()