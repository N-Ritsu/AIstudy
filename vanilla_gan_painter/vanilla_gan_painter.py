import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from typing import Tuple
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# --- 設定クラス ---
class Config:
    """
    学習やモデルに関する設定を管理するクラス。
    """
    BATCH_SIZE: int = 64
    Z_DIM: int = 100  # ノイズの次元
    IMAGE_SIZE: int = 28 * 28  # 画像のピクセル数
    LR: float = 0.0002  # 学習率
    NUM_EPOCHS: int = 50  # 学習エポック数
    BETA1: float = 0.5 # Adamオプティマイザのbeta1パラメータ
    # 生成画像の保存先ディレクトリ
    OUTPUT_DIR: str = "images"
    # 学習済みモデルの保存パス
    GENERATOR_PATH: str = "generator.pth"
    DISCRIMINATOR_PATH: str = "discriminator.pth"


# --- Generator（生成器）の定義 ---
class Generator(nn.Module):
    """
    潜在変数（ノイズ）から画像を生成するニューラルネットワーク
    """
    def __init__(self, z_dim: int, image_size: int):
        """
        Generatorのコンストラクタ
        Args:
            z_dim (int): 入力ノイズの次元数
            image_size (int): 生成する画像の総ピクセル数
        Returns:
            None
        """
        # 親クラスのコンストラクタを呼び出す
        super(Generator, self).__init__()

        # ニューラルネットワークの層を定義する
        # nn.Sequentialで層を順番に重ねていく
        self.model = nn.Sequential(
            # 入力はz_dim次元のノイズ
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),  # 活性化関数 LeakyReLU
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_size),
            # 出力層の活性化関数はtanh。これにより出力が-1〜1の範囲になる
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        順伝播の処理を定義
        Args:
            z (torch.Tensor): 入力ノイズのテンソル(バッチサイズ, z_dim)
        Returns:
            torch.Tensor: 生成された画像のテンソル(バッチサイズ, 1, 28, 28)
        """
        # 入力zをモデルに通して画像データを生成
        output = self.model(z)
        # 出力の形状を(バッチサイズ, 1, 28, 28)の画像形式に変換
        # 注意: MNISTの画像サイズ(28x28)にハードコードされています
        output = output.view(z.size(0), 1, 28, 28)
        return output


# --- Discriminator（識別器）の定義 ---
class Discriminator(nn.Module):
    """
    入力された画像がデータセット由来の本物かGeneratorが生成した偽物かを見分けるニューラルネットワーク
    """
    def __init__(self, image_size: int):
        """
        Discriminatorのコンストラクタ
        Args:
            image_size (int): 入力画像の総ピクセル数
        Returns:
            None
        """
        super(Discriminator, self).__init__()

        # ニューラルネットワークの層を定義
        self.model = nn.Sequential(
            # 入力はimage_size次元 (28*28=784)
            nn.Linear(image_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # 出力層の活性化関数はSigmoid。0から1の間の値（確率）を出力する
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播の処理を定義
        Args:
            x (torch.Tensor): 入力画像のテンソル(バッチサイズ, 1, 28, 28)
        Returns:
            torch.Tensor: 入力画像が本物である確率(バッチサイズ, 1)
        """
        # 入力画像xを(バッチサイズ, 784)の形状に平坦化（フラット化）する
        x_flat = x.view(x.size(0), -1)
        # モデルに画像を通して、本物である確率（0-1）を出力
        output = self.model(x_flat)
        return output

def prepare_dataloader(batch_size: int) -> DataLoader:
    """
    MNISTデータセットを準備し、DataLoaderを作成
    Args:
        batch_size (int): バッチサイズ
    Returns:
        DataLoader: MNIST学習用のDataLoader
    """
    # データの前処理を定義
    # ToTensor(): PILImageやnumpy配列をTensorに変換
    # Normalize((0.5,), (0.5,)): 画像のピクセル値を[-1, 1]の範囲に正規化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # MNISTデータセットをダウンロード・読み込み
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    # データセットからバッチ単位でデータを取得するためのDataLoaderを作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader

def train_gan(
    G: Generator,
    D: Discriminator,
    train_loader: DataLoader,
    config: Config
) -> None:
    """
    GeneratorとDiscriminatorの学習を実行
    Args:
        G (Generator): 生成器モデル
        D (Discriminator): 識別器モデル
        train_loader (DataLoader): 学習用データのDataLoader
        config (Config): 学習に関する設定
    Returns:
        None
    """
    # 損失関数：バイナリクロスエントロピー損失
    # GANの出力（本物か偽物かの確率）と正解ラベル（1 or 0）の差を計算
    criterion = nn.BCELoss()

    # オプティマイザ：Adam
    # 勾配を元にモデルのパラメータを更新するアルゴリズム
    optimizer_G = optim.Adam(G.parameters(), lr=config.LR, betas=(config.BETA1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=config.LR, betas=(config.BETA1, 0.999))
    
    # 進捗確認用の固定ノイズ
    # 毎回同じノイズを使うことで、学習の進捗が比較しやすくなる
    fixed_noise = torch.randn(config.BATCH_SIZE, config.Z_DIM)

    print("学習を開始します...")
    # --- 学習ループ ---
    for epoch in range(config.NUM_EPOCHS):
        for i, (real_images, _) in enumerate(train_loader):
            # --- Discriminator (D) の学習 ---
            # 勾配を初期化
            optimizer_D.zero_grad()

            # 1. 本物の画像で学習
            # 本物の画像に対する正解ラベルは1
            real_labels = torch.ones(real_images.size(0), 1)
            # Dに本物の画像を入力し、出力を得る
            real_outputs = D(real_images)
            # 本物の画像に対する損失を計算
            d_loss_real = criterion(real_outputs, real_labels)

            # 2. 偽物の画像で学習
            # Gに入力するノイズを生成
            noise = torch.randn(real_images.size(0), config.Z_DIM)
            # Gでノイズから偽物の画像を生成
            fake_images = G(noise)
            # 偽物の画像に対する正解ラベルは0
            fake_labels = torch.zeros(real_images.size(0), 1)
            # Dに偽物の画像を入力し、出力を得る
            # fake_images.detach()でGの勾配がDに伝わらないようにする
            fake_outputs = D(fake_images.detach())
            # 偽物の画像に対する損失を計算
            d_loss_fake = criterion(fake_outputs, fake_labels)

            # Dの損失は、本物の損失と偽物の損失の合計
            d_loss = d_loss_real + d_loss_fake
            # 誤差逆伝播
            d_loss.backward()
            # パラメータの更新
            optimizer_D.step()

            # --- Generator (G) の学習 ---
            # 勾配を初期化
            optimizer_G.zero_grad()
            # Gの目標は、Dが偽物画像を本物と誤認すること。
            # そのため、Gの学習では偽物画像に対する正解ラベルを1とする。
            labels = torch.ones(real_images.size(0), 1)
            # Dに再度、偽物画像を通して出力を得る（今回はGの学習のためdetachしない）
            outputs = D(fake_images)
            # Gの損失を計算
            g_loss = criterion(outputs, labels)
            # 誤差逆伝播
            g_loss.backward()
            # パラメータの更新
            optimizer_G.step()
        
        # 1エポック終了ごとに進捗を表示
        print(
            f"エポック [{epoch+1}/{config.NUM_EPOCHS}] | "
            f"D損失: {d_loss.item():.4f} | G損失: {g_loss.item():.4f}"
        )

        # 1エポックごとに画像を保存
        # fixed_noiseから画像を生成し、グリッド状にして保存
        with torch.no_grad(): # 勾配計算を無効化
            fake_images_for_save = G(fixed_noise)
            vutils.save_image(
                fake_images_for_save,
                os.path.join(config.OUTPUT_DIR, f"epoch_{epoch+1:03}.png"), # ファイル名にエポック番号を入れる
                normalize=True # 画像のピクセル値を0-1の範囲に正規化してくれる
            )

    print("学習が完了しました。")
    # 学習済みモデルのパラメータを保存
    torch.save(G.state_dict(), config.GENERATOR_PATH)
    torch.save(D.state_dict(), config.DISCRIMINATOR_PATH)
    print(f"学習済みモデルを {config.GENERATOR_PATH} と {config.DISCRIMINATOR_PATH} に保存しました。")


def generate_and_show_image(G: Generator, config: Config) -> None:
    """
    学習済みのGeneratorを使い、生成した画像のうち1枚を表示
    Args:
        G (Generator): 学習済みの生成器モデル
        config (Config): 設定情報
    Returns:
        None
    """
    # モデルを評価モードにする (Dropout層などがある場合に挙動が変わる)
    G.eval()

    # 新しいノイズから画像を生成
    with torch.no_grad(): # 勾配計算を無効化
        new_noise = torch.randn(config.BATCH_SIZE, config.Z_DIM)
        final_images = G(new_noise)

    # 生成された画像の中から1枚選んで表示
    # 最初の1枚を取得
    img_to_show = final_images[0]
    
    # Matplotlibで表示するために、テンソルを整形する
    # 1. CPUに移動: .cpu()
    # 2. 勾配情報を削除: .detach()
    # 3. NumPy配列に変換: .numpy()
    # 4. チャンネルの次元を最後に移動: (C, H, W) -> (H, W, C)
    img_np = img_to_show.cpu().detach().numpy().transpose(1, 2, 0)

    # ピクセル値の範囲を[-1, 1]から[0, 1]に戻す（表示のため）
    img_np = (img_np + 1) / 2

    # 画像を表示
    plt.imshow(img_np.squeeze(), cmap='gray') # squeeze()で次元1の軸を削除
    plt.title("Generated Image")
    plt.axis('off') # 軸を非表示に
    plt.show()


def main() -> None:
    """
    GANの学習と画像生成を実行するメイン処理
    Args:
        None
    Returns:
        None
    """
    # 設定をインスタンス化
    config = Config()
    
    # 画像保存ディレクトリを作成
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # モデルをインスタンス化
    G = Generator(z_dim=config.Z_DIM, image_size=config.IMAGE_SIZE)
    D = Discriminator(image_size=config.IMAGE_SIZE)
    
    # --- 学習済みモデルの読み込み or 新規学習 ---
    if os.path.exists(config.GENERATOR_PATH) and os.path.exists(config.DISCRIMINATOR_PATH):
        G.load_state_dict(torch.load(config.GENERATOR_PATH))
        D.load_state_dict(torch.load(config.DISCRIMINATOR_PATH))
        print("学習済みモデルを読み込みました。学習をスキップします。")
    else:
        print("学習済みモデルが見つかりません。新規に学習を開始します。")
        # データセットを準備
        train_loader = prepare_dataloader(config.BATCH_SIZE)
        # 学習を実行
        train_gan(G, D, train_loader, config)

    # 最終的な生成画像の表示
    generate_and_show_image(G, config)


if __name__ == "__main__":
    main()