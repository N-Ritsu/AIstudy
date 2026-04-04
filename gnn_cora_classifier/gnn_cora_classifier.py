import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import japanize_matplotlib

LEARNING_RATE = 0.01 # 学習率
WEIGHT_DECAY = 5e-4 # L2正則化の強さ（過学習を防ぐ）
EPOCHS = 200 # 学習エポック数
TSNE_RANDOM_STATE = 42 # t-SNEの乱数シード


# --- データセットの準備 ---
def load_dataset() -> Tuple[Data, int, int]:
    """
    Coraデータセットをロードし、その概要情報を表示。
    このデータセットは、学術論文の引用ネットワークで構成されている。
    - ノード: 論文
    - エッジ: 論文間の引用関係
    - 特徴量: 各論文の単語ベクトル (BoW)
    - ラベル: 論文の専門分野 (7クラス)
    Args:
        None
    Returns:
        Tuple[Data, int, int]: 
            - data: ロードされたグラフデータオブジェクト。
            - num_node_features: ノードの特徴量の次元数。
            - num_classes: 分類対象のクラス数。
    """
    # PyTorch GeometricのPlanetoidクラスを使い、Coraデータセットをダウンロード・ロード。
    # root: データセットの保存先ディレクトリ
    # name: データセット名
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0] # Coraデータセットは単一のグラフで構成されている。

    # データセットの統計情報を表示
    print("\nデータセット概要:")
    print(f"  データセット名: {dataset.name}")
    print(f"  ノード数 (論文数): {data.num_nodes}")
    print(f"  エッジ数 (引用関係数): {data.num_edges}")
    print(f"  ノード特徴量の次元数: {dataset.num_node_features}")
    print(f"  クラス数 (専門分野の数): {dataset.num_classes}")
    print(f"  学習用ノード数: {data.train_mask.sum()}")
    print(f"  検証用ノード数: {data.val_mask.sum()}")
    print(f"  テスト用ノード数: {data.test_mask.sum()}")
    print("-" * 30 + "\n")
    return data, dataset.num_node_features, dataset.num_classes


# --- GCNモデルの定義 ---
class GCN(torch.nn.Module):
    """
    2層のGraph Convolutional Network (GCN) モデル。
    Attributes:
        conv1 (GCNConv): 1層目のGCNレイヤ。ノード特徴量を16次元の中間表現に変換します。
        conv2 (GCNConv): 2層目のGCNレイヤ。中間表現を各クラスの予測スコアに変換します。
    """
    def __init__(self, num_node_features: int, num_classes: int):
        """
        GCNモデルのレイヤを初期化する。
        Args:
            num_node_features (int): 入力されるノード特徴量の次元数。
            num_classes (int): 分類するクラスの数。
        Returns:
            None
        """
        super(GCN, self).__init__()
        # 1層目のGCNレイヤ: 入力特徴量 -> 16次元の中間表現
        self.conv1 = GCNConv(num_node_features, 16)
        # 2層目のGCNレイヤ: 16次元の中間表現 -> クラス数の次元（各クラスへの所属確率）
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        モデルの順伝播処理を定義。
        Args:
            x (torch.Tensor): 全ノードの特徴量テンソル (形状: [num_nodes, num_node_features])。
            edge_index (torch.Tensor): グラフの接続情報を示すエッジリスト (形状: [2, num_edges])。
        Returns:
            torch.Tensor: 各ノードのクラス予測対数確率 (形状: [num_nodes, num_classes])。
        """
        # 1層目: GCN層で近傍ノードの情報を集約。隣接ノードの特徴量を受け取り、それらを集約して16次元の特徴量に変換。
        x = self.conv1(x, edge_index)
        x = F.relu(x) # 活性化関数ReLUを適用して非線形変換
        # 過学習を防ぐためにDropoutを適用
        x = F.dropout(x, training=self.training)
        # 2層目: 再度GCN層で情報を集約
        # 隣接ノードの、さらに隣接ノードの特徴量も集約される(隣接ノードの隣接ノード)ため、より広い範囲の情報を考慮してクラス予測が行われる。
        x = self.conv2(x, edge_index)
        
        # 出力を対数確率に変換 (NLLLossと組み合わせて使用)
        return F.log_softmax(x, dim=1)


# --- 学習処理の定義 ---
def train(model: GCN, data: Data, optimizer: Optimizer, criterion: _Loss) -> float:
    """
    モデルを1エポック分学習させる。
    Args:
        model (GCN): 学習対象のGCNモデル。
        data (Data): グラフデータ。
        optimizer (Optimizer): 最適化アルゴリズム。
        criterion (_Loss): 損失関数。
    Returns:
        float: 計算された学習損失の値。
    """
    model.train()  # モデルを学習モードに設定 (Dropoutなどを有効化)
    optimizer.zero_grad()  # 前のステップで計算された勾配をリセット
    
    # モデルに全ノードの特徴量とエッジ情報を入力し、順伝播
    out = model(data.x, data.edge_index)
    
    # 学習用ノード（train_mask=True）の予測結果と正解ラベルのみを使って損失を計算
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()  # 損失から勾配を計算 (誤差逆伝播)
    optimizer.step()  # 計算された勾配に基づいてモデルのパラメータを更新
    
    return loss.item()


# --- 評価関数の定義 ---
def evaluate(model: GCN, data: Data, mask: torch.Tensor) -> float:
    """
    指定されたデータセット（検証用またはテスト用）でモデルの精度を評価。
    Args:
        model (GCN): 評価対象の学習済みモデル。
        data (Data): グラフデータ。
        mask (torch.Tensor): 評価対象のノードを示すブール型マスク (例: data.val_mask)。
    Returns:
        float: 正解率 (Accuracy)。
    """
    model.eval()  # モデルを評価モードに設定 (Dropoutなどを無効化)
    with torch.no_grad():  # 勾配計算を無効化し、メモリ消費を抑え、計算を高速化
        out = model(data.x, data.edge_index)
        # 最も確率の高いクラスを予測結果とする
        pred = out.argmax(dim=1)
        
        # マスクで指定されたノードの予測結果と正解ラベルを比較
        correct = pred[mask] == data.y[mask]
        # 正解率を計算
        accuracy = int(correct.sum()) / int(mask.sum())
        
    return accuracy


# --- 結果の可視化 ---
def visualize_embeddings(model: GCN, data: Data, title: str, filename: str) -> None:
    """
    t-SNEを用いて高次元のノード埋め込みを2次元に圧縮し可視化。
    Args:
        model (GCN): 可視化対象のGCNモデル（学習前または学習後）。
        data (Data): グラフデータ。
        title (str): グラフのタイトル。
        filename (str): 保存する画像ファイル名。
    Returns:
        None
    """
    print(f"t-SNEによる可視化を開始します... (タイトル: {title})")
    model.eval() # モデルを評価モードに設定
    with torch.no_grad():
        # GCNモデルからノードの埋め込み（最終層の出力）を取得
        embeddings = model(data.x, data.edge_index).cpu().numpy()

    # t-SNEを用いて、高次元の埋め込みベクトルを2次元に圧縮
    tsne = TSNE(n_components=2, random_state=TSNE_RANDOM_STATE, n_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Matplotlibで散布図としてプロット
    plt.figure(figsize=(12, 10))
    # クラスごとに色分けしてプロット
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=data.y.cpu().numpy(), 
        cmap='viridis', # 色のテーマ
        s=15 # マーカーのサイズ
    )
    plt.title(title, fontsize=18)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    
    # 凡例を追加
    class_labels = [f'Class {i}' for i in range(data.y.max().item() + 1)]
    plt.legend(handles=scatter.legend_elements()[0], labels=class_labels, title="Classes")
    
    # グラフをファイルに保存
    plt.savefig(filename)
    print(f"可視化グラフを '{filename}' として保存しました。")
    plt.show() # 画面にグラフを表示
    print("-" * 30 + "\n")


def main() -> None:
    """
    GNNによるノード分類の学習・評価・可視化を行うプログラムのメイン処理。
    Args:
        None
    Returns:
        None
    """
    # 1. データ準備
    print("--- [1/5] データセットの準備 ---")
    data, num_features, num_classes = load_dataset()

    # 2. モデルとオプティマイザの初期化
    print("--- [2/5] モデルの初期化 ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features, num_classes).to(device)
    data = data.to(device) # データも同じデバイスに送る
    
    # 最適化手法としてAdamを使用
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 損失関数としてNegative Log Likelihood Lossを使用 (log_softmaxとセットで使う)
    criterion = torch.nn.NLLLoss()
    
    print(f"使用デバイス: {device}")
    print(f"モデル:\n{model}")
    print("-" * 30 + "\n")

    # 3. 学習前の可視化
    print("--- [3/5] 学習前のノード埋め込みを可視化 ---")
    # 初期状態のモデルでは、ノードはランダムに射影されるため、クラスごとに分離されていないはずです。
    visualize_embeddings(model, data, "学習前のノード埋め込み (t-SNE)", "embeddings_before_training.png")

    # 4. 学習ループ
    print("--- [4/5] モデルの学習 ---")
    for epoch in range(EPOCHS):
        loss = train(model, data, optimizer, criterion)
        
        # 20エポックごとに検証データで精度を評価し、学習の進捗を確認
        if (epoch + 1) % 20 == 0:
            val_acc = evaluate(model, data, data.val_mask)
            print(f'エポック: {epoch+1:03d}, 損失: {loss:.4f}, 検証精度: {val_acc:.4f}')
    print("学習が完了しました。")
    print("-" * 30 + "\n")

    # 5. 最終評価と学習後の可視化
    print("--- [5/5] 最終評価と学習後の可視化 ---")
    # 学習済みモデルをテストデータで最終評価
    test_accuracy = evaluate(model, data, data.test_mask)
    print(f"最終テスト精度: {test_accuracy:.4f}")

    # 学習後のモデルでは、ノードがクラスごとにクラスタを形成していることが期待されます。
    visualize_embeddings(model, data, "学習後のノード埋め込み (t-SNE)", "embeddings_after_training.png")
    
    print("全ての処理が完了しました。")


if __name__ == '__main__':
    main()