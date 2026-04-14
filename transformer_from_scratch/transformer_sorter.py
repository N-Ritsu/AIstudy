import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple, List

# 設定クラス
class Config:
    """
    学習やモデルに関する設定を管理するクラス
    """
    # データ関連
    VOCAB_SIZE: int = 20 # 0から19までの数字を扱う
    SEQ_LEN: int = 10 # 数列の長さ
    NUM_SAMPLES: int = 10000 # 学習データ数

    # トークン定義
    PAD_TOKEN: int = 0
    SOS_TOKEN: int = VOCAB_SIZE
    EOS_TOKEN: int = VOCAB_SIZE + 1
    
    # モデルのハイパーパラメータ
    EMB_SIZE: int = 32 # 埋め込みベクトルの次元数 (d_model)
    NHEAD: int = 4 # Multi-head Attentionのヘッド数
    FFN_HID_DIM: int = 64 # FeedForward層の中間次元数
    NUM_ENCODER_LAYERS: int = 2 # エンコーダの層数
    NUM_DECODER_LAYERS: int = 2 # デコーダの層数
    
    # 学習関連
    BATCH_SIZE: int = 128
    NUM_EPOCHS: int = 20
    LEARNING_RATE: float = 0.001

# デバイスの設定 (GPUがあれば使い、なければCPUを使う)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ生成
class SortDataset(Dataset):
    """
    数列ソートタスク用のカスタムデータセット
    """
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int) -> None:
        """
        データセットの初期化
        Args:
            num_samples(int): データセット内のサンプル数
            seq_len(int): 数列の長さ
            vocab_size(int): 数字の語彙サイズ (0からvocab_size-1までの数字を使用)
        Returns:
            None
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        # 語彙には数字の他にPADトークンも含まれるため、1からスタート
        self.vocab_start_idx = 1
        self.vocab_end_idx = vocab_size - 1

    def __len__(self) -> int:
        """
        データセットのサンプル数を返す
        Args:
            None
        Returns:
            num_samples: データセット内のサンプル数
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスのサンプルを返す
        Args:
            idx(int): サンプルのインデックス
        Returns:
            src: 入力シーケンス
            tgt: 出力シーケンス
        """
        # ランダムな数列を生成 (1からVOCAB_SIZE-1までの数字)
        sequence = np.random.randint(self.vocab_start_idx, self.vocab_end_idx + 1, self.seq_len)
        sorted_sequence = np.sort(sequence)
        
        # SOS/EOSトークンを追加
        # SOS/EOSトークンとは、シーケンスの開始と終了を示す特別なトークン
        # [SOS, 5, 2, 9, EOS] → [SOS, 2, 5, 9, EOS]
        src = torch.tensor([Config.SOS_TOKEN] + list(sequence) + [Config.EOS_TOKEN], dtype=torch.long) # 入力データ
        tgt = torch.tensor([Config.SOS_TOKEN] + list(sorted_sequence) + [Config.EOS_TOKEN], dtype=torch.long) # 正解データ
        
        return src, tgt

# モデル構築 (部品ごとの実装)
class PositionalEncoding(nn.Module):
    """
    位置情報をベクトルに加えるクラス
    """
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000) -> None:
        """
        位置エンコーディングの初期化
        位置エンコーディング: 単語の順序をTransformerが理解できるようにするための方法
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        - pos: シーケンス内での単語の位置
        - i: 埋め込みベクトルの次元のインデックス
        - d_model: 埋め込みベクトルの総次元数(各単語を何次元のベクトルで表すか "5" → [0.12, -0.45, 0.88, 0.03])
        Args:
            emb_size(int): 埋め込みベクトルの次元数
            dropout(float): ドロップアウト率
            maxlen(int): 位置エンコーディングの最大長さ
        Returns:
            None
        """
        super(PositionalEncoding, self).__init__()
        # 10000^(2i/d_model)部分
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000.0) / emb_size)
        # 位置のインデックス(pos)を生成
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # 最終的な位置エンコーディング行列を格納するための、中身が全て0の空のテンソルを作成
        pos_embedding = torch.zeros((maxlen, emb_size))
        # 偶数番目の次元に対し、sin(pos / 10000^(2i/d_model))を計算
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        # 奇数番目の次元に対し、cos(pos / 10000^(2i/d_model))を計算
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # 位置エンコーディング行列を(batch_size, seq_len, emb_size)の形にするため、次元を追加
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        """
        位置エンコーディングをトークン埋め込みに加える
        Args:
            token_embedding(torch.Tensor): トークンの埋め込みベクトル (batch_size, seq_len, emb_size)
        Returns:
            torch.Tensor: 位置エンコーディングが加えられたトークン埋め込み (batch_size, seq_len, emb_size)
        """
        # 単語のベクトルと位置エンコーディングを組み合わせることで、単語と位置の因果関係を理解できるようにセットにする
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    """
    トークンを埋め込みベクトルに変換するクラス
    例: "5" → [0.12, -0.45, 0.88, 0.03]
    """
    def __init__(self, vocab_size: int, emb_size: int) -> None:
        """
        トークン埋め込みの初期化
        Args:
            vocab_size(int): 語彙サイズ (0からvocab_size-1までの数字を使用)
            emb_size(int): 埋め込みベクトルの次元数
        Returns:
            None
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size) # 0からvocab_size-1までの数字をemb_size次元のベクトルに変換する
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # ソートする配列内のそれぞれの数字に対して、.embeddingで作成したベクトルを対応させる
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    """
    Transformerモデル全体を定義するクラス
    """
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int, vocab_size: int, ffn_hid_dim: int, dropout: float = 0.1) -> None:
        """
        Transformerモデルの初期化
        Args:
            num_encoder_layers(int): エンコーダの層数
            num_decoder_layers(int): デコーダの層数
            emb_size(int): 埋め込みベクトルの次元数 (d_model)
            nhead(int): Multi-head Attentionのヘッド数
            vocab_size(int): 語彙サイズ (0からvocab_size-1までの数字を使用)
            ffn_hid_dim(int): FeedForward層の中間次元数
            dropout(float): ドロップアウト率
        Returns:
            None
        """
        super(Seq2SeqTransformer, self).__init__()
        # Transformerモデルの定義
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ffn_hid_dim,
            dropout=dropout,
            batch_first=True # (N, S, E)形式の入力を受け付ける
        )
        # 最終的な出力層（emb_size次元のベクトルを用いて、各数字である確率を格納したvocab_size次元のベクトルに変換）
        self.generator = nn.Linear(emb_size, vocab_size)
        # トークン埋め込み層と位置エンコーディング層
        self.src_tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(vocab_size, emb_size)
        # ベクトルに位置情報を加える
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: torch.Tensor, trg: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor,
                src_padding_mask: torch.Tensor, tgt_padding_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Transformerの順伝播処理
        - パディングマスク: シーケンス内で、穴埋めのために格納された0の値に対し、モデルが学習に含めないように見えなくする
        - シーケンスマスク: まだ予測を行っていないトークンを見えないようにして、カンニングを防ぐ
        Args:
            src: ソース系列 (batch_size, src_seq_len)
            trg: ターゲット系列 (batch_size, tgt_seq_len)
            src_mask: ソース系列のマスク (src_seq_len, src_seq_len)
            tgt_mask: ターゲット系列のマスク (tgt_seq_len, tgt_seq_len)
            src_padding_mask: ソース系列のパディングマスク (batch_size, src_seq_len)
            tgt_padding_mask: ターゲット系列のパディングマスク (batch_size, tgt_seq_len)
            memory_key_padding_mask: メモリキーのパディングマスク (batch_size, src_seq_len)
        Returns:
            torch.Tensor: Transformerの出力 (batch_size, tgt_seq_len, vocab_size)
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    未来のトークンマスクを生成
    Args:
        sz(int): マスクのサイズ (通常はターゲット系列の長さ) 
    Returns:
        torch.Tensor: 未来のトークンをマスクするためのマスク (sz, sz)  
    """
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    学習時に必要な各種マスクを生成
    Args:
        src: ソース系列 (batch_size, src_seq_len)
        tgt: ターゲット系列 (batch_size, tgt_seq_len)
    Returns:
        src_mask: ソース系列のマスク (src_seq_len, src_seq_len)
        tgt_mask: ターゲット系列のマスク (tgt_seq_len, tgt_seq_len)
        src_padding_mask: ソース系列のパディングマスク (batch_size, src_seq_len)
        tgt_padding_mask: ターゲット系列のパディングマスク (batch_size, tgt_seq_len)
    """
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == Config.PAD_TOKEN)
    tgt_padding_mask = (tgt == Config.PAD_TOKEN)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# --- 3. 学習 ---
def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                dataloader: DataLoader) -> float:
    """
    1エポック分の学習を実行
    Args:
        model(nn.Module): 学習するモデル
        optimizer(torch.optim.Optimizer): モデルのパラメータを更新するオプティマイザ
        criterion(nn.Module): 損失関数
        dataloader(DataLoader): 学習データを提供するDataLoader
    Returns:
        float: 1エポック分の平均損失  
    """
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        
        # デコーダへの入力 (最後のトークンを除いた部分) を作成 → 教師強制
        tgt_input = tgt[:, :-1]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        # 順伝播
        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()
        
        # モデルの出力と正解データを比較
        tgt_out = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 推論と評価
def sort_sequence(model: nn.Module, src_sequence: List[int]) -> List[int]:
    """
    学習済みモデルを使って数列をソート
    Args:
        model(nn.Module): 学習済みのTransformerモデル
        src_sequence(List[int]): ソートしたい数列
    Returns:
        List[int]: ソートされた数列
    """
    model.eval()
    src = torch.tensor([Config.SOS_TOKEN] + src_sequence + [Config.EOS_TOKEN], dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_mask = (torch.zeros(src.shape[1], src.shape[1])).type(torch.bool).to(DEVICE)
    
    with torch.no_grad(): # 推論時は勾配計算を無効化
        memory = model.encode(src, src_mask)
    
    # SOSトークンから生成を開始
    ys = torch.ones(1, 1).fill_(Config.SOS_TOKEN).type(torch.long).to(DEVICE)
    for _ in range(Config.SEQ_LEN + 2): # EOSトークンも考慮して少し長めに
        with torch.no_grad(): # 推論時は勾配計算を無効化
            memory = memory.to(DEVICE)
            tgt_mask = (generate_square_subsequent_mask(ys.size(1))).to(DEVICE)
            
            out = model.decode(ys, memory, tgt_mask)
            
            # (バッチサイズ, シーケンス長, 埋め込みサイズ) の最後の時刻の出力を取得
            last_time_step_out = out[:, -1]
            
            prob = model.generator(last_time_step_out)

            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word == Config.EOS_TOKEN:
                break
            
    # SOSトークンを除外して返す
    return ys.flatten().tolist()[1:]

    
# メイン処理
def main() -> None:
    """
    自作Transformerのメイン関数
    Args:
        None
    Returns:
        None
    """
    print(f"使用デバイス: {DEVICE}")

    # データセットとDataLoaderの準備
    dataset = SortDataset(Config.NUM_SAMPLES, Config.SEQ_LEN, Config.VOCAB_SIZE)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # ボキャブラリサイズには特殊トークンも含む
    effective_vocab_size = Config.VOCAB_SIZE + 2

    # モデル、損失関数、オプティマイザの初期化
    model = Seq2SeqTransformer(
        Config.NUM_ENCODER_LAYERS, Config.NUM_DECODER_LAYERS,
        Config.EMB_SIZE, Config.NHEAD, effective_vocab_size,
        Config.FFN_HID_DIM
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_TOKEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    print("\n--- 学習開始 ---")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(model, optimizer, criterion, dataloader)
        end_time = time.time()
        print(f"エポック: {epoch}, 損失: {train_loss:.3f}, "
              f"エポック時間: {(end_time - start_time):.3f}s")
    print("--- 学習完了 ---\n")

    # --- 推論テスト ---
    print("--- 推論テスト ---")
    test_sequences = [
        [5, 2, 8, 1, 9, 4, 7, 3, 6, 0],
        [18, 15, 12, 10, 1, 3, 19, 7, 5, 11],
        [4, 4, 2, 9, 1, 1, 8, 8, 3, 5]
    ]

    for seq in test_sequences:
        # 入力からPADトークン(0)を除外して渡す
        input_seq = [x for x in seq if x != Config.PAD_TOKEN]
        sorted_result_with_eos = sort_sequence(model, input_seq)
        
        # EOSトークンを除外して表示
        sorted_result = [x for x in sorted_result_with_eos if x != Config.EOS_TOKEN]
        
        print(f"入力: {input_seq}")
        print(f"モデル出力: {sorted_result}")
        print(f"正解: {sorted(input_seq)}")
        print("-" * 20)

if __name__ == "__main__":
    import time
    main()