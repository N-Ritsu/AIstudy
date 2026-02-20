from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    """
    Hugging Faceの事前学習済みモデルを使って、テキストの感情分析を行うクラス
    """
    def __init__(self, model_name="koheiduck/bert-japanese-finetuned-sentiment"):
        """
        クラスのインスタンスが作成されたときに、モデルとトークナイザを読み込む
        Args:
            model_name (str): Hugging Faceのモデル名。デフォルトは "koheiduck/bert-japanese-finetuned-sentiment"。
        Returns:
            None
        """
        print("▶ SentimentAnalyzerを初期化しています...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print("モデルとトークナイザの読み込みが完了しました。")
        except Exception as e:
            print(f"モデル読み込み中にエラーが発生しました: {e}")
            # エラーが発生した場合は、プログラムが続行できないように例外を発生させる
            raise

    def analyze(self, text: str) -> str:
        """
        入力されたテキストの感情を分析し、'positive'または'negative'または'neutral'の文字列を返す
        Args:
            text (str): 分析したい日本語のテキスト
        Returns:
            str: 分析結果のラベル ('positive' or 'negative' or 'neutral')
        """
        # テキストをトークナイズ(トークンに分割)
        inputs = self.tokenizer(text, return_tensors="pt")

        # モデルで推論
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 最もスコアが高いクラスIDを取得
        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()

        # クラスIDをラベル名に変換して返す
        return self.model.config.id2label[predicted_class_id]