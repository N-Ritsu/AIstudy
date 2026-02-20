from flask import Flask, render_template, request, jsonify
from emotion_detector_by_bert import SentimentAnalyzer

app = Flask(__name__)
analyzer = SentimentAnalyzer()

# HTMLページ用のルーティング
# '/': トップページにアクセスした際
# methods=['GET', 'POST']: GETリクエストまたはPOSTリクエストが来た際
# GET: ブラウザを開く → index.htmlを表示
# POST: フォームにテキストを入力して送信 → 感情分析の結果を表示
@app.route('/', methods=['GET', 'POST'])
def index() -> str:
    """
    ユーザーがテキストを入力して感情分析を行うためのHTMLページのルーティング
    Args:
        None
    Returns:
        str: HTMLテンプレートをレンダリングして返す
    """
    sentiment_result = ""
    input_text = ""
    # POSTリクエストの場合、感情分析を実行
    if request.method == 'POST':
        input_text = request.form['text']
        if input_text:
            sentiment_result = analyzer.analyze(input_text)
    # render_template: HTMLテンプレートに動的なデータを埋め込んで返す
    return render_template('index.html', sentiment=sentiment_result, text=input_text)

# JSON API用のエンドポイントを定義
# '/api/analyze'に、'POST'リクエストが来た際に呼び出される
@app.route('/api/analyze', methods=['POST'])
def api_analyze() -> jsonify:
    """
    JSONリクエストを受け取り、感情分析結果をJSONで返すAPI
    Args:
        None
    Returns:
        JSON形式で、入力テキストと分析結果を含む辞書を返す
    """
    # リクエストがJSON形式かチェック
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # JSONデータから 'text' を取得
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    # 感情分析を実行
    sentiment = analyzer.analyze(text)

    # 結果をJSON形式で返す
    return jsonify({"text": text, "sentiment": sentiment})


if __name__ == '__main__':
    # Webサーバーを起動する
    app.run()