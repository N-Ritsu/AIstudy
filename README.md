# AIstudy
### AIエンジニアにむけての学習用リポジトリ

具体的なプログラムの詳細についてはそれぞれのREADMEをご覧ください。  
学習用のため、復習しやすいように敢えてコメント文を冗長に記述しております。ご了承ください。

以下、プログラム一覧です。

### ・bitcoin_crypto_analyzer
公開APIから取得したビットコインの過去の価格データに基づき、市場のトレンドとリスクを分析・可視化するデータ分析ツールです。

### ・bluesky_ai_cat_feed
Blueskyのタイムラインから、猫に関する画像を投稿しているものをAIによって判別し、抽出するフィード作成ツールです。

### ・book_scraper
books.toscrape.comの各カテゴリページから、本の名前と値段のデータをスクレイピングするプログラムです。  
※books.toscrape.comはスクレイピング学習用サイトのため大丈夫だが、他サイトでスクレイピングする際は利用規約とrobots.textを必ず確認すること。

### ・cheat_detector
ランダムな数値生成により擬似的に生成したプレイヤーデータから、チートが疑われるデータを検出するプログラムです。

### ・corporate_qa_chatbot
RAG技術を使用して、指定されたPDFドキュメント群に基づいて質問に回答するチャットボットシステムです。

### ・deep_q_network_for_cartpole
深層強化学習の基本的なアルゴリズムであるDQN（Deep Q-Network）を実装し、古典的な制御問題であるCartPole環境を攻略するAIエージェントを学習させるプログラムです。

### ・emotion_detector_by_bert
Hugging Faceで公開されている事前学習済みモデルを利用して、日本語テキストの感情を分析し、ブラウザ上・コマンドラインツール・API呼び出しという３種のインタフェースで応答するNLPツールです。

### ・gradient_boosting_for_customer_churn_prediction
通信会社の顧客データセットを用いて、顧客がサービスを解約するかどうかを予測するプログラムです。

### ・image_generator_by_mnist
手書き数字のデータセット(MNIST)を用いて、変分オートエンコーダ(VAE)という深層学習モデルを学習させることで、新たな手書き数字の画像を生成するプログラムです。

### ・movielens_recommender_by_cosine_similarity
MovieLens 100Kデータセットを利用し、ユーザーベース協調フィルタリングとコサイン類似度を用いて、個々のユーザーにおすすめの映画を推薦するプログラムです。

### ・news_topic_analyzer_by_lda
大量のニュース記事データを、トピックモデル(LDA)を用いて内容ごとに自動で分類するプログラムです。

### ・q_learning_frozenlake
Q学習アルゴリズムを用いて、gymnasiumライブラリのFrozenLake環境でエージェントを学習させ、効率的な経路探索能力を獲得させるプログラムです。

### ・rag_chatbot
LLMに外部のテキストファイルから知識を参照させ、その情報に基づいてユーザーの質問に回答するRAGシステムです。

### ・similar_image_search_by_cnn
事前学習済みのCNN（畳み込みニューラルネットワーク）モデルを用いて、指定した画像と似ている画像をデータセットの中から検索するプログラムです。

### ・simple_face_auth
基準となる顔写真と複数の未知の顔写真に対しそれぞれ顔認証を行い、未知の顔写真に写る人物が既知の人物と一致するかAIによって判定するプログラムです。 

### ・stock_price_predictor
PyTorchで構築したLSTMモデルを用いて、過去の株価データから未来の株価を予測するプログラムです。

### ・t5_text_summarizer
Transformerベースの言語モデルであるT5を用いて、日本語の長文テキストを要約し、また、その中から話題のキーワードを抽出するコマンドラインツールです。

### ・time_series_anomaly_detector_by_autoencoder
PyTorchで構築したLSTM Autoencoderモデルを用いて、時系列データの中から異常を検出するプログラムです。

### ・vanilla_gan_painter
PyTorchを用いてGAN（敵対的生成ネットワーク）を構築し、手書き数字データセットを学習させることで、新たな手書き数字画像を生成するプログラムです。

### ・whisper_cli_transcriber
OpenAIの音声認識モデルWhisperを利用して、音声ファイルから文字起こしを行うコマンドラインツールです。

### ・yolov8_object_detector
事前学習済みの物体検出モデル"YOLOv8"とOpenCVを利用して、静止画や動画ファイルから物体を検出・追跡するプログラムです。 