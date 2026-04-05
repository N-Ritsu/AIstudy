# AIstudy
### AIエンジニアにむけての学習用リポジトリ

具体的なプログラムの詳細についてはそれぞれのREADMEをご覧ください。  
また、各プログラムのライブラリ・モデル・アルゴリズムの採用理由はそれぞれのdesign_notes.txtをご覧ください。  
学習用のため、復習しやすいように敢えてコメント文を冗長に記述しております。ご了承ください。

以下、プログラム一覧です。

### ・arima_sarima_comparison
ARIMAモデルとSARIMAモデルでローリング予測を行い、性能を比較するプロジェクトです。

### ・bitcoin_crypto_analyzer
公開APIから取得したビットコインの過去の価格データに基づき、市場のトレンドとリスクを分析・可視化するデータ分析ツールです。

### ・bluesky_ai_cat_feed
Blueskyのタイムラインから、猫に関する画像を投稿しているものをAIによって判別し、抽出するフィード作成ツールです。

### ・book_scraper
books.toscrape.comの各カテゴリページから、本の名前と値段のデータをスクレイピングするプログラムです。  
※books.toscrape.comはスクレイピング学習用サイトのため大丈夫だが、他サイトでスクレイピングする際は利用規約とrobots.textを必ず確認すること。

### ・causal_inference_engine
Eコマースサイトにおけるメールクーポン施策が、セレクションバイアスに影響されない、顧客の購買額に与える真の効果を因果推論の手法を用いて測定するプロジェクトです。

### ・cheat_detector
ランダムな数値生成により擬似的に生成したプレイヤーデータから、チートが疑われるデータを検出するプログラムです。

### ・classification_model_comparison
ロジスティック回帰、決定木、ナイーブベイズ、SVM、ランダムフォレストといった基礎的な機械学習モデルの性能と解釈性を多角的に比較・可視化するプロジェクトです。

### ・corporate_qa_chatbot
RAG技術を使用して、指定されたPDFドキュメント群に基づいて質問に回答するチャットボットシステムです。

### ・deep_q_network_for_cartpole
深層強化学習の基本的なアルゴリズムであるDQN（Deep Q-Network）を実装し、古典的な制御問題であるCartPole環境を攻略するAIエージェントを学習させるプログラムです。

### ・emotion_detector_by_bert
Hugging Faceで公開されている事前学習済みモデルを利用して、日本語テキストの感情を分析し、ブラウザ上・コマンドラインツール・  flaskを用いたAPI呼び出しという３種のインタフェースで応答するNLPツールです。

### ・feature_engineering_pipeline_and_pytest
scikit-learnのPipelineとColumnTransformerを用いて、機械学習の前処理パイプラインを構築するプロジェクトです。さらに、そのパイプラインが意図通りに動作することを保証するため、pytestを用いた単体テストを実装しています。

### ・gnn_cora_classifier
グラフニューラルネットワーク（GNN）の一種であるGCNを用い、学術論文の引用関係グラフから各論文の専門分野を分類するプロジェクトです。

### ・gradient_boosting_for_customer_churn_prediction
通信会社の顧客データセットを用いて、顧客がサービスを解約するかどうかを予測するプログラムです。

### ・hybrid_recommender_analysis
協調フィルタリング、コンテンツベース推薦、そして両者を組み合わせたハイブリッド推薦という、3つの異なるアプローチを実装・比較するプロジェクトです。

### ・image_generator_by_mnist
手書き数字のデータセット(MNIST)を用いて、変分オートエンコーダ(VAE)という深層学習モデルを学習させることで、新たな手書き数字の画像を生成するプログラムです。

### ・imbalanced_data_sampler_comparison
不均衡なデータセットにおける分類問題で、SMOTEを用いたオーバーサンプリング/アンダーサンプリングがどのような効果をもたらすかを比較・分析するプロジェクトです。

### ・matrix_factorization_comparison
代表的な行列因子分解であるNaive SVD・NMF・SVD++・FunkSVDという4つの手法を、異なる特性を持つデータセットで比較・評価し、その予測精度と解釈性を分析するプロジェクトです。

### ・ml_pipeline_project
Apache Airflowを使用し、機械学習モデルの学習・評価・デプロイといった一連のプロセスを自動化するMLパイプラインを構築するプロジェクトです。

### ・movielens_recommender_by_cosine_similarity
MovieLens 100Kデータセットを利用し、ユーザーベース協調フィルタリングとコサイン類似度を用いて、個々のユーザーにおすすめの映画を推薦するプログラムです。

### ・news_topic_analyzer_by_lda
大量のニュース記事データを、トピックモデル(LDA)を用いて内容ごとに自動で分類するプログラムです。

### ・pyucm_analyzer
状態空間モデル（構造時系列モデル、BSM）を用いて、最新のCO2濃度時系列データを分析し、その構造を解釈するプロジェクトです。

### ・q_learning_frozenlake
Q学習アルゴリズムを用いて、gymnasiumライブラリのFrozenLake環境でエージェントを学習させ、効率的な経路探索能力を獲得させるプログラムです。

### ・rag_chatbot
LLMに外部のテキストファイルから知識を参照させ、その情報に基づいてユーザーの質問に回答するRAGシステムです。

### ・regression_model_comparison
8つの主要な回帰モデル（線形回帰、Ridge、Lasso、決定木、SVR、ランダムフォレスト、勾配ブースティング、LightGBM）の性能と挙動を、特性の異なる4つのデータセットを用いて網羅的に比較・可視化するプロジェクトです。

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

### ・unsupervised_anomaly_detection_tracker_with_mlflow
Kaggleで公開されているクレジットカード取引データセットを用いて、不正利用取引を検知する機械学習プログラムです。  
3つの異なる教師なし学習モデル（Isolation Forest, OneClassSVM, Local Outlier Factor）に対し、それぞれハイパーパラメータチューニングを行いMLflowで管理、そして3つのモデルの評価を組み合わせるアンサンブル手法により高い精度を誇ります。

### ・vanilla_gan_painter
PyTorchを用いてGAN（敵対的生成ネットワーク）を構築し、手書き数字データセットを学習させることで、新たな手書き数字画像を生成するプログラムです。

### ・whisper_cli_transcriber
OpenAIの音声認識モデルWhisperを利用して、音声ファイルから文字起こしを行うコマンドラインツールです。

### ・xai_for_housing_regression
XGBoostを用いてカリフォルニアの住宅価格を予測し、その予測根拠をSHAPライブラリを用いて可視化するプログラムです。  

### ・yolov8_object_detector
事前学習済みの物体検出モデル"YOLOv8"とOpenCVを利用して、静止画や動画ファイルから物体を検出・追跡するプログラムです。 