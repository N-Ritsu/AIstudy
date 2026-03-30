# Causal Inference Engine

## 概要
Eコマースサイトにおけるメールクーポン施策が、セレクションバイアスに影響されない、顧客の購買額に与える真の効果を因果推論の手法を用いて測定するプロジェクトです。  
AIエンジニアを目指すにあたり、単なる相関関係と因果関係の違いを深く理解し、ビジネスの意思決定を誤らせるセレクションバイアスを除去する統計的手法を実践するために開発しました。

## 実行結果
サマリープロット
![causal_inference_engine実行結果](./causal_inference_engine_result.png)

## 主な機能
- 年齢、訪問回数、性別といった属性を持つ擬似的な顧客データを生成
- create_causal_dataset.pyにて、特定の顧客層に施策が偏るセレクションバイアスを意図的にデータ内に再現
- 介入群（クーポン送付）と対照群（クーポンなし）の単純な平均購買額の差を計算し、バイアスの影響を可視化
- scikit-learnのLogisticRegressionを用い、顧客の属性からクーポンが送られる確率（傾向スコア）を推定
- scikit-learnのNearestNeighborsを用い、傾向スコアが近い顧客同士を効率的にマッチングさせる傾向スコアマッチングを実装
- マッチング後のデータを用いて、バイアスが除去された施策の純粋な効果（平均処置効果, ATE）を算出

## 使用技術
・言語
  Python
・ライブラリ
  pandas
  scikit-learn
  numpy
  statsmodels

## 導入・実行方法
### 1. リポジトリをクローン
```bash
git clone https://github.com/N-Ritsu/AIstudy.git
cd AIstudy/causal_inference_engine
```
### 2. Conda仮想環境の構築と有効化
```bash
conda create --name causal_inference_engine_env python=3.10 -y
conda activate causal_inference_engine_env
```
### 3. 必要なライブラリをインストール
```bash
pip install -r requirements.txt
```
### 4 . プログラムを実行
```bash
python causal_inference_engine.py
```

## 開発を通して
私はこのcausal_inference_engineの開発が、初めての因果推論の実装経験となりました。  
結果を見ると、対照群の平均購買額がマッチ前に比べて高くなっていることが分かり、同じような傾向スコアの顧客にフォーカスすることで、クーポンが送られやすい"元から購買意欲のある顧客"が対照群とされたことが考えられます。このことから、やはりもともと購買意欲に優先的にクーポンが送られているという事実が確認できます。  
そして、実際の推定効果も、マッチ前は真の値より14.15の差がありましたが、傾向スコアマッチング後は0.23の差となり、劇的に改善されました。  
このプロジェクトを通して、AIエンジニアとして分析を行う上で、一見正しいように見えて実は隠れた因果関係によって真の値の推定ができていないという事例の存在と、その対処の重要性について知ることができました。