# Book Scraper

## 概要  
books.toscrape.comの各カテゴリページから、本の名前と値段のデータをスクレイピングするプログラムです。
AIエンジニアを目指すにあたり、pythonでのスクレイピング経験を積むために、このプロジェクトを開発しました。

## 実行結果  
![book_scraper実行結果](./book_scraper.gif)

## 主な機能  
- books.toscrape.comにアクセスし、トップページのHTMLから全カテゴリのURLを抽出
- 取得した各カテゴリを巡回して、全ての書籍のタイトルと価格を取得
- 取得した本のデータをCSVファイルとして出力
- スクレイピング状況をコンソールに出力

## 使用技術  
・言語  
  Python  
・ライブラリ   
  requests  
  BeautifulSoup4  
  csv  

## 導入・実行方法  
### 1. リポジトリをクローン  
```bash
git clone https://github.com/N-Ritsu/AIstudy.git  
cd AIstudy/book_scraper
```
### 2. 必要なライブラリをインストール
```bash
pip install -r requirements.txt
```
### 3. プログラムを実行
```bash
python book_scraper.py
```

## 開発を通して  
私はこのBook Scraperの開発が、初めてのスクレイピング経験となりました。
開発で最も苦労したのは、書籍によってクラス名が異なるケースに対応するプログラムの作成です。より安定した親要素からの相対的な位置（CSSセレクタ）でデータを取得する、堅牢なロジックを実装しました。  

この開発を通して、実践的なスクレイピングに対する理解を深め、同時にHTMLの読解力が向上しました。
