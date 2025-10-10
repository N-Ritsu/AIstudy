import requests
import csv
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin # URLを安全に連結するためのライブラリ

def fetch_page(url: str) -> str | None:
  """
  受け取ったurlにアクセスし、そのページのHTMLを取得する
  Args:
    url (str): アクセス先のurl 
  Returns:
    str | None
    - アクセス先のHTMLデータ
    - アクセス中にエラーが発生した際の戻り値
  """
  try:
    # URLにアクセスしてHTMLを取得する
    response = requests.get(url)
    # リンクが見つからなかった際、tryに引っかかるようにエラーを発生させる
    response.raise_for_status()
    response.encoding = response.apparent_encoding # 文字化け防止
    return response.text
  except requests.exceptions.RequestException as e:
    print(f"URLへのアクセス中にエラーが発生しました：{e}")
    return None
   
def get_category_urls(html: str, base_url_for_join: str) -> list[str]:
  """
  受け取ったurlにアクセスし、そのページのHTMLを取得する
  Args:
    html (str): トップページのHTML
    base_url_for_join (str): 相対パスを絶対パスに変換するためのベースurl
  Returns:
    str: 各カテゴリのurl
  """
  soup = BeautifulSoup(html, 'html.parser')
  category_urls = []
  # 'div.side_categoriesの中の、ulの中の、liの中の、aの中身' というCSSセレクタで目的のaタグをすべて取得
  category_links = soup.select('div.side_categories ul li a')
  
  # 最初のリンクは 'Books' 全体なのでスキップ (スライス[1:]を使う)
  for link in category_links[1:]:
      # 相対パスを絶対パスに変換する
      relative_url = link.get('href')
      absolute_url = urljoin(base_url_for_join, relative_url)
      category_urls.append(absolute_url)
      
  print(f"{len(category_urls)}件のカテゴリURLを取得しました。")
  return category_urls
   
def parse_books_info(html: str) -> list[dict]:
  """
  受け取ったhtmlの中から、本の情報を取得する
  Args:
    html (str): 各カテゴリのHTML
  Returns:
    list: 各カテゴリの書籍のタイトルと値段の辞書のリスト
  """
  # response.text を Beautiful Soup で解析できるように変換
  soup = BeautifulSoup(html, 'html.parser')
  books_data = []
  #articlesタグに商品情報が載っているため、それをそれぞれ入手
  articles = soup.find_all('article', class_='product_pod')

  for article in articles:
    # # articleタグの中の<h3>を探し、さらにその中の<a>を入手
    a_tag = article.select_one('h3 a')
    # aタグが見つかったら、.get()を使って安全にtitle属性を取得。なければデフォルト値を代入
    title = a_tag.get('title', '（タイトル不明）') if a_tag else "(タイトル不明)"
    # articleタグの中の<div class = "product_price">を探し、さらにその中の<p>の中の最初の要素(first_of_type)を入手
    price_element = article.select_one('div.product_price p:first-of-type')
    price = price_element.text if price_element else "(値段不明)"
    books_data.append({"title": title, "price": price})
  return books_data

def write_data_by_csv(books_data: list[dict]) -> None:
  """
  書籍情報の辞書のリストを受け取り、CSVファイルに書き出す。
  Args:
    books_data (list[dict]): 全カテゴリの書籍のタイトルと値段の辞書のリスト
  Returns:
    None
  """
  with open('books_data.csv', 'w', encoding='utf-8', newline="") as f:
    fieldnames = ['title', 'price']
    # 辞書を書き込むためのDictWriterを用意
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    # 最初にヘッダーを書き込む
    writer.writeheader()
    # 辞書のリストを一度に書き込む
    writer.writerows(books_data)
    print("books_data.csv にデータを保存しました。")

def main() -> None:
  """
  プログラムのエントリーポイント。全体の処理を実行する。
  Args:
    None
  Returns:
    None
  """
  # 練習サイトのURL
  #値の変わらない"定数"であれば、大文字表記する
  BASE_URL = "https://books.toscrape.com/index.html" # 例として練習サイトを使います
  base_html = fetch_page(BASE_URL)
  all_books_data = []
  if base_html:
    # ステップ1: 全カテゴリのURLを取得
    category_urls = get_category_urls(base_html, BASE_URL)
    # ステップ2: 取得したURLを一つずつ巡回
    for url in category_urls:
      print(f"\nカテゴリページをスクレイピング中: {url}")
      each_category_html = fetch_page(url)
      if each_category_html:
        books_data_in_category = parse_books_info(each_category_html)
        if books_data_in_category:
          print(f"{len(books_data_in_category)}冊の本の情報を取得しました。")
          all_books_data.extend(books_data_in_category) # 結果を総合リストに追加
          time.sleep(1)
        else:
          print("このカテゴリからは情報を取得できませんでした。")
      else:
        print("このカテゴリページのhtmlを正常に取得できませんでした")
    if all_books_data:
      write_data_by_csv(all_books_data)
    else:
      print("\n最終的に取得できた本の情報はありませんでした。ファイルは作成されません。")
  else:
    print("トップページのhtmlを正常に取得できませんでした")

if __name__ == "__main__":
  main()