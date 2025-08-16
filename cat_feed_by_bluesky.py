import os
import requests
from dotenv import load_dotenv
from PIL import Image
from transformers import pipeline
from atproto import Client
from atproto_client.models.app.bsky.embed.images import Main as ImagesEmbed
from atproto_client.models.app.bsky.embed.record_with_media import Main as RecordWithMediaEmbed
from atproto_client.models.app.bsky.embed.record import View as RecordEmbed

image_classifier = pipeline('image-classification', model='google/vit-base-patch16-224')
load_dotenv()
# 自分のBlueskyのIDと、生成したアプリパスワードを入力
BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE")
BLUESKY_PASSWORD = os.getenv("BLUESKY_PASSWORD")
if not BLUESKY_HANDLE or not BLUESKY_PASSWORD:
    print("エラー: 環境変数 BLUESKY_HANDLE または BLUESKY_PASSWORD が設定されていません。")
    exit() # プログラムを終了

def is_cat_image(image_url):
    #画像URLを受け取り、猫の画像かどうかを判定する関数
    try:
        # URLから画像をダウンロードして開く
        image = Image.open(requests.get(image_url, stream=True).raw)
        # AIモデルで画像を分類
        results = image_classifier(image)
        
        # 結果の中に'cat'という単語が含まれているかチェック
        for result in results:
            if 'cat' in result['label']:
                print(f"  -> AIの判定: {result['label']} (スコア: {result['score']:.2f}) ... 猫です！")
                return True

        print("  -> AIの判定: 猫ではありません。")
        return False
    except Exception as e:
        print(f"  -> 画像の処理中にエラー: {e}")
        return False
    
#型ヒント。strもしくはNoneが返されますということの明示
def extract_image_url(post_view) -> str | None:
  # 関数の入り口で、前提条件をチェックする (ガード節)
  #post_viewの中にpost属性があるか、post_view.postの中にrecord属性があるか、post_view.post.recordの中にembed属性があるか
  if not hasattr(post_view, 'post') or not hasattr(post_view.post, 'record') or not hasattr(post_view.post.record, 'embed'):
    return None

  image_embed_to_check = None # 埋め込みデータがなかった時等用の返還デフォルト値
  embed = post_view.post.record.embed #埋め込みデータを取得
  
  # 1-1. それが「画像埋め込み」なら、それをチェック対象とする
  if isinstance(embed, ImagesEmbed):
    image_embed_to_check = embed #埋め込みデータ(画像)をimage_embed_to_checkに代入
  # 1-2. それが「画像付き引用リポスト」なら、その中の「メディア部分」をチェック対象とする
  elif isinstance(embed, RecordWithMediaEmbed) and isinstance(embed.media, ImagesEmbed):
    image_embed_to_check = embed.media #埋め込みデータ(リポスト)の中のメディア部分(画像データ)をimage_embed_to_checkに代入
  # 1-3. それが「通常の引用リポスト」なら、引用されている「元の投稿」をチェックする
  # 引用されている投稿（record）に、さらに埋め込みがあるかチェック
  elif isinstance(embed, RecordEmbed) and hasattr(embed.record, 'embeds') and embed.record.embeds:
      # 複数の埋め込みがある可能性があるのでループで見る
    for inner_embed_view in embed.record.embeds:
      if isinstance(inner_embed_view, ImagesEmbed):
          image_embed_to_check = inner_embed_view
          break # 一つ見つけたらループを抜ける
  
  #第二ガード節
  if not image_embed_to_check:
     return None

  image_info = image_embed_to_check.images[0]
  # image_infoに"image"属性があるか？ー＞image_info.imageの中に"ref"属性はあるか？ー＞image_info.image.refの中に"link"属性はあるか？
  if hasattr(image_info, 'image') and image_info.image.ref and image_info.image.ref.link:
    cid = image_info.image.ref.link
    did = post_view.post.author.did # 投稿者のID 
    # Blueskyの画像CDNのURL形式で、完全なURLを組み立てる
    return f"https://cdn.bsky.app/img/feed_fullsize/plain/{did}/{cid}@jpeg"
  return None


def main():
  # Blueskyクライアントを作成し、ログインする
  client = Client()
  client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)
  print("ログイン成功！")
  # 自分のタイムラインを取得する (最大20件)
  response = client.get_timeline(limit=20)

  # 取得した投稿を一つずつ見てみる
  if response and response.feed:
      print("\n--- 猫フィードの作成開始 ---")
      cat_feed = []
      for post_view in response.feed:
        print(f"\n投稿をチェック中: {post_view.post.record.text[:50]}...")
        image_url = extract_image_url(post_view)
        if image_url:
          print("  -> 画像を発見！AIの判定を開始します...")
          if is_cat_image(image_url):
              cat_feed.append(post_view)
        else:
          print("  -> この投稿（または引用先）には、直接の画像埋め込みはありませんでした。")
 
  print(f"\n--- 猫フィード作成完了！ {len(cat_feed)}件の猫の投稿を見つけました ---")

if __name__ == "__main__":
  main()