import requests
import json

# 設定
INPUT_JSON_PATH = "input_data.json"
OUTPUT_JSON_PATH = "api_result.json"
API_URL = "http://127.0.0.1:5000/api/analyze"

def main() -> None:
    """
    JSON配列ファイルを読み込み、各要素の感情分析を行い、結果を配列として保存する
    Args:
        None
    Returns:
        None
    """
    try:
        # 1. 入力JSONファイルを読み込む
        print(f"▶入力ファイル '{INPUT_JSON_PATH}' を読み込んでいます...")
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        # 読み込んだデータがリスト形式かチェック
        if not isinstance(data_list, list):
            print("エラー: 入力ファイルはJSONの配列（リスト）形式である必要があります。 [...] で囲ってください。")
            return

        # 2. 結果を格納するための空のリストを用意
        results_list = []
        
        print(f"▶{len(data_list)}件のデータを順次分析します...")

        # 3. データのリストをループで1件ずつ処理
        for item in data_list:
            # 各要素に 'text' キーが含まれているかチェック
            if 'text' in item:
                text_to_analyze = item['text']
                print(f"  - 分析中: 「{text_to_analyze[:30]}...」")

                # APIを呼び出す (送るデータはitem全体でも、textだけでもOK)
                response = requests.post(API_URL, json=item)
                response.raise_for_status()
                
                # 応答データを結果リストに追加
                results_list.append(response.json())
            else:
                print(f"  - スキップ: 'text'キーが見つからないデータがありました。 {item}")
        
        # 4. 全ての分析結果をまとめて新しいJSONファイルに書き出す
        print(f"\n▶全ての分析結果を '{OUTPUT_JSON_PATH}' に保存しています...")
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)

        print(f"完了{len(results_list)}件の分析結果を '{OUTPUT_JSON_PATH}' に保存しました。")

    except FileNotFoundError:
        print(f"エラー: 入力ファイル '{INPUT_JSON_PATH}' が見つかりません。")
    except json.JSONDecodeError:
        print(f"エラー: '{INPUT_JSON_PATH}' は正しいJSON形式ではありません。")
    except requests.exceptions.RequestException as e:
        print(f"エラー: APIの呼び出しに失敗しました。Webサーバー(app.py)が起動しているか確認してください。")
        print(f"   詳細: {e}")

if __name__ == '__main__':
    main()