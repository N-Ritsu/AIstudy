import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Tuple, Dict, Any

# 定数としてファイルパスを定義
DATA_FILE_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

def load_data(file_path: str) -> pd.DataFrame:
    """
    指定されたパスからCSVデータを読み込み、データフレームとして返す
    Args:
        file_path (str): 読み込むCSVファイルのパス
    Returns:
        pd.DataFrame: 読み込まれたデータフレーム
    """
    print("--- データの読み込みを開始します ---")
    df = pd.read_csv(file_path)
    print("データの読み込みが完了しました。")
    print("\n[INFO] 読み込み直後のデータフレーム情報:")
    df.info()
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレームを受け取り、モデル学習に適した形に前処理を行う
    Args:
        df (pd.DataFrame): 前処理対象のデータフレーム
    Returns:
        pd.DataFrame: 前処理済みのデータフレーム
    """
    print("\n--- データの前処理を開始します ---")
    
    # customerIDは予測に不要なため削除
    df_processed = df.drop('customerID', axis=1)

    # TotalChargesは数値であるべきだが、文字列型になっているor欠損あり(他の列より、データ数が少し少なかった)
    # 空白文字などが含まれている可能性があるため、数値に変換できない値を欠損値(NaN)に置き換える(errors='coerce')
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    
    # .dropna: 欠損値がある行を削除する
    # inplace=True: 元のデータフレーム自体を直接変更(値を返すわけではない)
    df_processed.dropna(inplace=True)

    print("特徴量エンジニアリングを実行します...")

    # --- 特徴量1: 契約期間(tenure)をグループ化 ---
    # 契約期間を離散的なカテゴリに分けることで、モデルが非線形な関係を捉えやすくなる
    # 例えば、12ヶ月未満の顧客は特に離反しやすい、といった傾向を学習しやすくなる
    bins = [0, 12, 36, 60, 72] # 0-12ヶ月, 13-36ヶ月, 37-60ヶ月, 61-72ヶ月
    labels = ['new_customer', 'mid_term_customer', 'long_term_customer', 'loyal_customer']
    # df_processedに新しい列'tenure_group'を追加(例: df_processed['tenure_group'][0] = new_customer, df_processed['tenure_group'][1] = mid_term_customer,...)
    df_processed['tenure_group'] = pd.cut(df_processed['tenure'], bins=bins, labels=labels, right=False) # right=False: 区間の右端を含まない(0か月以上12か月未満など)

    # --- 特徴量2: オプションサービス未契約フラグ ---
    # (df_processedの元データに、df_processed['OnlineSecurity'], df_processed['OnlineBackup'],...などの列がある(Yes/Noの値を持つ))
    # 主要なオプションサービスを契約しているかどうかをチェック
    optional_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    # 'No'の数を数えオプションサービスの数と一致すれば(すべてNoなら)、全て未契約と判断(True/Falseを格納)
    df_processed['no_optional_services'] = (df_processed[optional_services] == 'No').sum(axis=1) == len(optional_services)
    # bool型(True/False)を数値(1/0)に変換
    df_processed['no_optional_services'] = df_processed['no_optional_services'].astype(int)

    # --- カテゴリ変数を数値に変換 ---
    # Churnもここで'Yes'/'No'から1/0に変換しておく
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})

    # gender列は Male/Female を 1/0 に変換
    df_processed['gender'] = df_processed['gender'].map({'Male': 1, 'Female': 0})

    # 3種類以上のカテゴリを持つ列をワンホットエンコーディング
    # categorical_colsに含まれている列は、全て３つ以上のカテゴリを持つ。例えば、'InternetService'列は'No', 'DSL', 'Fiber optic'の3種類がある
    # 新しく作った'tenure_group'もここで一緒に処理する
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'Contract', 'PaymentMethod', 'tenure_group'] # 'tenure_group'を追加
    # pd.get_dummies: ワンホットエンコーディングを実行
    # ワンホットエンコーディングとは: 
    # Step1: Contract列に含まれるユニークな値（'Month-to-month', 'One year', 'Two year'）を全て見つけ出す
    # Step2: それぞれのユニークな値の名前を持つ、新しい列を作成（Contract_Month-to-month, Contract_One year, Contract_Two year）
    # Step3: 元のContract列を削除
    # Step4: 各行について、元の値に該当する新しい列に1を、それ以外の列に0を入れる
    # drop_first=True: 最初のカテゴリを基準として削除し、多重共線性を防ぐ
    # (例: contractには、Contract_Month-to-month	Contract_One_year	Contract_Two_yearがあるが、Contract_Month-to-monthはその他2つに該当しない場合として一意に定まるため、いらない)
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    print("データの前処理が完了しました。")
    print("\n[INFO] 前処理後のデータフレーム情報:")
    df_processed.info()
    return df_processed

def train_and_evaluate_model(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[lgb.LGBMClassifier, pd.Index]:
    """
    前処理済みデータフレームからモデルを学習し、性能を評価する
    Args:
        df (pd.DataFrame): 前処理済みのデータフレーム
        params (Dict[str, Any]): モデルのハイパーパラメータ
    Returns:
        Tuple[lgb.LGBMClassifier, pd.Index]: 学習済みモデルと、使用した特徴量のリスト
    """
    print("\n--- モデルの学習と評価を開始します ---")
    # 'Churn'列(解約or継続)が予測したい目的なのでyに、それ以外が予測に使う説明変数なのでXに格納
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # データを訓練用(80%)とテスト用(20%)に分割する
    # stratify=yを指定することで、元のデータのChurnの比率を保ったまま分割する("Churnが1":"Churnが0"の比率が同じになるように分割する)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 引数で受け取った最適なパラメータと、前回効果のあった設定を組み合わせてモデルを定義
    # class_weight='balanced': 多数派である「継続」を予測する方が全体の正解率が上がりやすい → 少数派である「離反」の予測を軽視しがち → データが少ないクラス(離反)を間違えたら、ペナルティをより重くするように自動で調整
    model = lgb.LGBMClassifier(
        class_weight='balanced', 
        random_state=42, # 完全なランダムではなく、42という固定されたランダム値を使うことで、再現性を確保 → 改善の効果を比較しやすくなる
        **params  # 最適化で見つけたパラメータを展開して設定
    )
    print("モデルの学習を開始...")
    model.fit(X_train, y_train)
    print("モデルの学習が完了しました。")

    # 学習済みモデルを使って、テストデータで予測を行い、性能を評価
    print("\n--- モデルの性能評価 ---")
    y_pred = model.predict(X_test)
    
    print(f"正解率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print("\n--- 混同行列 (Confusion Matrix) ---")
    print(confusion_matrix(y_test, y_pred))
    print("\n--- 分類レポート (Classification Report) ---")
    print(classification_report(y_test, y_pred, target_names=['継続 (0)', '離反 (1)']))
    
    return model, X.columns

def visualize_feature_importance(model: lgb.LGBMClassifier, features: pd.Index):
    """
    学習済みモデルから特徴量の重要度を抽出し、グラフとして可視化する
    Args:
        model (lgb.LGBMClassifier): 学習済みのLightGBMモデル
        features (pd.Index): 特徴量の名前が入ったリスト
    Returns:
        None
    """
    print("\n--- 結果の解釈（特徴量の重要度を可視化）を開始します ---")
    
    # 日本語の文字化け対策
    # WSL2環境などでは、事前に `sudo apt-get install -y fonts-ipafont-gothic` の実行が必要
    plt.rcParams['font.family'] = 'IPAexGothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 特徴量名と、その重要度スコアをデータフレームにまとめる
    feature_imp_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_ # 各特徴量の重要度スコアを取得(重みとは違い、方向性はない)
    }).sort_values('importance', ascending=False) # 重要度の高い順にソート

    # グラフとして可視化
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_imp_df)
    plt.title('特徴量の重要度 (LightGBM)')
    plt.xlabel('重要度 (Importance)')
    plt.ylabel('特徴量 (Feature)')
    plt.tight_layout()
    plt.show()
    print("グラフを表示しました。")

def tune_hyperparameters(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Optunaを使って、LightGBMのハイパーパラメータをチューニングする
    F1スコアが最大になるパラメータの組み合わせを探す
    Args:
        df (pd.DataFrame): 前処理済みのデータフレーム
    Returns:
        Dict[str, Any]: 最適化されたハイパーパラメータ
    """
    print("\n--- EX: ハイパーパラメータチューニングを開始します ---")
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    def objective(trial: optuna.Trial) -> float:
        # 探索するハイパーパラメータの範囲を定義
        # それぞれ、optuna.Trialが自動で最適な値を選んでくれる
        params = {
            'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000), # n_estimators: 決定木(勾配ブースティングの学習量)の数
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), # learning_rate: 各決定木がどれだけ強くパラメータを修正するか
            'max_depth': trial.suggest_int('max_depth', 3, 10), # max_depth: 各決定木の深さ(どれだけ細かく学習するか)の最大値
            'num_leaves': trial.suggest_int('num_leaves', 20, 100), # num_leaves: 各決定木にて、顧客を分類する箱の最大値
        }
        
        model = lgb.LGBMClassifier(class_weight='balanced', random_state=42, **params)
        model.fit(X_train, y_train) # モデルの学習
        y_pred = model.predict(X_val) # 予測させる
        
        # 今回の目的（バランス）に合わせ、離反クラス(1)のF1スコアを最大化の指標とする
        score = f1_score(y_val, y_pred, pos_label=1)
        return score

    study = optuna.create_study(direction='maximize') # 起動。returnされたF1スコアを最大化するのを目的とする
    study.optimize(objective, n_trials=50) # 50回の試行で最適値を探す。最適化において、objective関数が呼び出される

    print("チューニングが完了しました。")
    print(f"最適なF1スコア: {study.best_value:.4f}")
    print("最適なハイパーパラメータ:")
    print(study.best_params)
    
    return study.best_params


def main() -> None:
    """
    勾配ブースティングによる顧客の離反予測のメイン処理
    Args:
        None
    Returns:
        None
    """
    # 1. データの読み込み
    customer_df = load_data(DATA_FILE_PATH)
    
    # 2. データの前処理
    processed_df = preprocess_data(customer_df)
    
    # 3. ハイパーパラメータチューニングの実行
    best_params = tune_hyperparameters(processed_df)
    
    # 4. 最適なパラメータで最終的なモデルを学習・評価
    trained_model, feature_names = train_and_evaluate_model(processed_df, best_params)
    
    # 5. 結果の解釈と可視化
    visualize_feature_importance(trained_model, feature_names)

    print("\n--- 全ての処理が完了しました ---")

if __name__ == "__main__":
    main()