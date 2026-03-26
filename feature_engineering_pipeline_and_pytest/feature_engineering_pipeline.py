import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn import set_config
from typing import List, Dict, Any, Optional, Tuple

# 特徴量とターゲットのカラム名
TARGET_COLUMN = 'purchased'
NUMERIC_FEATURES = ['age', 'income']
NOMINAL_FEATURES = ['city']
ORDINAL_FEATURES = ['education']

# 順序特徴量のカテゴリ定義
ORDINAL_CATEGORIES = [['High School', 'Bachelor', 'Master', 'PhD']]

# データ分割のパラメータ
TEST_SIZE = 0.3
RANDOM_STATE = 42


def print_pipeline_structure(estimator: BaseEstimator, indent: int = 0, step_name: Optional[str] = None) -> None:
    """
    PipelineやColumnTransformerの構造を表示する。
    Args:
        estimator (BaseEstimator): 構造を表示したいscikit-learnの推定器（Pipelineなど）。
        indent (int, optional): 表示のインデントレベル。デフォルトは0。
        step_name (Optional[str], optional): 表示するステップ名。デフォルトはNone。
    Returns:
        None
    """
    # インデント１つに対してスペース2つ
    indent_str = '  ' * indent
    
    # ステップ名(パイプライン内の処理をどんな順にステップを進めるか)を表示
    if step_name:
        print(f"{indent_str}- {step_name}: ", end="")
    else:
        # ステップ名がない場合、インデントを合わせるためにスペースだけ表示
        print(indent_str, end="")
    
    # estimator自身がpipelineオブジェクトなら
    if isinstance(estimator, Pipeline):
        print("Pipeline")
        # Pipelineの中の各ステップを表示するためのループ
        for name, step in estimator.steps:
            print_pipeline_structure(step, indent + 1, name)
            
    # estimator自身がColumnTransformerオブジェクト(一番外側のpipeline内の、前処理部分)なら
    elif isinstance(estimator, ColumnTransformer):
        print("ColumnTransformer")
        # fit後なら transformers_、fit前なら transformers を使用
        transformers_list = getattr(estimator, 'transformers_', estimator.transformers)
        # ColumnTransformerの中の各変換器(どんな要素の処理を担当するか)を表示するためのループ
        for name, trans, cols in transformers_list:
            # 処理内容が 'drop' (列削除) や 'passthrough' (何もしない) の場合は、詳細表示は不要なのでスキップする
            if trans == 'drop' or trans == 'passthrough':
                continue
            print(f"{indent_str}  └─ {name} → {cols}") # ColumnTransformer内の各変換器の名前と処理する担当カラムを表示
            print_pipeline_structure(trans, indent + 2) # 各変換器がもつ処理内容(pipelineオブジェクト)表示するためのループ

    # 一番末端の、ImputerやScalerなどの処理について
    else:
        # estimatorのクラス名(処理の名前)を取得
        cls_name = estimator.__class__.__name__
        # 処理後の現在のパラメータを取得
        params = estimator.get_params()
        
        try:
            # 処理前のオブジェクトを生成し、パラメータを処理後のものと比較する
            default_params = type(estimator)().get_params()
            changed = {k: v for k, v in params.items() if str(v) != str(default_params.get(k))}
        except Exception:
            # パラメータなしのコンストラクタを持たない推定器のためのフォールバック
            changed = params
        
        # 変更されたパラメータがある場合は、処理によってパラメータの置き換えが行われていることを示すため、それも表示
        if changed:
            param_str = ", ".join(f"{k}={v}" for k, v in changed.items())
            print(f"{cls_name}({param_str})")
        else:
            print(cls_name)


def create_sample_data() -> pd.DataFrame:
    """
    サンプル用のデータフレームを作成する。
    Args:
        None
    Returns:
        pd.DataFrame: 機械学習に使用するサンプルデータ。
    """
    data: Dict[str, List[Any]] = {
        'age': [25, 30, np.nan, 45, 50, 55, 60, 65, 70, 75],
        'city': ['Tokyo', 'Osaka', 'Nagoya', 'Tokyo', np.nan, 'Osaka', 'Fukuoka', 'Fukuoka', 'Tokyo', 'Nagoya'],
        'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', np.nan, 'Master', 'High School', 'PhD', 'Bachelor'],
        'income': [500, 600, 750, 1000, 1200, 1100, 900, 400, 1500, 800],
        'purchased': [0, 1, 0, 1, 1, 1, 0, 0, 1, 1]
    }
    return pd.DataFrame(data)


def build_preprocessor() -> ColumnTransformer:
    """
    特徴量エンジニアリングのための前処理パイプラインを構築する。
    数値データ、名義カテゴリデータ、順序カテゴリデータに対してそれぞれ異なる前処理パイプラインを定義し、ColumnTransformerで統合する。
    Args:
        None
    Returns:
        ColumnTransformer: 構築された前処理パイプライン。
    """
    # 数値特徴量に対する処理パイプライン
    # 欠損値を平均値で補完し、その後スケーリングを行う
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # 名義カテゴリカル特徴量に対する処理パイプライン
    # 欠損値を最頻値で補完し、One-Hotエンコーディングを適用する
    categorical_nominal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 順序カテゴリカル特徴量に対する処理パイプライン
    # 欠損値を最頻値で補完し、定義された順序でエンコーディングを行う
    categorical_ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    # ColumnTransformer を使って、各特徴量に適切な処理を割り当てる
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat_nom', categorical_nominal_transformer, NOMINAL_FEATURES),
            ('cat_ord', categorical_ordinal_transformer, ORDINAL_FEATURES)
        ],
        remainder='passthrough'  # ここで指定されなかったカラムはそのまま通す
    )
    return preprocessor


def main() -> None:
    """
    前処理パイプラインを構築し、サンプルデータを用いて学習と評価を行うメイン関数。
    Args:
        None
    Returns:
        None
    """
    # scikit-learnの推定器の表示設定
    set_config(display='text', print_changed_only=True)

    # --- データ準備 ---
    df = create_sample_data()

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # --- パイプライン構築 ---
    preprocessor = build_preprocessor()

    # 前処理パイプラインと分類器を結合した最終的なモデルパイプライン
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    # --- 学習と評価 ---
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # --- 結果表示 ---
    print(f"モデルの正解率: {accuracy:.4f}\n")

    print("--- パイプライン構造 ---")
    print_pipeline_structure(model)


if __name__ == "__main__":
    main()