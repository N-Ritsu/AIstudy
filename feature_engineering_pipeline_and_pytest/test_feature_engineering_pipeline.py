import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Generator
# pytestのキャプチャ機能のための型をインポート
from _pytest.capture import CaptureFixture
# テスト対象のスクリプトから、テストしたい関数をインポート
# オブジェクトそのものではなく、オブジェクトを生成する関数をインポートする
from feature_engineering_pipeline import (build_preprocessor, print_pipeline_structure)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    テストで使用するサンプルデータを作成するフィクスチャ。
    Args:
        None
    Returns:
        pd.DataFrame: 欠損値を含むテスト用のデータフレーム。
    """
    data: Dict[str, Any] = {
        'age': [25, 30, np.nan, 45],
        'city': ['Tokyo', 'Osaka', 'Nagoya', np.nan],
        'education': ['Bachelor', 'Master', np.nan, 'PhD'],
        'income': [500, 600, 750, 1000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor() -> Generator[ColumnTransformer, None, None]:
    """
    テスト用の前処理パイプライン(preprocessor)を生成するフィクスチャ。
    Args:
        None
    Yields:
        ColumnTransformer: build_preprocessor() で構築された前処理パイプライン。
    """
    # build_preprocessor関数を呼び出して、テスト対象を生成
    yield build_preprocessor()


def test_numeric_transformer_output(sample_data: pd.DataFrame, preprocessor: ColumnTransformer) -> None:
    """
    数値データ用パイプラインが正しく機能することをテストする。
    1. 欠損値を補完する。
    2. 正しい形状(shape)でデータを返す。
    3. スケーリングにより平均がほぼ0、標準偏差がほぼ1になる。
    Args:
        sample_data (pd.DataFrame): テスト用のサンプルデータ。
        preprocessor (ColumnTransformer): テスト対象の前処理パイプライン。
    Returns:
        None
    """
    # テスト対象のパイプラインをpreprocessorから名前で取得する
    # fitしないとnamed_transformers_は使えないため、一度fitする
    preprocessor.fit(sample_data)
    numeric_transformer = preprocessor.named_transformers_['num']

    # 数値カラムのみを抽出
    numeric_data = sample_data[['age', 'income']]
    
    # fit_transformを実行
    transformed = numeric_transformer.transform(numeric_data)
    
    # 1. 欠損値がないことを確認
    assert not np.isnan(transformed).any(), "変換後のデータにNaNが含まれています"
    
    # 2. 形状が正しいことを確認 (4行, 2列)
    assert transformed.shape == (4, 2), f"期待される形状(4, 2)と異なります: {transformed.shape}"
    
    # 3. 平均がほぼ0であることを確認 (小数点以下の誤差を許容)
    # fit時に計算された平均・標準偏差で変換されるため、入力データのみで計算した平均・標準偏差とは異なる
    # そのため、変換後のデータの平均が0, 標準偏差が1に近いことを確認する
    mean_after_transform = numeric_transformer.fit(numeric_data).transform(numeric_data).mean(axis=0)
    std_after_transform = numeric_transformer.fit(numeric_data).transform(numeric_data).std(axis=0)
    assert np.allclose(mean_after_transform, [0, 0]), f"平均が0ではありません: {mean_after_transform}"
    assert np.allclose(std_after_transform, [1, 1]), f"標準偏差が1ではありません: {std_after_transform}"


def test_preprocessor_output_shape(sample_data: pd.DataFrame, preprocessor: ColumnTransformer) -> None:
    """
    ColumnTransformer(preprocessor)が、複数のデータ型を処理し、最終的に期待される列数を持つ配列を返すことをテストする。
    Args:
        sample_data (pd.DataFrame): テスト用のサンプルデータ。
        preprocessor (ColumnTransformer): テスト対象の前処理パイプライン。
    Returns:
        None
    """
    # preprocessorでデータを変換
    transformed = preprocessor.fit_transform(sample_data)
    
    # --- 期待される列数の計算 ---
    # 数値: age, income -> 2列
    # 名義カテゴリ(city): Tokyo, Osaka, Nagoya -> 3列 (OneHotEncoderがfit時に学習)
    # 順序カテゴリ(education): education -> 1列 (OrdinalEncoder)
    # 合計: 2 + 3 + 1 = 6列
    expected_columns = 6
    
    assert transformed.shape == (sample_data.shape[0], expected_columns), \
        f"期待される列数 {expected_columns} と異なります: {transformed.shape[1]}"
    assert not np.isnan(transformed).any(), "前処理後のデータにNaNが含まれています"


def test_full_model_pipeline(sample_data: pd.DataFrame, preprocessor: ColumnTransformer) -> None:
    """
    モデル全体(Pipeline)が、fitとpredictをエラーなく実行できることをテストする。
    Args:
        sample_data (pd.DataFrame): テスト用のサンプルデータ。
        preprocessor (ColumnTransformer): テスト対象の前処理パイプライン。
    Returns:
        None
    """
    # 前処理パイプラインと分類器を結合した最終的なモデルパイプラインをテスト内で構築
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    # テスト用のターゲット変数を作成
    y_sample = pd.Series([0, 1, 0, 1])
    
    try:
        # 学習
        model.fit(sample_data, y_sample)
        
        # 予測
        predictions = model.predict(sample_data)
        
        # 予測結果の数と入力データの行数が一致することを確認
        assert len(predictions) == len(sample_data), "予測結果の数が入力データと一致しません"
        
    except Exception as e:
        pytest.fail(f"モデルのfit/predict中に予期せぬエラーが発生しました: {e}")


def test_print_pipeline_structure(capsys: CaptureFixture, preprocessor: ColumnTransformer) -> None:
    """
    自作の表示用関数がエラーなく実行され、主要なキーワードを出力に含んでいるかをテストする。
    Args:
        capsys (CaptureFixture): pytestの標準出力キャプチャ機能。
        preprocessor (ColumnTransformer): テスト対象の前処理パイプライン。
    Returns:
        None
    """
    # テスト対象のモデルを構築
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])
    
    # 関数を実行
    print_pipeline_structure(model)
    
    # print文などで画面に出力された内容を、変数として捕まえる
    captured = capsys.readouterr()
    output = captured.out
    
    # 出力に主要なステップ名やクラス名が含まれているかを確認
    expected_keywords = [
        "Pipeline", "preprocessor", "ColumnTransformer", "classifier", 
        "LogisticRegression", "num", "cat_nom", "cat_ord"
    ]
    for keyword in expected_keywords:
        assert keyword in output, f"出力にキーワード '{keyword}' が含まれていません"