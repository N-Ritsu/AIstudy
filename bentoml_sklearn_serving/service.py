import numpy as np
import bentoml

# モデルを直接取得
iris_model = bentoml.sklearn.get("iris_clf:latest")

@bentoml.service(
    name="iris_classifier",
    resources={"cpu": "2", "memory": "1Gi"},
    traffic={"timeout": 30},
)
class IrisClassifier:
    def __init__(self):
        self.model = iris_model.load_model()

    @bentoml.api
    def classify(self, input_series: np.ndarray) -> np.ndarray:
        """
        入力されたNumpy配列に対して、分類予測を実行する。
        Args:
            input_series (np.ndarray): 予測対象の特徴量を含むNumpy配列。
        Returns:
            np.ndarray: 予測されたクラスの配列。
        """
        result = self.model.predict(input_series)
        return result

# serve で使うエイリアス
svc = IrisClassifier