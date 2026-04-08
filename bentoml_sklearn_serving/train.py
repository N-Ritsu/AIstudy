import bentoml
from sklearn import svm
from sklearn import datasets

# 1. データセットをロード
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 2. モデルを学習
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

# 3. 学習済みモデルをBentoMLのローカルストアに保存
# - "iris_clf": モデルの名前
# - clf: 保存するモデルオブジェクト
# これにより、モデルはバージョン管理され、service.pyから安全に呼び出せる。
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(f"モデルを保存しました: {saved_model}")