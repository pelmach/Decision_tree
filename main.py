from sklearn.datasets import load_wine
from sklearn import tree
from matplotlib import pyplot as plt

clf = tree.DecisionTreeClassifier(random_state=0)
wine = load_wine()

clf = clf.fit(wine.data, wine.target)

tree.plot_tree(clf,
               feature_names=wine.feature_names,
               class_names=wine.target_names,
               filled=True)
plt.show()