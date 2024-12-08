
import imp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

df = pd.read_csv("Statistic.csv")
data = np.array(df)
X = np.array([d[: -1] for d in data])
Y = np.array([d[-1] for d in data])

Forest = RandomForestClassifier(n_estimators = 15)
Forest.fit(X, Y)
joblib.dump(Forest, "Forest.m")


"""

clf1 = svm.SVC(kernel = "linear", C = 100)
clf1.fit(xtrain, ytrain)
print("svm score of train: {:.3f}".format(clf1.score(xtrain, ytrain)))   # 0.896
print("svm score of test: {:.3f}".format(clf1.score(xtest, ytest)))   # 0.739

clf2 = tree.DecisionTreeClassifier(max_depth = 3)
clf2.fit(xtrain, ytrain)
print("tree score of train: {:.3f}".format(clf2.score(xtrain, ytrain)))   # 0.881
print("tree score of test: {:.3f}".format(clf2.score(xtest, ytest)))   # 0.826

clf3 = RandomForestClassifier(n_estimators = 3, random_state = 3)
clf3.fit(xtrain, ytrain)
print("forest score of train: {:.3f}".format(clf3.score(xtrain, ytrain)))   # 0.985
print("forest score of test: {:.3f}".format(clf3.score(xtest, ytest)))   # 0.870

clf4 = KNeighborsClassifier(n_neighbors = 2)
clf4.fit(xtrain, ytrain)
print("knn score of train: {:.3f}".format(clf4.score(xtrain, ytrain)))   # 0.866
print("knn score of test: {:.3f}".format(clf4.score(xtest, ytest)))   # 0.739

clf5 = GaussianNB()
clf5.fit(xtrain, ytrain)
print("bayes score of train: {:.3f}".format(clf5.score(xtrain, ytrain)))   # 0.866
print("bayes score of test: {:.3f}".format(clf5.score(xtest, ytest)))   # 0.696

clf6 = MLPClassifier(solver = "lbfgs", random_state = 2)
clf6.fit(xtrain, ytrain)
print("mlp score of train: {:.3f}".format(clf6.score(xtrain, ytrain)))   # 0.970
print("mlp score of test: {:.3f}".format(clf6.score(xtest, ytest)))   # 0.739

"""

