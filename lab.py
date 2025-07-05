#loading required labels
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#loading datasets
iris = datasets.load_iris()

#printing description and features
print(iris.DESCR)
features = iris.data
labels = iris.target
print(features[0], labels[0])

#training the classification
clf = KNeighborsClassifier()
clf.fit(features, labels)
preds = clf.predict(features)

print(preds)
