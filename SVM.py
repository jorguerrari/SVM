import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()  # here i can see all available

#print(cancer.feature_names)
#print(cancer.target_names)

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

classes = ['malignant' 'benign']

# SVM
clf = svm.SVC(kernel= "linear", C=2)  # C is the soft margin, there are other kernels
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html for parameters

# Compare using k nearest neighbors
#clf = KNeighborsClassifier(n_neighbors=15)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
