import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# loading the dataset
data = pd.read_csv('smestaj.csv', delimiter=',')

# x - features; y- results
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# scaling the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# linear svm
svc_linear = SVC(kernel='linear', random_state=0)
svc_linear.fit(X_train, y_train)
pred_linear = classifier_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, pred_linear)
print('Accuracy of linear SVM:', accuracy_linear)

# rbf svm
svc_rbf = SVC(kernel='rbf', random_state=0)
svc_rbf.fit(X_train, y_train)
pred_rbf = classifier_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, pred_rbf)
print('Accuracy of SVM with RBF kernel:', accuracy_rbf)
