import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('smestaj.csv', delimiter=',')

# Separate the data into features (X) and target (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train a SVM model with a linear kernel
classifier_linear = SVC(kernel='linear', random_state=0)
classifier_linear.fit(X_train, y_train)
y_pred_linear = classifier_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print('Accuracy of linear SVM:', accuracy_linear)

# Train a SVM model with a radial basis function kernel
classifier_rbf = SVC(kernel='rbf', random_state=0)
classifier_rbf.fit(X_train, y_train)
y_pred_rbf = classifier_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print('Accuracy of SVM with RBF kernel:', accuracy_rbf)
