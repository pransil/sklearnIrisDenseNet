# Simple SKLearn code for denseNet classsifier

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
iris =  datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


my_classifier = MLPClassifier(solver='adam', alpha=1e-5, max_iter=4000,
                              hidden_layer_sizes=(5, 5, 6), random_state=1)

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))
