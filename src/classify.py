# Begin Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
# End Classifiers
from sklearn.metrics import accuracy_score

def benchmark_classifiers(classifiers, x_train, y_train, x_test, y_test):
    scores = {}
    for clf, name in classifiers:
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        scores[name] = accuracy
    return scores

if __name__ == '__main__':
    from data import get_ml_data

    classifiers = [
        (KNeighborsClassifier(n_neighbors=12), "kNN"),
        (RidgeClassifier(tol=1e-2, solver='lsqr'), 'Ridge regression'),
        (Perceptron(n_iter=20), 'Perceptron'),
        (PassiveAggressiveClassifier(n_iter=10), 'Passive Agressive Classifier'),
        (RandomForestClassifier(n_estimators=200), 'Random Forest'),
        (MultinomialNB(alpha=0.01), 'Multinomial Naive Bayes'),
        (BernoulliNB(alpha=0.01), "Bernoulli Naive Bayes"),
        (LinearSVC(penalty='l2', tol=1e-3), "SVM"),
    ]
    ml_data = get_ml_data()
    scores = benchmark_classifiers(classifiers, ml_data['x_train'], ml_data['y_train'], ml_data['x_test'], ml_data['y_test'])
    print(scores)