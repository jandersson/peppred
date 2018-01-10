# Begin Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
# End Classifiers
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def benchmark_classifiers(classifiers, x_train, y_train, x_test, y_test):
    scores = {}
    for clf, name in classifiers:
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        scores[name] = accuracy
    return scores

def tune_knn(data):
    tuned_parameters = [{
        'n_neighbors': list(range(1,21))
    }]
    clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters)
    clf.fit(data['x_train'], data['y_train'])
    return clf

def get_classifiers():
    """Return a list of call classifiers used in this project"""
    
    return [
        (KNeighborsClassifier(n_neighbors=12), "kNN"),
        (RidgeClassifier(tol=1e-2, solver='lsqr'), 'Ridge regression'),
        (Perceptron(n_iter=20), 'Perceptron'),
        (PassiveAggressiveClassifier(n_iter=10), 'Passive Agressive Classifier'),
        (RandomForestClassifier(n_estimators=200), 'Random Forest'),
        (MultinomialNB(alpha=0.01), 'Multinomial Naive Bayes'),
        (BernoulliNB(alpha=0.01), "Bernoulli Naive Bayes"),
        (LinearSVC(penalty='l2', tol=1e-3), "SVM"),
    ]

if __name__ == '__main__':
    from data import get_ml_data

    # classifiers = [
    #     (KNeighborsClassifier(n_neighbors=12), "kNN"),
    #     (RidgeClassifier(tol=1e-2, solver='lsqr'), 'Ridge regression'),
    #     (Perceptron(n_iter=20), 'Perceptron'),
    #     (PassiveAggressiveClassifier(n_iter=10), 'Passive Agressive Classifier'),
    #     (RandomForestClassifier(n_estimators=200), 'Random Forest'),
    #     (MultinomialNB(alpha=0.01), 'Multinomial Naive Bayes'),
    #     (BernoulliNB(alpha=0.01), "Bernoulli Naive Bayes"),
    #     (LinearSVC(penalty='l2', tol=1e-3), "SVM"),
    # ]
    classifiers = get_classifiers()
    ml_data = get_ml_data()
    # scores = benchmark_classifiers(classifiers, ml_data['x_train'], ml_data['y_train'], ml_data['x_test'], ml_data['y_test'])
    # print(scores)
    clf = tune_knn(ml_data)
    print(clf.best_params_)
    print(clf.best_score_)