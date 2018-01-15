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

def tune_classifier(clf, params, data):
    """Tunes a classifier. Does not tune a fish."""
    clf = GridSearchCV(clf, params)
    clf.fit(data['x_train'], data['y_train'])
    return clf

def get_classifiers():
    """Return a list of call classifiers used in this project"""
    classifiers = {
        'knn': {'clf': KNeighborsClassifier(), 'name': 'kNN'},
        'perceptron': {'clf': Perceptron(n_iter=100), 'name': 'Perceptron'},
        'random_forest': {'clf': RandomForestClassifier(), 'name': 'Random Forest'},
        'svm': {'clf': LinearSVC(), 'name': 'SVM'},
        'nb': {'clf': BernoulliNB(), 'name': 'Naive Bayes'},
    }
    return classifiers

def get_tuned_params(classifier_name):
    """Return a hash of the tuned parameters"""
    tuned_params = {
        'knn': [{'n_neighbors': list(range(1, 21))}],
        'perceptron': [{'n_iter': [100]}],
        'random_forest': [{'n_estimators': [10, 20, 100, 150]}],
        'svm': [{'penalty': ['l2'], 'tol': [1e-3, 1e-2, 1e-1]}],
        'nb': [{'alpha': [0.01, 0.05, 0.25]}]
        }
    return tuned_params.get(classifier_name)

if __name__ == '__main__':
    pass
    # from data import get_ml_data
    # import warnings
    # warnings.filterwarnings('ignore')


    # ml_data = get_ml_data()
    # classifiers = get_classifiers()
    # scores = benchmark_classifiers(classifiers, ml_data['x_train'], ml_data['y_train'], ml_data['x_test'], ml_data['y_test'])
    # print(scores)
    # clf = tune_classifier(classifiers[0][0], tuned_parameters['knn'], ml_data)
    # for name, classifier in classifiers.items():
    #     clf = tune_classifier(classifier['clf'], tuned_parameters[name], ml_data)
    #     print(f"{classifier['name']}")
    #     print(clf.best_params_)
    #     print(clf.best_score_)