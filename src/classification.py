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
        'svm': {'clf': LinearSVC(), 'name': 'SVM'}
    }
    return classifiers

def get_tuned_params(classifier_name):
    """Return a hash of the tuned parameters"""
    tuned_params = {
        'knn': [{'n_neighbors': list(range(1, 21))}],
        'perceptron': [{'n_iter': [100]}],
        'random_forest': [{'n_estimators': [10, 20, 100, 150]}],
        'svm': [{'penalty': ['l2'], 'tol': [1e-3, 1e-2, 1e-1]}]
        }
    return tuned_params.get(classifier_name)

if __name__ == '__main__':
    from data import get_ml_data
    import warnings
    warnings.filterwarnings('ignore')

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
    classifiers = {
        'knn': {'clf': KNeighborsClassifier(), 'name': 'kNN'},
        'perceptron': {'clf': Perceptron(n_iter=100), 'name': 'Perceptron'},
        'random_forest': {'clf': RandomForestClassifier(), 'name': 'Random Forest'},
        'svm': {'clf': LinearSVC(), 'name': 'SVM'}
    }
    # classifiers = get_classifiers()
    ml_data = get_ml_data()
    # scores = benchmark_classifiers(classifiers, ml_data['x_train'], ml_data['y_train'], ml_data['x_test'], ml_data['y_test'])
    # print(scores)
    tuned_parameters = {
        'knn': [
            {
                'n_neighbors': list(range(1,21))
            }
        ],
        'perceptron': [
            {
                'n_iter': [100]
            }
        ],
        'random_forest': [
            {
                'n_estimators': [10, 20, 100, 150]
            }
        ],
        'svm': [
            {
                'penalty': ['l2'],
                'tol': [1e-3, 1e-2, 1e-1]
            }
        ]
    }
    # clf = tune_classifier(classifiers[0][0], tuned_parameters['knn'], ml_data)
    for name, classifier in classifiers.items():
        clf = tune_classifier(classifier['clf'], tuned_parameters[name], ml_data)
        print(f"{classifier['name']}")
        print(clf.best_params_)
        print(clf.best_score_)