import argparse
from classification import get_classifiers, get_tuned_params, tune_classifier
from data import get_ml_data, read_input_file
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



if __name__ == '__main__':
    clf_choices = ['nb', 'random_forest', 'svm']
    parser = argparse.ArgumentParser(description='Signal Peptide Predictor')
    parser.add_argument('classifier', choices=clf_choices + ['benchmark'], help='Which classifier to use')
    parser.add_argument('--slice_length', help='Length of sequence to analyze', type=int, default=35)
    parser.add_argument('--n', help='The N in N-Gram ', type=int, default=3)
    parser.add_argument('--file', help='File of a sequence to predict')
    args = parser.parse_args()
    classifiers = get_classifiers()


    if args.classifier == 'benchmark':
        print("Benchmarking classifiers")
        for tm in ['tm', 'non_tm', None]:
            data = get_ml_data((args.n, args.n), args.slice_length, tm=tm)
            print('*' * 80)
            print(f"{tm}")
            for clf_name in clf_choices:
                clf = classifiers[clf_name]
                clf = tune_classifier(clf['clf'], get_tuned_params(clf_name), data)
                clf.fit(data['x_train'], data['y_train'])
                pred = clf.predict(data['x_test'])
                score = metrics.accuracy_score(data['y_test'], pred)
                print('=' * 80)
                print(f"{classifiers[clf_name]['name']}")
                print("accuracy: %0.3f" % score)
                print("confusion matrix:")
                print(metrics.confusion_matrix(data['y_test'], pred))
                print(f"params: {clf.best_params_}")
            # print("classification report:")
            # print(metrics.classification_report(data['y_test'], pred, data['feature_names']))
    elif args.file:
        data = get_ml_data((args.n,args.n), args.slice_length)
        file_data = read_input_file(args.file, data['vectorizer'], (args.n, args.n), args.slice_length)
        clf = classifiers[args.classifier]
        clf['clf'] = tune_classifier(clf['clf'], get_tuned_params(args.classifier), data)
        clf['clf'].fit(data['x_train'], data['y_train'])
        pred = clf['clf'].predict(file_data)
        hits = sum(pred)
        print(f"Found {hits} signal peptides in {len(pred)} sequences ({hits/len(pred):.1%})")
    else:
        parser.error("Please select a file to analyze or use 'benchmark' as a classifier")

    

