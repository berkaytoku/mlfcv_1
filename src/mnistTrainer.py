__author__ = 'berkaytoku'
# Code snippets used from scikit-learn API documentation

from sklearn import svm, tree, metrics, cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from operator import itemgetter
import numpy as np
import mnistParser

def train_classifier(classifiername, imagearray, labelarray):
    # Classifiers are defined with the best hyperparameters found by randomized grid search
    if(classifiername == "svm"):
        classifier = svm.SVC(1, "poly", 2)
    elif(classifiername == "dtree"):
        classifier = tree.DecisionTreeClassifier(criterion="entropy", max_features=409, max_depth=18)
    elif(classifiername == "rforest"):
        classifier = RandomForestClassifier(criterion= 'gini', max_features= 32, n_estimators= 120, max_depth= 16)

    classifier = classifier.fit(imagearray, labelarray)
    return classifier

def test_classifier(classifier, imagearray, labelarray):
    prediction = classifier.predict(imagearray)
    print metrics.classification_report(labelarray, prediction, digits=4)

def export_tree(classifier):
    if type(classifier) != tree.DecisionTreeClassifier: return
    from sklearn.externals.six import StringIO
    import pydot
    dot_data = StringIO()
    tree.export_graphviz(classifier, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("../output/tree.png")

def plot_importances(classifier):
    importances = classifier.feature_importances_
    importances = importances.reshape((28, 28))

    import matplotlib.pyplot as plt
    plt.matshow(importances, cmap=plt.cm.hot)
    plt.show()

def cross_validate_classifier(classifier, images, labels):
    scores = cross_validation.cross_val_score(classifier, images, labels, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def optimize_parameters(clf, images, labels):
    # specify parameters and distributions to sample from
    param_dist = {"degree": sp_randint(1, 6),
                  "kernel": ["linear", "poly", "rbf", "sigmoid"],
                  "C": sp_randint(1, 8)}

    # run randomized search
    n_iter_search = 500
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)

    random_search.fit(images, labels)
    report(random_search.grid_scores_)

dataLimit = 60000
trainImageArray = mnistParser.parse_image_data("train-images.idx3-ubyte", dataLimit)
trainLabelArray = mnistParser.parse_label_data("train-labels.idx1-ubyte", dataLimit)

testImageArray = mnistParser.parse_image_data("t10k-images.idx3-ubyte")
testLabelArray = mnistParser.parse_label_data("t10k-labels.idx1-ubyte")

clf = train_classifier("dtree", trainImageArray, trainLabelArray)
# optimize_parameters(clf, trainImageArray, trainLabelArray)
test_classifier(clf, testImageArray, testLabelArray)
export_tree(clf)
cross_validate_classifier(clf, trainImageArray, trainLabelArray)
plot_importances(clf)