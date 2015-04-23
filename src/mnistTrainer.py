__author__ = 'berkaytoku'

from sklearn import svm
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import mnistParser

def train_classifier(classifiername, imagearray, labelarray):
    # Classifiers are defined with the best hyperparameters
    if(classifiername == "svm"):
        classifier = svm.SVC(1, "poly", 2)
    elif(classifiername == "dtree"):
        classifier = tree.DecisionTreeClassifier()
    elif(classifiername == "rforest"):
        classifier = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=12, max_features="auto")

    classifier = classifier.fit(imagearray, labelarray)
    return classifier

def test_classifier(classifier, imagearray, labelarray):
    prediction = classifier.predict(imagearray)
    print metrics.classification_report(labelarray, prediction)

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

dataLimit = 60000
trainImageArray = mnistParser.parse_image_data("train-images.idx3-ubyte", dataLimit)
trainLabelArray = mnistParser.parse_label_data("train-labels.idx1-ubyte", dataLimit)

testImageArray = mnistParser.parse_image_data("t10k-images.idx3-ubyte")
testLabelArray = mnistParser.parse_label_data("t10k-labels.idx1-ubyte")

clf = train_classifier("rforest", trainImageArray, trainLabelArray)
test_classifier(clf, testImageArray, testLabelArray)
export_tree(clf)
cross_validate_classifier(clf, trainImageArray, trainLabelArray)
plot_importances(clf)