"""

Greedy Feature Selection using Logistic Regression as base model
to optimize Area Under the ROC Curve

__author__ : Abhishek

Credits : Miroslaw @ Kaggle

"""


import numpy as np

import sklearn.linear_model as lm
from sklearn import metrics, preprocessing


class greedyFeatureSelection(object):

    def __init__(self, data, labels, scale=1, verbose=0):
        if scale == 1:
            self._data = preprocessing.scale(np.array(data))
        else:
            self._data = np.array(data)
        self._labels = labels
        self._verbose = verbose

    def evaluateScore(self, X, y):
        model = lm.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc

    def selectionLoop(self, X, y):
        score_history = []
        good_features = set([])
        num_features = X.shape[1]
        while len(score_history) < 2 or score_history[-1][0] > score_history[-2][0]:
            scores = []
            for feature in range(num_features):
                if feature not in good_features:
                    selected_features = list(good_features) + [feature]

                    Xts = np.column_stack(X[:, j] for j in selected_features)

                    score = self.evaluateScore(Xts, y)
                    scores.append((score, feature))

                    if self._verbose:
                        print "Current AUC : ", np.mean(score)

            good_features.add(sorted(scores)[-1][1])
            score_history.append(sorted(scores)[-1])
            if self._verbose:
                print "Current Features : ", sorted(list(good_features))

        # Remove last added feature
        good_features.remove(score_history[-1][1])
        good_features = sorted(list(good_features))
        if self._verbose:
            print "Selected Features : ", good_features

        return good_features

    def transform(self, X):
        X = self._data
        y = self._labels
        good_features = self.selectionLoop(X, y)
        return X[:, good_features]
