# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 20:16:46 2021

@author: Rafa
"""


from sklearn.datasets import load_digits
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

# =============================================================================
# KNN con 1 partición estratificada de train/test y con validación cruzada.
# =============================================================================

X, y = load_digits(return_X_y = True)

# x_plot = np.reshape(X[0, :], (8, 8))
# from matplotlib import pyplot as plt

# plt.imshow(x_plot)

sss = StratifiedShuffleSplit(n_splits = 10, test_size = 0.1, random_state = 0)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2, 3]}

from sklearn.model_selection import GridSearchCV

miModelo = KNeighborsClassifier()

# =============================================================================
# https://scikit-learn.org/stable/modules/model_evaluation.html
# =============================================================================

gs = GridSearchCV(estimator = miModelo, param_grid = params, scoring = 'accuracy', cv = 5)
gs.fit(X_train, y_train)

kk = gs.cv_results_

## Analizar cv results y entender bien que es cada cosa.

clfBest= gs.best_estimator_

clfBest.fit(X_train, y_train)

y_pred = clfBest.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(clfBest, X_test, y_test)
