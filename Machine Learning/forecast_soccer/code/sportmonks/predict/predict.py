# Author: Bruno Esposito
# Last modified: 07/11/17

#!/usr/bin/python3
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV

import config
import work_data
logger = config.config_logger(__name__, 10)
np.random.seed(1234)


def descriptive_stats(data):
    points = data.replace(
        {'points': {0: 'lose', 1: 'tie', 3: 'win'}})['points']
    goals_points = pd.crosstab(data['goals'],points)

    logger.info('Getting descriptive stats:')
    print('Goals and points:\n{0}'.format(goals_points))
    print('\nPoints frequency:\n{0}'.format(points.value_counts()))
    print('\nGoals frequency:\n{0}'.format(data['goals'].value_counts()))
    return


def preprocess_data(data, draws=False):
    data = data.copy()
    if not draws:
        data = data.loc[data['points'] != 1]
        y_data = data['points'].replace(3, 1)
        logger.info('Obs remaining after excluding draws: {0}'.format(len(data)))
    else:
        y_data = data['points'].replace(3, 2)
    x_data = data[work_data.model_variables()].copy()
    x_data = work_data._splice(x_data, 'opp_', work_data.model_variables())
    return y_data, x_data


def standardize(my_df):
    output = my_df.apply(lambda x: preprocessing.scale(np.array(x)))
    return output


def split(x, y, size=0.25):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size)
    return x_train, x_test, y_train, y_test


def report_model_output(model_output, label):
    logger.info('{2} -- Mean: {0:.3g} -- std: {1:.3g}'.format(np.mean(model_output), np.std(model_output), label))
    return


def logistic_cv(x, y, n_cv=10, lam=1):
    logit = LogisticRegression(C=lam)
    cv_score = cross_val_score(logit, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def logistic_lasso_cv(x, y, n_cv=10, lam=1):
    logit_lasso = LogisticRegression(C=lam, penalty='l1', solver='liblinear')
    cv_score = cross_val_score(logit_lasso, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def svm_cv(x, y, n_cv=10, lam=1):
    svm = SVC(C=lam, kernel='sigmoid')
    cv_score = cross_val_score(svm, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def neural_network_cv(x, y, n_cv=10, layers=(10, 5)):
    nn = MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=layers, random_state=1)
    cv_score = cross_val_score(nn, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def tree_cv(x, y, n_cv=10):
    tree = DecisionTreeClassifier()
    cv_score = cross_val_score(tree, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def adaBoost_cv(x, y, n_cv=10):
    adaBoost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1,
                                                                        criterion='entropy'),
                                  n_estimators=50)
    cv_score = cross_val_score(adaBoost, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def gbm_cv(x, y, n_cv=10):
    gbm = GradientBoostingClassifier(n_estimators=29, max_depth=4, criterion='friedman_mse')
    cv_score = cross_val_score(gbm, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def treeBoost_cv(x, y, n_cv=10):
    treeBoost = ExtraTreesClassifier(n_estimators=50, bootstrap=True)
    cv_score = cross_val_score(treeBoost, x, y, cv=n_cv, scoring=make_scorer(accuracy_score))
    return cv_score


def gbm_grid(x, y, n_cv=10):
    space_est = range(1, 30)
    space_depth = range(1, 5)
    space_loss = ['deviance', 'exponential']
    space_criterion = ['friedman_mse', 'mse']
    param_grid = {'n_estimators': space_est, 'max_depth': space_depth,
                  'criterion': space_criterion}
    gbm = GradientBoostingClassifier()
    grid_gbm = GridSearchCV(gbm, param_grid, cv=n_cv, n_jobs=7)
    grid_gbm.fit(x, y)
    return grid_gbm
