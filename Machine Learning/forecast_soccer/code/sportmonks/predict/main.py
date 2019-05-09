# Author: Bruno Esposito
# Last modified: 07/11/17

#!/usr/bin/python3
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import config
import work_data
import predict


def main():
    t0 = time.time()
    logger = config.config_logger(__name__, 10)

    stats_path = './data/sportmonks/with_data/'
    save_path = './data/sportmonks/'
    create_data = False

    logger.info('Create dataset : {0}'.format(create_data))
    if create_data:
        league_names = work_data.get_data_list(stats_path)
        logger.info('Leagues found: {0}'.format(len(league_names)))

        match_data_tot = pd.DataFrame({})
        for league in league_names:
            logger.info('Opening {0}'.format(league))
            match_data_raw = work_data.load_data(stats_path, selection=league, date_filter='2016-07-13')
            logger.info('Dimensions of raw data: {0}'.format(match_data_raw.shape))
            match_data_raw = work_data.get_selection(match_data_raw)
            match_data_raw = work_data.fill_selected_vars(match_data_raw)
            match_data_raw = work_data.drop_rows_NA(match_data_raw)
            logger.info('Dimensions after cleaning: {0}'.format(match_data_raw.shape))
            match_data_tot = match_data_tot.append(match_data_raw)

        window = 4
        match_data_tot.reset_index(drop=True, inplace=True)
        logger.info('Dimension of all leagues DB: {0}'.format(match_data_tot.shape))
        logger.info('Duplicating vars')
        match_data = work_data.duplicate_stats(match_data_tot)
        logger.info('Finding average with window {0}'.format(window))
        match_data = work_data.get_averages(match_data, window=window)
        logger.info('Dimensions after preprocessing: {0}'.format(match_data.shape))
        logger.info('Stat variables: {0}'.format(len(work_data.stats_variables())))
        match_data.to_csv(save_path + 'sportmonks_final.csv')
        logger.info('Final DB saved')

    logger.info('Opening sportmonks DB')
    match_data = pd.read_csv(save_path + 'sportmonks_final.csv', index_col=1)

    logger.info('Beginning analysis section')
    #predict.descriptive_stats(match_data)

    keep_draws = True
    logger.info('Generating attributes and standardizing - Keep draws: {0}'.format(keep_draws))
    y_data, x_data = predict.preprocess_data(match_data, draws=keep_draws)
    x_data = predict.standardize(x_data)
    logger.info('Number of attributes included: {0}'.format(x_data.shape[1]))
    logger.info('Number of obs: {0}'.format(x_data.shape[0]))
    #print(x_data.describe().transpose().to_string())

    train_nn = False
    train_logit = False
    train_logit_lasso = False
    train_svm = False
    train_tree = False
    train_adaBoost = True
    train_gbm = True
    train_treeBoost = True
    train_staged = False
    n_cv = 10
    lam_list = [0.001, 0.01, 0.05, 0.1, 0.2, 1, 10, 100]
    layers = [(10,5), (20,10), (30,10), (50,10), (100,10), (100,20), (100,30), (100,50,10)]

    if train_nn:
        logger.info('Training models: Neural Network')
        cv_nn = list()
        for layer in layers:
            temp_nn = predict.neural_network_cv(x_data, y_data, n_cv=n_cv, layers=layer)
            cv_nn.append(temp_nn)
            print('.', end='')
        print(' ')
        for i in range(len(layers)):
            predict.report_model_output(cv_nn[i], 'NN_{0}'.format(layers[i]))

    if train_logit:
        logger.info('Training models: Logit')
        cv_logit = list()
        for lam in lam_list:
            temp_logit = predict.logistic_cv(x_data, y_data, n_cv=n_cv, lam=lam)
            cv_logit.append(temp_logit)
            print('.', end='')
        print(' ')
        for i in range(len(lam_list)):
            predict.report_model_output(cv_logit[i], 'Logit_{0}'.format(lam_list[i]))

    if train_logit_lasso:
        logger.info('Training models: Logit_Lasso')
        cv_logit_lasso = list()
        for lam in lam_list:
            temp_logit_lasso = predict.logistic_lasso_cv(x_data, y_data, n_cv=n_cv, lam=lam)
            cv_logit_lasso.append(temp_logit_lasso)
            print('.', end='')
        print(' ')
        for i in range(len(lam_list)):
            predict.report_model_output(cv_logit_lasso[i], 'Logit_Lasso_{0}'.format(lam_list[i]))

    if train_svm:
        logger.info('Training models: SVM')
        cv_svm = list()
        for lam in lam_list:
            temp_svm = predict.svm_cv(x_data, y_data, n_cv=n_cv, lam=lam)
            cv_svm.append(temp_svm)
            print('.', end='')
        print(' ')
        for i in range(len(lam_list)):
            predict.report_model_output(cv_svm[i], 'SVM_{0}'.format(lam_list[i]))

    if train_tree:
        logger.info('Training models: Decision Tree')
        cv_tree = list()
        temp_tree = predict.tree_cv(x_data, y_data, n_cv=n_cv)
        cv_tree.append(temp_tree)
        predict.report_model_output(cv_tree, 'Tree')
        print(' ')

    if train_adaBoost:
        logger.info('Training models: AdaBoost')
        cv_adaBoost = list()
        temp_adaBoost = predict.adaBoost_cv(x_data, y_data, n_cv=n_cv)
        cv_adaBoost.append(temp_adaBoost)
        predict.report_model_output(cv_adaBoost, 'AdaBoost')
        print(' ')

    if train_gbm:
        logger.info('Training models: gbm')
        cv_gbm = list()
        temp_gbm = predict.gbm_cv(x_data, y_data, n_cv=n_cv)
        cv_gbm.append(temp_gbm)
        predict.report_model_output(cv_gbm, 'GBM')
        print(' ')

        logger.info('Grid: gbm')
        grid_gbm = predict.gbm_grid(x_data, y_data)
        print(grid_gbm.best_params_)
        print(grid_gbm.best_score_)

    if train_treeBoost:
        logger.info('Training models: TreeBoost')
        cv_treeBoost = list()
        temp_treeBoost = predict.treeBoost_cv(x_data, y_data, n_cv=n_cv)
        cv_treeBoost.append(temp_treeBoost)
        predict.report_model_output(cv_treeBoost, 'TreeBoost')
        print(' ')

    if train_staged:
        x_train, x_test, y_train, y_test = predict.split(x_data, y_data, size=0.25)
        adaBoost_model = predict.adaBoost(x_train, y_train, n_iter=300)
        adaBoost_model.fit(x_train, y_train)

        xgBoost_model = predict.xgBoost(x_train, y_train, n_iter=300)
        xgBoost_model.fit(x_train, y_train)

        adaBoost_test_errors = []
        xgBoost_test_errors = []

        for adaBoost_test_predict, xgBoost_test_predict in zip(
                adaBoost_model.staged_predict(x_test), xgBoost_model.staged_predict(x_test)):
            adaBoost_test_errors.append(1. - accuracy_score(adaBoost_test_predict, y_test))
            xgBoost_test_errors.append(1. - accuracy_score(xgBoost_test_predict, y_test))

        n_trees_ada = len(adaBoost_test_errors)
        n_trees_xg = len(xgBoost_test_errors)
        print(n_trees_ada, n_trees_xg)
        logger.info('Best accuracy - AdaBoost: {0:.3f}'.format(1 - min(adaBoost_test_errors)))
        logger.info('Best accuracy - xgBoost: {0:.3f}'.format(1 - min(xgBoost_test_errors)))

        #plt.figure(figsize=(15, 5))
        #plt.subplot(131)
        plt.figure()
        plt.plot(range(1, n_trees_ada + 1),
                 adaBoost_test_errors, c='black', label='adaBoost')
        plt.plot(range(1, n_trees_xg + 1),
                 xgBoost_test_errors, c='red', label='xgBoost')
        plt.legend()
        plt.ylim(0.2, 0.8)
        plt.ylabel('Test Error')
        plt.xlabel('Number of Trees')
        plt.show()


    #TODO exclude first and last 4 matches
    #TODO redo CV using GridSearchCV
    #TODO use PCA to preprocess the data
    #TODO implement SoftVoting

    config.time_taken_display(t0)


if __name__ == '__main__':
    main()

