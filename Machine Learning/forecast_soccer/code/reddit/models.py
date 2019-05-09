# Forked from https://github.com/sharan-naribole/reddit-sentiment-soccer-prediction
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import learning_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression


def anova_test(df,flair_type):
    '''
    Create an ANOVA test. It is just a mean t-test done through OLS.
    '''
    anova_df = df.loc[df['Flair Type']==flair_type].dropna()
    mod = ols('Sentiment_Polarity ~ Result',
                data=anova_df).fit()
    return(sm.stats.anova_lm(mod, typ=2))


def generate_accuracy_curve(clf, X_train, y_train, label_train, label_valid, 
                            color_train = 'blue', color_valid='green'):
    """
    Function to generate training and validation curves
    """
  
    # The learning curve function will generate a tuple containing thre vectors:
    #   1. Contains the number of observations included in the trained model
    #   2. Contains the accuracy of each fold (there are 10 fols) when
    #      predicting train data.
    #   3. Contains the accuracy of each fold when predicting test data.
    train_sizes, train_scores, test_scores =\
                learning_curve(estimator=clf,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)
    # print(train_sizes)
    # print('\n', train_scores)
    # print('\n', test_scores)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot the average accuracy predicting within the sample (training set)    
    plt.plot(train_sizes, train_mean,
             color=color_train, marker='o',
             markersize=5, label=label_train)

    # Plot +-1 standard deviation
    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    # Plot the average accuracy predicting outside the sample (test set)
    plt.plot(train_sizes, test_mean,
             color=color_valid, linestyle='--',
             marker='s', markersize=5,
             label=label_valid)

    # Plot +-1 standard deviation
    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')
    
    return

