# Forked from https://github.com/sharan-naribole/reddit-sentiment-soccer-prediction
# I have done minor modifications
# Local modules
import preprocess
import sentiment
import models

# General
import logging
import time

# Data Collection and Transformations
import praw
import numpy as np
import pandas as pd
import datetime as dt
import pickle
from sklearn.preprocessing import Imputer 
#import unicodedata
#import re
#import editdistance


# Sentiment Analysis
from textblob import TextBlob

# Statistical Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
import random
#import scipy

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Class imbalance 
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

# Plotting 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline


matplotlib.style.use('ggplot')
#sns.set(style="white")
plt.rcParams['figure.figsize'] = [10,8]

random.seed(1234)
logging.getLogger('imblearn.base').setLevel(40)


# ======== FUNCTIONS =========
# --------- GENERAL ----------
def check_negative(number):
    ''' Raise error if number is below zero. '''
    if number < 0:
        logger.error('{0} is negative'.format(number))
        raise ValueError('{0} should not be negative'.format(number))

def delta_time_in_HMS(begin, finish = None):
    ''' Find how much time has passed between begin and finish. '''
    raw = finish - begin
    hours = round(raw // 3600)
    minutes = round((raw % 3600) // 60)
    seconds = round((raw % 3600) % 60)
    return (raw, hours, minutes, seconds)

def time_taken_display(begin, finish = None):
    ''' 
    Display in logger how much time has passed between 
    begin and finish. 
    '''
    if finish == None:
        finish = time.time()

    if finish < begin:
        logger.error('Finish time lower than begin time. '
                     'Begin: {0} - Finish: {1}'.format(begin, finish))
        raise ValueError('Finish time cannot be lower than begin time')

    [check_negative(x) for x in (begin, finish)]

    raw, hours, minutes, seconds = delta_time_in_HMS(begin, finish)
    logger.debug('Excecution took a raw time of {0} seconds'.format(
                     round(raw, 5)))
    logger.info(('Excecution took {0} hours, {1} minutes and '
                '{2} seconds').format(hours, minutes, seconds))


# ---------- CONFIG ----------
def config_logger(name, level = 10):
    ''' Config logger output with level 'level'. '''
    logging.basicConfig(
                level = level,
                format = ('%(asctime)s - %(name)s - '
                          '%(levelname)s - %(message)s'))
    global logger
    logger = logging.getLogger(name)
    return logger


# ---------- MAIN ------------
def main():

    RAW_OUTPUT = './data/reddit/raw_data.csv'
    RAW_PICKLE = './data/reddit/raw_pickle.pckl'
    GRAPH_PATH = './output/graphs/reddit/'
    LOGGER_LEVEL = 20
    EXTRACT_DATA = False

    t0 = time.time()
    config_logger('logger', LOGGER_LEVEL)

    cfg_parser = configparser.ConfigParser()
    cfg_parser.read(cfg_name)
    user_agent = str(cfg_parser.get('Reddit', 'user_agent'))
    client_id = str(cfg_parser.get('Reddit', 'client_id'))
    client_secret = str(cfg_parser.get('Reddit', 'client_secret'))
    username = str(cfg_parser.get('Reddit', 'username'))
    password = str(cfg_parser.get('Reddit', 'password'))

    # -------------- Extract Data ---------------
    if EXTRACT_DATA:
        reddit = praw.Reddit(
                 user_agent=user_agent,
                 client_id=client_id,
                 client_secret=client_secret,
                 username=username,
                 password=password)

        logger.info('Bot used: {0}'.format(reddit.user.me()))

        # Datetime objects
        start_date = dt.date(2013,1,1)
        end_date = dt.date(2017,4,5)
        logger.info('Start: {0}'.format(start_date))
        logger.info('End: {0}'.format(end_date))

        # Converting to UNIX timestamp
        start_time = int(time.mktime(start_date.timetuple()))
        end_time = int(time.mktime(end_date.timetuple()))

        subreddit = reddit.subreddit('soccer')
        #Typical terms in a submission on /r/soccer for Post-Match Thread
        query = "(and title:'post-match thread')"
        #List of Submission instances
        submissions = subreddit.submissions(
                      start = start_time, 
                      end = end_time, 
                      extra_query = query)

        # -------------- Transform Data --------------
        data = []
        counter = 0

        for submission in submissions:
    
            # Print characteristics of the submission
            title = preprocess.string_normalize(submission.title)
            logger.info('Processing: {0} -- {1}'.format(
                        time.ctime(submission.created), title))

            # Update counts
            counter += 1
    
            # Excluding threads with less than 100 comments
            if(submission.num_comments < 100):
                continue
    
            # Get Match Info
            match_info = preprocess.get_match_info(title)
    
            # If title doesn't match with regex format
            if not match_info:
                continue
            else:
                # List for current record
                record = []
        
                # Score: 
                # + 1 if Team A won
                # 0 if Team B won
                # -1 if Team B won
                match_result = int(match_info[1]) - int(match_info[2])
                if match_result:
                    match_result = match_result/abs(match_result)
                record.append(match_result)
    
                # Title
                record.append(title)
    
                # Submission Creation Time
                creation_time = time.ctime(submission.created)
                record.append(creation_time)
                #print(creation_time)
    
                # Team Names
                team_a = match_info[0].strip()
                team_b = match_info[3].strip()
                record.append(team_a)
                record.append(team_b)
        
                # Get the four sentiment metrics for Team A, Team B and 
                # other comments.
                polarity_metrics = sentiment.submission_sentiment(
                                   submission, 
                                   team_a,
                                   team_b,
                                   Ntop = 10)
                record += polarity_metrics
        
                # Submission Score and Number of Comments
                record.append(submission.num_comments)
                record.append(submission.score)
        
                # Append to the main list
                data.append(record)  
   
        logger.infor('There are {0} raw submissions'.format(counter))
        logger.info('Attributes:\n{0}'.format(submission.__dict__.keys()))
        
        # Create dataframe with extracted data
        logger.info('Creating dataframe')
        raw_df = pd.DataFrame(data)
        raw_df.columns = (["Result","Title","Time","Home Team","Away Team",
                           "Home Team Score Pol","Home Team Subject Pol",
                           "Home Team Pol SD","Home Team Merged Pol",
                           "Away Team Score Pol","Away Team Subject Pol",
                           "Away Team Pol SD","Away Team Merged Pol",
                           "Others Score Pol","Others Subject Pol",
                           "Others Pol SD","Others Merged Pol","Comments",
                           "Submission Score"])
 
        logger.info('Obervations: {0} -- Variables: {1}'.format(
                    raw_df.shape[0], raw_df.shape[1]))
        
        # Save as CSV
        logger.info('Saving raw data: {0}'.format(RAW_OUTPUT))
        raw_df.to_csv(RAW_OUTPUT)

        # Save as pickle
        logger.info('Saving raw pickle: {0}'.format(RAW_PICKLE))
        a = open(RAW_PICKLE, 'wb')
        pickle.dump(raw_df, a)
        a.close()
        # ----------------------------------------
    
    # Open raw pickle datset
    logger.info('Opening raw dataframe: {0}'.format(RAW_PICKLE))
    b = open(RAW_PICKLE, 'rb')
    raw_df = pickle.load(b)
    b.close()

    logger.info('Obervations: {0} -- Variables: {1}'.format(
                    raw_df.shape[0], raw_df.shape[1]))

    # -------------- Process Data ---------------
    # Split data in train and test datasets
    train_df, test_df = train_test_split(
                            raw_df, 
                            test_size = 0.1,
                            random_state = 1234)
    #print(train_df.head())

    # Keep only relevant freatures for training
    X_train = train_df.iloc[:,5:-2]
    y_train = train_df.iloc[:,0]

    X_test = test_df.iloc[:,5:-2]
    y_test = test_df.iloc[:,0]

    logger.info('Shape of the train sample: X:{0} -- Y:{1}'.format(
                X_train.shape, y_train.shape))
    logger.info('Shape of the test sample: X:{0} -- Y:{1}'.format(
                X_test.shape, y_test.shape))
    logger.info('Train dataset description:\n{0}'.format(
                X_train.describe()))

    # Fill inf observations with NAs
    logger.info('Fill inf with NAs')
    X_train.replace([np.inf,-np.inf],np.nan,inplace=True)
    X_test.replace([np.inf,-np.inf],np.nan, inplace=True)
   
    # Graph histogram of outcome
    sns.distplot(y_train,kde=False)
    plt.ylabel("Frequency")
    plt.savefig(GRAPH_PATH + 'outcome_freq.png')
    plt.close()

    # ------------ Descriptives stats ------------
    # Merge X and Y of train dataset
    train_eda = pd.concat([X_train,y_train],axis=1,join='outer')
    logger.info('Train dataset shape: {0}'.format(train_eda.shape))


    # Reshaping the dataframe with our metrics of interest
    df_melt_score = pd.melt(train_eda,
                    id_vars = ['Result'],
                    value_vars = ['Home Team Score Pol',
                                  'Away Team Score Pol',
                                  'Others Score Pol'],
                    var_name = 'Flair Type',
                    value_name = 'Sentiment_Polarity')
 
    df_melt_score['Flair Type'] = df_melt_score['Flair Type'].apply(
                                  lambda x: x.split(' Score')[0])
    #print(df_melt_score)

    # Generage Box graph for polarity weighted by score
    plt.figure(figsize=(15,10))
    g = sns.factorplot(x='Sentiment_Polarity', y='Flair Type', row='Result',
                       data=df_melt_score[df_melt_score.notnull()],
                       orient='h', size=2, aspect=3.5, palette='Set3',
                       kind='box')
    plt.xlim(-0.25,0.5)
    plt.savefig(GRAPH_PATH + 'box_scorePol.png')
    plt.close()

    # Generate histograms for polarity weighted by score
    grid = sns.FacetGrid(df_melt_score, 
                         col='Result', 
                         row='Flair Type', 
                         size=2.2, 
                         aspect=1.6, 
                         xlim=(-1,1))
    grid.map(plt.hist, 'Sentiment_Polarity', alpha=.5, bins=20)
    grid.add_legend()
    plt.savefig(GRAPH_PATH + 'hist_scorePol.png')
    plt.close()

    # Generate ANOVA tests (t-tests with OLS) to see if polarity weighted
    # by score if significantly different for each result. Do this for each
    # flair type.
    logger.info('Generate t-tests for polarity grouped by result')
    logger.info('HOME TEAM ANOVA test:\n{0}'.format(
                models.anova_test(df_melt_score,'Home Team')))
    logger.info('AWAY TEAM ANOVA test:\n{0}'.format(
                models.anova_test(df_melt_score,'Away Team')))
    logger.info('Others ANOVA test:\n{0}'.format(
                models.anova_test(df_melt_score,'Others')))

    # ------------ Fix NAs ------------
    logger.info('Count NAs:\n{0}'.format(X_train.isnull().sum()))

    # Impute median value to NAs
    Ndim = 13
    imr = Imputer(missing_values='NaN', strategy='median', axis=0)
    impute_df_train = imr.fit_transform(np.array(X_train.iloc[:,:Ndim]))
    impute_df_test = imr.transform(np.array(X_test.iloc[:,:Ndim]))

    X_train.iloc[:,:Ndim] = impute_df_train
    X_test.iloc[:,:Ndim] = impute_df_test

    # ---------- More descriptives stats ---------
    # Create a heatmap of the correlation matrix
    train_corr = X_train.corr()
    fig = plt.figure(figsize=(8,10))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(train_corr,cmap = cmap, xticklabels=True, yticklabels=True, 
                vmax=1, center=0, square=True, linewidths=.5, 
                cbar_kws={"shrink": .5})
    plt.yticks(rotation=0)
    plt.xticks(rotation=90) 
    plt.tight_layout()
    plt.savefig(GRAPH_PATH + 'corr_trainSet.png')
    plt.close()


    # --------------- Classificator --------------
    # Create a classificator using sentiment variables to predict if the 
    # match was (i) win/lose or (ii) draw. 
    # Create new outcome :
    #   - = 1 if win or lose
    #   - = 0 if draw
    y_train = np.abs(y_train)
    y_test = np.abs(y_test)
    
    # Plot new oucome histogram
    sns.set(font_scale=1.5) 
    sns.distplot(y_train,kde=False)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(GRAPH_PATH + 'classif_outcome_freq.png')
    plt.close()

    # Plot accuracy curves
    plt.figure(figsize=(10,8))
    plt.axhline(y=0.8, linestyle = '--', color = 'red',)

    # Generate validation curves for two methodologies:
    #   - Logistic regression
    #   - SMOTE logistic regression
    sm = SMOTE(random_state=42)
    scl = StandardScaler()
    clf = LogisticRegression(random_state=0)

    pipe_lr = make_pipeline(scl,clf)
    pipe_lr_smote = make_pipeline(sm,scl,clf)

    (models.generate_accuracy_curve(pipe_lr, X_train, y_train,
                            "Training Accuracy w/o SMOTE",
                            "Validation Accuracy w/o SMOTE",
                            "blue","green"))

    (models.generate_accuracy_curve(pipe_lr_smote, X_train, y_train,
                            "Training Accuracy w/ SMOTE",
                            "Validation Accuracy w/ SMOTE",
                            "magenta","cyan"))

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right',fontsize = 'medium')
    plt.ylim([0, 1.0])
    plt.tight_layout()
    plt.savefig(GRAPH_PATH + 'learning_curve.png')
    plt.close()

    # --------------- Voting Classifier --------------
    RANDOM_STATE_SMOTE = 42    
    sm = SMOTE(random_state=RANDOM_STATE_SMOTE)    
    sc = StandardScaler()
    clf_lr = LogisticRegression(penalty='l2', 
                                C=1,
                                random_state=1)
    clf_gnb = GaussianNB()
    clf_knn = KNeighborsClassifier(n_neighbors=1,
                                   p=2,
                                   metric='minkowski')
    clf_rf = RandomForestClassifier()

    #TODO Check results of these metrics without SMOTE 
    #TODO Include results for Logit SMOTE
    pipe_lr = make_pipeline(sc,clf_lr)
    pipe_lr_smote = make_pipeline(sm,sc,clf_lr)
    pipe_gnb = make_pipeline(sm,sc,clf_gnb)
    pipe_knn = make_pipeline(sm,sc,clf_knn)
    pipe_rf = make_pipeline(sm,sc,clf_rf)

    clf_vote_soft = VotingClassifier(estimators=[
                    ('lr', pipe_lr), ('rf', pipe_rf), ('gnb', pipe_gnb),
                    ('knn', pipe_knn)], voting='soft')

    clf_labels = ['Logistic Regression', 'Random Forest', 'Gaussian NB', 
                  'KNN','Voting - Soft']
    clf_list = [pipe_lr, pipe_rf, pipe_gnb, pipe_knn, clf_vote_soft]
    logger.info('10-fold cross validation:')
    for clf, label in zip(clf_list, clf_labels):
        scores = cross_val_score(estimator=clf,
                                 X=X_train,
                                 y=y_train,
                                 cv=10,
                                 scoring='roc_auc')
        print('ROC AUC: {0:.2f} (+/- {1:.2f}) [{2}]'.format(
              scores.mean(), scores.std(), label)) 

    # ----------- Classification report ----------
    #TODO implement these reports for more estimators
    #TODO seems like there are too little observations in the test data.
    #     Perhaps we should consider more.

    # Each classification report generates:
    #   1. Precision
    #   2. Recall
    #   3. F1-score (harmonic mean of precision and recall)
    #   4. support (number of ocurrences for each class)

    # Classification report for Naive Bayes
    pipe_gnb.fit(X_train,y_train)
    y_pred = pipe_gnb.predict(X_test)
    print(classification_report(
              y_test, y_pred, target_names=['Draw','Win/Loss']))

    # Classification report for Random Forrest
    pipe_rf.fit(X_train,y_train)
    y_pred = pipe_rf.predict(X_test)
    print(classification_report(
              y_test, y_pred, target_names=['Draw','Win/Loss']))


    time_taken_display(t0)

if __name__ == '__main__':
    main()

