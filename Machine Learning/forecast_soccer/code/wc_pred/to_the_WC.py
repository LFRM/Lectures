import numpy as np
import pandas as pd
import logging
import time
import itertools

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

def sort_table(my_df):
    output = my_df.copy()
    output = output.sort_values(['points', 'dif', 'favor'], 
                 ascending = False)
    return output

def update_standings(standings, results):
    standings = standings.copy()
    results = results.copy()
    for index, row in results.iterrows():
        teamA = row['team_name']
        teamB = row['op_team_name']
        standings.loc[teamA, 'points'] = \
            standings.loc[teamA,'points'] + row['points'] 
        standings.loc[teamB, 'points'] = \
            standings.loc[teamB,'points'] + row['op_points']
        standings.loc[teamA, 'dif'] = \
            standings.loc[teamA,'dif'] + row['dif'] 
        standings.loc[teamB, 'dif'] = \
            standings.loc[teamB,'dif'] + row['op_dif']
    return standings

def check_top5(standings, name):
    top5 = standings.index[:5]
    #print(top5)
    if name in top5:
        return 1
    else:
        return 0

def gen_points(outcomes):
    if outcomes:
        return 3
    else:
        return 0

def gen_op_points(outcomes):
    if outcomes:
        return 0
    else:
        return 3

def gen_dif(outcomes):
    if outcomes:
        return 1
    else:
        return -1

def gen_op_dif(outcomes):
    if outcomes:
        return -1
    else:
        return 1


def main():
    target_country = 'Argentina'

    config_logger('logger', 10)
    t0 = time.time()
    standings_or = pd.read_csv('./data/to_the_WC/standings.csv', 
                    index_col = 0,
                    header = 0)
    probs_or = pd.read_csv('./data/to_the_WC/probabilities.csv',
                index_col = 0,
                header = 0)

    print(standings_or)
    print(probs_or)

    standings_or = standings_or.set_index('team')
    probs_or['predicted'] = probs_or['predicted']/100
    probs_or['op_points'] = 0
    probs_or['points'] = 0


    odds = []
    peru_WC = []
    space = list(itertools.product((0,1), repeat = 10))
    for outcome in space:
        probs = probs_or
        standings = standings_or 
        probs['outcome'] = pd.Series(outcome)
        #print(probs)
        temp_odds = []
        for index, row in probs.iterrows():
            if row['outcome'] == 1:
                temp_odds.append(row['predicted'])
                #row['points'] = 3
                #row['op_points'] = 0
            else:
                temp_odds.append(1-row['predicted'])
                #row['points'] = 0
                #row['op_points'] = 3


        probs['points'] = probs['outcome'].apply(gen_points)
        probs['op_points']= probs['outcome'].apply(gen_op_points)
        probs['dif'] = probs['outcome'].apply(gen_dif)
        probs['op_dif']= probs['outcome'].apply(gen_op_dif)
        standings = update_standings(standings, probs)
        standings = sort_table(standings)
        peru_in_WC = check_top5(standings, target_country)
        #print(peru_in_WC) 
        #print(probs)
        #print(standings)

        odds.append(np.prod(temp_odds))
        peru_WC.append(peru_in_WC)
        

    
    print(sum(odds))
    print(sum(peru_WC))
    final_prob = np.dot(odds, peru_WC)
    logger.info('Probability of getting to the WC for {0}: {1:.3f}'.format(
                target_country, final_prob))

    time_taken_display(t0)





if __name__=='__main__':
    main()


