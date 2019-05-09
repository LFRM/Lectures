# Author: Bruno Esposito
# Last modified: 07/11/17

#!/usr/bin/python3
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import config
logger = config.config_logger(__name__, 10)


def get_data_list(path):
    files_list = [x for x in listdir(path) if isfile(join(path, x))]
    return files_list


def load_data(path, selection=None, date_filter = '2016-07-13'):
    if selection:
        file_name = path + selection
        dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
        output = pd.read_csv(file_name, index_col=0, parse_dates=['time.starting_at.date'],
                             date_parser=dateparser)
        output = output[output['stats_available'] == 1]
        output = output[output['time.starting_at.date'] > date_filter]
    else:
        #TODO import all files in path
        output = []
    return output


def gen_is_home(team_series, local_series):
    output = []
    for x,y in zip(team_series,local_series):
        if x == y:
            output.append(1)
        else:
            output.append(0)
    return output


def gen_local_visit(is_home, local_var, visit_var):
    output = []
    for (x,local,visit) in zip(is_home,local_var,visit_var):
        if x == 1:
            output.append(local)
        else:
            output.append(visit)
    return output


def gen_score(is_home, local_score, visit_score):
    outputA = []
    outputB = []
    for (x,local,visit) in zip(is_home,local_score,visit_score):
        if x == 1:
            outputA.append(local)
            outputB.append(visit)
        else:
            outputA.append(visit)
            outputB.append(local)
    return outputA,outputB


def gen_points(team_score, op_score):
    output = []
    for (team,op) in zip(team_score,op_score):
        if team == op:
            output.append(1)
        elif team > op:
            output.append(3)
        else:
            output.append(0)
    return output


def extract_position_from_str(my_str, pos):
    if pd.isnull(my_str):
        return np.nan

    formation = [int(s) for s in my_str.split('-') if s.isdigit()]
    if pos == 1:
        output = formation[0]
    elif pos == 3:
        output = formation[-1]
    elif pos == 2:
        output = sum(formation[1:-1])
    else:
        logger.error('pos argument must be 1,2 or 3')
        return np.nan
    return output


def get_selection(my_df):
    my_df = my_df.copy()
    selected = [x for x in selected_variables() if x not in variables_with_NA()]
    output = my_df[selected].copy()
    output['is_home'] = gen_is_home(output['team_id'], output['localteam_id'])

    locality = output['is_home']
    output['team_name'] = gen_local_visit(locality, output['localTeam.data.name'],
                                         output['visitorTeam.data.name'])
    output['coach_id'] = gen_local_visit(locality, output['coaches.localteam_coach_id'],
                                         output['coaches.visitorteam_coach_id'])
    output['goals']= gen_local_visit(locality,output['scores.localteam_score'],
                                     output['scores.visitorteam_score'])
    output['op_goals'] = gen_local_visit(locality, output['scores.visitorteam_score'],
                                         output['scores.localteam_score'])
    output['formation'] = gen_local_visit(locality, output['formations.localteam_formation'],
                                          output['formations.visitorteam_formation'])
    output['n_defenders'] = output['formation'].map(lambda x: extract_position_from_str(x,1))
    output['n_midfielders'] = output['formation'].map(lambda x: extract_position_from_str(x,2))
    output['n_strikers'] = output['formation'].map(lambda x: extract_position_from_str(x,3))
    output['points']= gen_points(output['goals'], output['op_goals'])
    output['avg_points'] = output['points']
    output['avg_goals'] = output['goals']
    output = output.drop(variables_to_drop(), axis=1)
    return output


def variables_to_drop():
    my_vars = ['coaches.localteam_coach_id','coaches.visitorteam_coach_id',
               'scores.localteam_score','scores.visitorteam_score',
               'formations.localteam_formation', 'formations.visitorteam_formation',
               'localteam_id','visitorteam_id','stats_available', 'localTeam.data.name',
               'visitorTeam.data.name']
    return my_vars


def selected_variables():
    my_vars = ['league_id','season_id','fixture_id','localteam_id','visitorteam_id',
               'team_id','time.starting_at.date','localTeam.data.name',
               'visitorTeam.data.name','scores.localteam_score',
               'scores.visitorteam_score','attacks.attacks',
               'attacks.dangerous_attacks','attendance','coaches.localteam_coach_id',
               'coaches.visitorteam_coach_id','corners',
               'formations.localteam_formation','formations.visitorteam_formation',
               'fouls','free_kick','goal_kick',
               'offsides','passes.accurate','passes.percentage','passes.total',
               'possessiontime','redcards','referee_id','round_id','saves',
               'shots.blocked','shots.insidebox','shots.offgoal','shots.ongoal',
               'shots.outsidebox','shots.total','standings.localteam_position',
               'standings.visitorteam_position','stats_available','substitutions',
               'throw_in','yellowcards','time.minute']
    return my_vars


def variables_with_NA():
    my_vars = ['attacks.attacks','attacks.dangerous_attacks','attendance','free_kick',
               'goal_kick','standings.localteam_position','standings.visitorteam_position',
               'substitutions','throw_in']
    return my_vars


def fill_selected_vars(my_df):
    my_df = my_df.copy()
    for var in variables_for_fill_zero():
        my_df[var] = my_df[var].fillna(0)
    return my_df


def variables_for_fill_zero():
    my_vars = ['corners','fouls','offsides','redcards','saves','yellowcards']
    return my_vars


def drop_rows_NA(my_df):
    my_df = my_df.copy()
    output = my_df.dropna(axis=0, inplace=False)
    return output


def remove_unique(my_df):
    my_df = my_df.copy()
    temp = my_df['fixture_id'].value_counts()
    targets = temp[temp == 1].index.tolist()
    logger.info('Matches without complete stats: {0}'.format(len(targets)))
    output = my_df.loc[my_df['fixture_id'].map(lambda x: x not in targets)]
    return output


def duplicate_stats(my_df):
    my_df = my_df.copy()
    my_df = remove_unique(my_df)
    my_df = my_df.sort_values('fixture_id')
    target = my_df[variables_to_duplicate()]
    target = _splice(target, 'op_', variables_to_duplicate())
    my_df.drop(variables_to_duplicate(), inplace=True, axis=1)
    output = pd.concat([my_df, target], axis=1)
    op_goals = output['op_goals'].replace(0, 1)
    op_passes = output['op_passes.accurate'].replace(0, 1)
    op_shots = output['op_shots.total'].replace(0, 1)
    output['goals_ratio'] = output['goals'].divide(op_goals)
    output['passes_ratio'] = output['passes.accurate'].divide(op_passes)
    output['shots_ratio'] = output['shots.total'].divide(op_shots)
    return output


def _swap_pairwise(col):
    """ Swap rows pairwise; i.e. swap row 0 and 1, 2 and 3, etc.  """
    col = pd.np.array(col)
    for index in range(0, len(col), 2):
        val = col[index]
        col[index] = col[index + 1]
        col[index + 1] = val
    return col


def _splice(data, prefix, variables):
    """ Splice both rows representing a game into a single one. """
    data = data.copy()
    op = data.copy()
    op_cols = [prefix + x for x in variables]
    op.columns = op_cols
    op = op.apply(_swap_pairwise)
    output = pd.concat([data, op], axis=1)
    return output


def variables_to_duplicate():
    my_vars = ['points','avg_points','avg_goals','corners','fouls','offsides','passes.accurate',
               'passes.percentage','passes.total','redcards','saves',
               'shots.blocked','shots.insidebox','shots.offgoal','shots.ongoal',
               'shots.outsidebox','shots.total','yellowcards', 'formation', 'n_defenders',
               'n_midfielders', 'n_strikers', 'team_id', 'team_name']
    return my_vars


def get_averages(my_df, window=4):
    my_df = my_df.copy()
    teams = set(my_df['team_id'].values)
    output = pd.DataFrame({})
    for team in teams:
        temp = my_df[my_df['team_id'] == team].copy()
        if len(temp) < window + 1:
            continue
        temp.sort_values('time.starting_at.date', axis=0, inplace=True)
        #if team == 18:
        #    print(temp.head(8).to_string())
        temp_stats = temp[stats_variables()].copy()
        temp_stats[variables_90min()] = temp_stats[variables_90min()].apply(lambda x: x/temp['time.minute'])
        temp_stats[variables_100perct()] = temp_stats[variables_100perct()].apply(lambda x: x/100)
        temp_stats = temp_stats.apply(lambda x: x.rolling(window).mean()).shift()
        temp = temp[non_stats_variables()].copy().join(temp_stats)
        temp['coach_id'] = temp['coach_id'].map(lambda x: x.is_integer()).astype('int')
        output = output.append(temp)

    output = drop_rows_NA(output)
    output = remove_unique(output)
    output.sort_values('fixture_id', axis=0, inplace=True)
    output.reset_index(drop=True, inplace=True)
    logger.info('Matches removed beacuse one team did not have enough info: {0}'.format(len(my_df)-len(output)))
    return output


def variables_90min():
    my_vars = ['avg_goals','corners','fouls','offsides','passes.accurate','passes.total',
               'redcards','saves','shots.blocked','shots.insidebox','shots.offgoal',
               'shots.ongoal','shots.outsidebox','shots.total','yellowcards']
    my_vars = my_vars + ['op_'+x for x in my_vars if x not in no_duplicate_vars()]
    return my_vars


def variables_100perct():
    my_vars = ['passes.percentage','possessiontime']
    my_vars = my_vars + ['op_' + x for x in my_vars if x not in no_duplicate_vars()]
    return my_vars


def no_duplicate_vars():
    return ['coach_id','possessiontime','passes_ratio','shots_ratio','goals_ratio','is_home']


def stats_variables():
    my_vars = ['avg_goals','avg_points','corners','fouls','offsides',
               'passes.accurate','passes.percentage','passes.total','possessiontime',
               'redcards','saves','shots.blocked','shots.insidebox','shots.offgoal',
               'shots.ongoal','shots.outsidebox','shots.total','yellowcards', 'coach_id',
               'n_defenders','n_midfielders','n_strikers', 'goals_ratio', 'passes_ratio',
               'shots_ratio']
    my_vars = my_vars + ['op_'+x for x in my_vars if x not in no_duplicate_vars()]
    return my_vars


def non_stats_variables():
    my_vars = ['league_id','season_id','fixture_id','time.starting_at.date','team_id','op_team_id',
               'team_name','op_team_name','goals','op_goals','points','is_home',
               'op_points','round_id','referee_id','formation','op_formation','time.minute']
    return my_vars


def model_variables():
    my_vars = ['is_home','avg_goals','avg_points','corners','fouls','offsides',
               'passes.accurate','passes.percentage','passes.total','possessiontime',
               'redcards','saves','shots.blocked','shots.insidebox','shots.offgoal',
               'shots.ongoal','shots.outsidebox','shots.total','yellowcards', 'coach_id',
               'n_defenders','n_midfielders','n_strikers', 'goals_ratio', 'passes_ratio',
               'shots_ratio']
    my_vars = my_vars + ['op_'+x for x in my_vars if x not in no_duplicate_vars()]
    return my_vars
