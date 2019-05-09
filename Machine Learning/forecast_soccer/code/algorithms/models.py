import pandas as pd
import numpy as np
import math
import code.algorithms.config as config
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson, skellam

logger = config.config_logger(__name__, 10)

random.seed(1111)


class Fixture(object):
    """
    Class for the fixture of a league.
    """
    def __init__(self, fixture, name, local_fixture=True):
        """
        Notes: The are certain headers that the my_fixture arg must have. Careful.
        Args:
            fixture (:obj: `pd.Dataframe`): fixture of a league.
            name (str): name of the fixture.
            local_fixture (bool): True if all local team ids are included in a single
                column named 'localteam_id'. False if locality is determined by a
                column names 'is_home' and team ids are under the column 'team_id'.
        """
        self.local_fixture = local_fixture
        self.fixture = fixture
        self.name = name

    @property
    def fixture(self):
        return self._fixture

    @fixture.setter
    def fixture(self, fixture):
        if self.local_fixture:
            if 'localTeam.data.name' not in list(fixture.columns.values):
                raise ValueError('localTeam.data.name is missing but local_fixture is True')
            else:
                self._fixture = fixture
        else:
            if 'is_home' not in list(fixture.columns.values):
                raise ValueError('is_home is missing but local_fixture is False')
            else:
                self._fixture = fixture

    def __str__(self):
        """
        Returns:
            Print name of the league and dimensions of the fixture DataFrame.
        """
        return 'League: {0} - Shape: {1}'.format(self.name, self.fixture.shape)

    def get_last_match(self):
        return max(pd.to_datetime(self.fixture['time.starting_at.date'], format="%Y-%m-%d"))

    def get_match_years(self):
        return self.fixture['time.starting_at.date'].apply(lambda x: x[:4])

    def get_last_year(self):
        temp_year = max(set(self.get_match_years()))
        last_match = self.get_last_match()
        if last_match.month < 6:
            return int(temp_year)
        else:
            return int(temp_year) + 1

    def get_seasons(self):
        """
        Get set with the seasons in the fixture dataset.
        Returns:
            Set with the seasons in the fixture dataset.
        """
        if 'season_id' not in list(self.fixture.columns.values):
            raise ValueError('season_id not in fixture')
        else:
            return set(self.fixture['season_id'])

    def get_team_ids(self):
        if self.local_fixture:
            return set(self.fixture['localteam_id'])
        else:
            return set(self.fixture['team_id'])

    def get_team_names(self):
        if self.local_fixture:
            return set(self.fixture['localTeam.data.name'])
        else:
            return set(self.fixture['Team.data.name'])

    def subset_season(self, season_id):
        """
        Extract matches played in a certain season.
        Args:
            season_id (str): id of the season requested.

        Returns:
            Fixture object with the games played in the season requested.
        """
        df = self.fixture
        local_fixture = self.local_fixture
        output = df.loc[df['season_id'] == season_id]
        name = self.name + ' - season {0}'.format(season_id)
        return Fixture(fixture=output, name=name, local_fixture=local_fixture)

    def get_team_games(self, team_id, home):
        """
        Extract the games of certain team as local, visit or both.
        Args:
            team_id (str): id of the requested team.
            home (int: from 0 to 2): 0 for visit games.
                                     1 for local games.
                                     2 for both.

        Returns:
            Fixture object containing the games of certain team.
        """
        df = self.fixture
        local_fixture = self.local_fixture
        if local_fixture:
            team_name = df.loc[df['localteam_id'] == team_id]['localTeam.data.name'].iloc[0]
            name = self.name + ' - {0}'.format(team_name)
            if home == 0:
                output = df.loc[df['visitorteam_id'] == team_id]
            elif home == 1:
                output = df.loc[df['localteam_id'] == team_id]
            elif home == 2:
                output = df.loc[(df['localteam_id'] == team_id) | (df['visitorteam_id'] == team_id)]
            else:
                raise ValueError('home must be an integer between 0 and 2')
        else:
            if home == 0:
                output = df.loc[(df['team_id'] == team_id) & (df['is_home'] == 0)]
                output = output.append(df.loc[(df['op_team_id'] == team_id) & (df['is_home'] == 1)])
            elif home == 1:
                output = df.loc[(df['team_id'] == team_id) & (df['is_home'] == 1)]
                output = output.append(df.loc[(df['op_team_id'] == team_id) & (df['is_home'] == 0)])
            elif home == 2:
                output = df.loc[(df['team_id'] == team_id) | (df['op_team_id'] == team_id)]
            else:
                raise ValueError('home must be an integer between 0 and 2')
            team_name = df.loc[df['team_id'] == team_id]['Team.data.name'].iloc[0]
            name = self.name + ' - {0}'.format(team_name)
        return Fixture(fixture=output, name=name, local_fixture=local_fixture)

    def get_team_scores(self, team_id):
        """
        Convert fixture database from a local_fixture to a non local_fixture for a certain team.
        Args:
            team_id (str): id of the requested team.

        Returns:
            Fixture object containing the games of certain team in a non local_fixture structure.
        """
        local = self.get_team_games(team_id=team_id, home=1)
        local_fixture = local.fixture.copy()
        local_fixture['is_home'] = 1
        local_fixture = local_fixture.rename(columns={'scores.localteam_score': 'score',
                                                      'scores.visitorteam_score': 'op_score',
                                                      'localteam_id': 'team_id',
                                                      'localTeam.data.name': 'Team.data.name',
                                                      'visitorteam_id': 'op_team_id',
                                                      'visitorTeam.data.name': 'op_Team.data.name'})
        visitor = self.get_team_games(team_id=team_id, home=0)
        visitor_fixture = visitor.fixture.copy()
        visitor_fixture['is_home'] = 0
        visitor_fixture = visitor_fixture.rename(columns={'scores.visitorteam_score': 'score',
                                                          'scores.localteam_score': 'op_score',
                                                          'visitorteam_id': 'team_id',
                                                          'visitorTeam.data.name': 'Team.data.name',
                                                          'localteam_id': 'op_team_id',
                                                          'localTeam.data.name': 'op_Team.data.name'})
        output_fixture = local_fixture
        output_fixture = output_fixture.append(visitor_fixture)
        output_fixture = output_fixture.sort_values('time.starting_at.date')
        output = Fixture(name=team_id, fixture=output_fixture, local_fixture=False)
        return output

    def clean_fixture(self):
        """
        Keep only variables selected. Drop matches that have not been played yet. Drop duplicates.
        Drop missing values.

        Returns:
            Fixture object.
        """
        fixture = self.fixture
        season_dict_inv = invert_dictionary(self.get_seasons_dict())

        if not self.local_fixture:
            fixture = fixture.loc[fixture['is_home'] == 1]
        else:
            try:
                condition = (fixture['season_id'] == season_dict_inv[self.get_last_year()]) & \
                            (np.isnan(fixture['team_id']))
                fixture = fixture.loc[~condition]
            except KeyError:
                logger.warning('There was a KeyError omited - clean_fixture method')

        if self.local_fixture:
            vars_to_keep = self.variables_to_keep()
        else:
            vars_to_keep = self.variables_to_keep()
        fixture = fixture[vars_to_keep].drop_duplicates().dropna()
        self.fixture = fixture
        return self

    def drop_x_games_first_last(self, x):
        """
        Drop the first and last x games of a Fixture.
        Args:
            x (int): number of games to be dropped at the start/end.

        Returns:
            Fixture object.
        """
        if x == 0:
            return self
        my_fixture = self.fixture
        my_fixture = my_fixture.sort_values('time.starting_at.date')
        if len(my_fixture) > 15:
            output = my_fixture.iloc[x:-x]
        else:
            output = my_fixture.iloc[x:]
            logger.warning('Team has less than 15 matches: {0}'.format(self.name))
        name = self.name + ' - {0} dropped'.format(x)
        return Fixture(fixture=output, name=name, local_fixture=self.local_fixture)

    def remove_x_games(self, n):
        """
        Drop the first and last n matches in each season.
        Args:
            n (int): number of games to be dropped at the start/end.

        Returns:
            Fixture object.
        """
        output = pd.DataFrame([])
        original_name = self.name
        logger.info('Main Fixture original size: {0}'.format(self.fixture.shape))
        for season in self.get_seasons():
            temp_season = self.subset_season(season)
            teams = temp_season.get_team_ids()
            logger.info('Season {1} original size: {0}'.format(temp_season.fixture.shape, season))
            for team_id in teams:
                temp_team = temp_season.get_team_games(team_id, home=1)
                temp_clean = temp_team.drop_x_games_first_last(n)
                output = output.append(temp_clean.fixture)
        output = output.sort_values('time.starting_at.date')
        name = '{0} - {1} games dropped'.format(original_name, n)
        return Fixture(output, name=name, local_fixture=self.local_fixture)

    def get_score_rolling_mean(self, window_scored, window_conceded):
        """
        Generate the rolling mean of goals scored and goals recieved. For the former,
        the window is window_scored, for the latter, window_conceded. Afterwards, the
        result vector is shifted one position. This way, we do not include the current
        result (score of the match) into the rolling mean computation.
        Args:
            window_scored (int): size of the window for goals scored.
            window_conceded (int): size of the window for goals conceded.

        Returns:
            Fixture object with rolling mean included.
        """
        original_name = self.name
        team_ids = self.get_team_ids()
        output = pd.DataFrame([])
        for team_id in team_ids:
            team_fixture = self.get_team_scores(team_id=team_id).fixture
            team_fixture['roll_score'] = team_fixture['score'].rolling(
                window=window_scored).sum().shift(1)
            team_fixture['roll_op_score'] = team_fixture['op_score'].rolling(
                window=window_conceded).sum().shift(1)
            output = output.append(team_fixture)
        only_main_team = output[['team_id', 'Team.data.name', 'time.starting_at.date',
                                 'roll_op_score']]
        only_main_team = only_main_team.sort_values(['time.starting_at.date',
                                                     'team_id']).reset_index(drop=True)
        only_main_team = only_main_team['roll_op_score']
        output = output.sort_values(['time.starting_at.date', 'op_team_id']).reset_index(drop=True)
        output['roll_op_score'] = only_main_team
        output = output.sort_values('time.starting_at.date')
        name = original_name + ' - roll sum'
        return Fixture(name=name, fixture=output, local_fixture=False)

    def generate_dataset(self, win_scored=10, win_conceded=2, games_removed=0):
        """
        Generate rolling means of goals scored and recieved. Drop the first
        and last games_removed. Drop missing values.
        Args:
            win_scored (int): size of the window for goals scored.
            win_conceded (int): size of the window for goals conceded.
            games_removed (int): number of games to be removed at the start
                and end of the season.

        Returns:
            Fixture object.
        """
        output = self.get_score_rolling_mean(window_scored=win_scored, window_conceded=win_conceded)
        output = output.remove_x_games(games_removed)
        output.fixture = output.fixture.dropna()
        return output

    def get_team_names_and_ids(self):
        if self.local_fixture:
            names_and_ids = set(zip(self.fixture['localteam_id'], self.fixture['localTeam.data.name']))
        else:
            names_and_ids = set(zip(self.fixture['team_id'], self.fixture['Team.data.name']))
        output = {}
        for team_id, team_name in names_and_ids:
            output[team_name] = team_id
        return output

    def add_champion_dummy(self, champions_df):
        origin_name = self.name
        teams_dict = self.get_team_names_and_ids()
        champions_df = champions_df.replace(teams_dict)
        output = pd.DataFrame([])
        for season in self.get_seasons():
            season_dict = self.get_seasons_dict()
            target_year = season_dict[season]
            champions = self.get_champions_in_period(champions_df, target_year, 4)
            temp_fixture = self.subset_season(season).fixture.copy()
            if self.local_fixture:
                temp_fixture['champion'] = temp_fixture['localteam_id'].apply(
                    lambda x: create_dummy_for_champions(x, champions))
            else:
                temp_fixture['champion'] = temp_fixture['team_id'].apply(
                    lambda x: create_dummy_for_champions(x, champions))
            output = output.append(temp_fixture)
        output = output.sort_values('time.starting_at.date')
        name = origin_name + ' - add champion'
        self.fixture = output
        self.name = name
        return self

    def get_seasons_dict(self):
        my_fixture = self.fixture
        my_fixture['year'] = self.get_match_years().astype('int')
        output = {}
        league_name = self.name.split(' ')[0]
        last_season = self.get_last_season()[league_name]
        for season in self.get_seasons():
            if season == last_season:
                continue
            temp = my_fixture.loc[my_fixture['season_id'] == season, :]
            year_mean = math.ceil(np.mean(temp['year']))
            output.update({season: year_mean})
        output.update({last_season: self.get_last_year()})
        return output

    def train_model(self):
        my_fixture = self.fixture[self.variables_in_model()]
        model = smf.glm(formula='score ~ is_home + roll_score + roll_op_score + champion', data=my_fixture,
                        family=sm.families.Poisson()).fit()
        return model

    def exclude_last_x_seasons(self, n):
        seasons_dict = invert_dictionary(self.get_seasons_dict())
        max_year = max(list(seasons_dict.keys()))
        drop_range = range(max_year+1-n, max_year+1)
        my_fixture = self.fixture.copy()
        keep, drop = my_fixture, pd.DataFrame([])
        for year in drop_range:
            target = seasons_dict[year]
            drop = drop.append(my_fixture.loc[my_fixture['season_id'] == target, :])
            keep = keep.loc[keep['season_id'] != target, :]
        drop = drop_single_matches(drop)
        name_keep = self.name + ' - train'
        name_drop = self.name + ' - test'
        keep = Fixture(name=name_keep, fixture=keep, local_fixture=self.local_fixture)
        drop = Fixture(name=name_drop, fixture=drop, local_fixture=self.local_fixture)
        return keep, drop

    def add_predictions(self, predictions):
        my_fixture = self.fixture.copy()
        my_fixture['expected_score'] = predictions
        name = self.name + ' - predicted'
        return Fixture(fixture=my_fixture, name=name, local_fixture=self.local_fixture)

    def convert_to_matches(self):
        my_fixture = self.fixture.copy()
        my_fixture = my_fixture.sort_values(['time.starting_at.date', 'fixture_id'])
        my_fixture = drop_single_matches(my_fixture)
        output = my_fixture.loc[my_fixture['is_home'] == 1].copy().reset_index(drop=True)
        temp = my_fixture.loc[my_fixture['is_home'] == 0].copy().reset_index(drop=True)
        output['op_expected_score'] = temp['expected_score']
        return Fixture(name='match predictions', fixture=output, local_fixture=self.local_fixture)

    def get_matches_prediction(self, model):
        prediction = model.predict(self.fixture)
        my_fixture = self.add_predictions(prediction)
        output = my_fixture.convert_to_matches()
        output = output.get_match_probabilities()
        output.fixture['winner'] = output.fixture.apply(get_winner, axis=1)
        return output

    def get_match_probabilities(self):
        my_fixture = self.fixture.copy()
        local_score = self.fixture['expected_score']
        visitor_score = self.fixture['op_expected_score']
        local_prob_list, visitor_prob_list, tie_prob_list, winner = [], [], [], []
        for i in range(self.fixture.shape[0]):
            match_prob = simulate_match(local_score.iloc[i], visitor_score.iloc[i], max_goals=10)
            local_prob, tie_prob, visitor_prob = sum_triangle_and_diagonal_from_matrix(match_prob)
            local_prob_list.append(local_prob)
            tie_prob_list.append(tie_prob)
            visitor_prob_list.append(visitor_prob)
            if local_prob == max(local_prob, tie_prob, visitor_prob):
                winner.append(self.fixture['Team.data.name'].iloc[i])
            elif visitor_prob == max(local_prob, tie_prob, visitor_prob):
                winner.append(self.fixture['op_Team.data.name'].iloc[i])
            else:
                winner.append('tie')
        my_fixture['local_prob'] = pd.Series(local_prob_list, index=my_fixture.index)
        my_fixture['tie_prob'] = pd.Series(tie_prob_list, index=my_fixture.index)
        my_fixture['visitor_prob'] = pd.Series(visitor_prob_list, index=my_fixture.index)
        my_fixture['expected_winner'] = pd.Series(winner, index=my_fixture.index)
        return Fixture(fixture=my_fixture, name=self.name, local_fixture=self.local_fixture)

    def clean_results(self):
        self.fixture = self.fixture[self.result_variables()]\
            .sort_values(['time.starting_at.date', 'fixture_id'])
        return

    def get_accuracy(self):
        my_fixture = self.fixture.copy()
        total = my_fixture.shape[0]
        good = np.sum([my_fixture['expected_winner'] == my_fixture['winner']])
        return good/total

    def determine_winner(self):
        output = self
        output.fixture['winner_mod'] = output.fixture.apply(get_winner_mod, axis=1)
        return output

    def variables_to_keep(self):
        if self.local_fixture:
            return ['league_id', 'season_id', 'fixture_id', 'localteam_id', 'visitorteam_id',
                    'time.starting_at.date', 'localTeam.data.name', 'visitorTeam.data.name',
                    'scores.localteam_score', 'scores.visitorteam_score']
        else:
            return ['league_id', 'season_id', 'fixture_id', 'team_id', 'op_team_id', 'is_home',
                    'time.starting_at.date', 'Team.data.name', 'op_Team.data.name', 'score', 'op_score']

    def convert_2match_to_1match(self):
        if self.local_fixture:
            raise ValueError('local_fixture is True, fixture already in 1match format')
        else:
            my_df = self.fixture
            my_df = my_df[self.variables_to_keep()].sort_values('fixture_id')
            output = my_df.loc[my_df['is_home'] == 1]
            output = output.rename(columns={'Team.data.name': 'localTeam.data.name',
                                            'op_Team.data.name': 'visitorTeam.data.name',
                                            'team_id': 'localteam_id',
                                            'op_team_id': 'visitorteam_id',
                                            'score': 'scores.localteam_score',
                                            'op_score': 'scores.visitorteam_score'})
            del output['is_home']
            self.local_fixture = True
            self.fixture = output
            return self


    @staticmethod
    def get_last_season():
        last_season = {'82Bundesliga': 8026,
                       '8Premier_League': 6397,
                       '564La_Liga': 8442,
                       '301Ligue_1': 6405,
                       'MLS': 2014,
                       'la_liga': 2014,
                       'premier': 2014,
                       'comebol': 2,
                       'wc2018': 2019}
        return last_season

    @staticmethod
    def get_champions_in_period(champions_df, year, window):
        output = set()
        for i in range(year-window, year):
            output.update(champions_df[str(i)].tolist())
        return output

    @staticmethod
    def variables_in_model():
        output = ['score', 'roll_score', 'roll_op_score', 'champion', 'is_home']
        return output

    @staticmethod
    def result_variables():
        output = ['fixture_id', 'time.starting_at.date', 'Team.data.name', 'op_Team.data.name', 'expected_score',
                  'op_expected_score', 'local_prob', 'tie_prob', 'visitor_prob', 'expected_winner', 'winner',
                  'score', 'op_score', 'is_home']
        return output


def convert_format_time(my_series):
    return my_series.apply(lambda x: x[:10])


def get_results_frequency(fixture):
    seasons_dict = fixture.get_seasons_dict()
    my_league = fixture.determine_winner().fixture[['season_id', 'winner_mod']]
    my_league = pd.get_dummies(my_league, columns=['winner_mod'])
    my_league['total'] = 1
    output = my_league.groupby('season_id').sum().reset_index().replace({'season_id': seasons_dict})
    output['season_id'] = output['season_id'].astype('int')
    output = output.sort_values('season_id')
    return output


def get_league_dictionary():
    league_dict = {'82Bundesliga': 82,
                   '8Premier_League': 8,
                   '564La_Liga': 564,
                   '301Ligue_1': 301}
    return league_dict


def create_dummy_for_champions(team_id, champion_set):
        if team_id in champion_set:
            return 1
        else:
            return 0


def invert_dictionary(my_dict):
    return {v: k for k, v in my_dict.items()}


def simulate_match(home_goals_avg, away_goals_avg, max_goals=10):
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for
                 team_avg in [home_goals_avg, away_goals_avg]]
    return np.outer(np.array(team_pred[0]), np.array(team_pred[1]))


def drop_single_matches(df):
    df = df.sort_values('fixture_id')
    i = 0
    output = pd.DataFrame([])
    fixture_id = df['fixture_id']
    while i < df.shape[0]-1:
        if fixture_id.iloc[i] == fixture_id.iloc[i+1]:
            output = output.append(df.iloc[i])
            output = output.append(df.iloc[i+1])
            i += 2
        else:
            i += 1
    return output


def convert_2match_to_1match(my_df):
    my_df = my_df.sort_values('fixture_id')
    local = my_df.loc[my_df['is_home'] == 1]
    visitor = my_df.loc[my_df['is_home'] == 0]
    visitor = visitor.rename(columns={'expected_goals': 'op_expected_goals'})
    del visitor['is_home']
    output = pd.merge(local, visitor, how='inner', on=['fixture_id'])
    return output



def sum_triangle_and_diagonal_from_matrix(my_matrix):
    upper, lower, diagonal = 0, 0, 0
    for i in range(my_matrix.shape[0]):
        for j in range(my_matrix.shape[1]):
            if i > j:
                lower += my_matrix[i, j]
            if i < j:
                upper += my_matrix[i, j]
            if i == j:
                diagonal += my_matrix[i, j]
    return lower, diagonal, upper


def get_winner(row):
    local = row['score']
    visitor = row['op_score']
    if local > visitor:
        return row['Team.data.name']
    elif local < visitor:
        return row['op_Team.data.name']
    else:
        return 'tie'


def get_winner_mod(row):
    local = row['scores.localteam_score']
    visitor = row['scores.visitorteam_score']
    if local > visitor:
        return 'local'
    elif local < visitor:
        return 'visit'
    else:
        return 'tie'


def predict_model(model, test, ignore_cols):
    """ Runs a simple predictor that will predict if we expect a team to
        win.
    """

    x_test = _splice(_coerce(_clone_and_drop(test, ignore_cols)))
    x_test['intercept'] = 1.0
    predicted = model.predict(x_test)
    result = test.copy()
    result['predicted'] = predicted
    return result


def _clone_and_drop(data, drop_cols):
    """ Returns a copy of a dataframe that doesn't have certain columns. """
    clone = data.copy()
    for col in drop_cols:
        if col in clone.columns:
            del clone[col]
    return clone


def _splice(data):
    """ Splice both rows representing a game into a single one. """
    data = data.copy()
    opp = data.copy()
    opp_cols = ['opp_%s' % (col,) for col in opp.columns]
    opp.columns = opp_cols
    opp = opp.apply(_swap_pairwise)
    del opp['opp_is_home']

    return data.join(opp)


def _swap_pairwise(col):
    """ Swap rows pairwise; i.e. swap row 0 and 1, 2 and 3, etc.  """
    col = pd.np.array(col)
    for index in range(0, len(col), 2):
        val = col[index]
        col[index] = col[index + 1]
        col[index+1] = val
    return col


def _coerce_types(vals):
    """ Makes sure all of the values in a list are floats. """
    return [1.0 * val for val in vals]


def _coerce(data):
    """ Coerces a dataframe to all floats, and standardizes the values. """
    return _standardize(data.apply(_coerce_types))


def non_feature_cols():
    return ['league_id', 'season_id', 'matchid', 'time.starting_at.date', 'teamid', 'op_teamid',
            'op_Team.data.name', 'Team.data.name', 'op_team_name', 'score', 'op_score',
            'op_points', 'round_id', 'referee_id', 'formation', 'op_formation', 'points',
            'time.minute']


def train_model(data, ignore_cols):
    """ Trains a logistic regression model over the data. Columns that
        are passed in ignore_cols are considered metadata and not used
        in the model building.
    """
    # Validate the data
    data = prepare_data(data)
    logger.info('Observations used in the model: {0}'.format(len(data)))
    target_col = 'points'
    (train, test) = split(data)
    train = train.loc[data['points'] != 1]
    (y_train, x_train) = _extract_target(train, target_col)
    x_train2 = _splice(_coerce(_clone_and_drop(x_train, ignore_cols)))

    y_train2 = [int(yval) == 3 for yval in y_train]
    logger.info('Training model')
    model = build_model_logistic(y_train2, x_train2, alpha=8.0)
    return model, test


def prepare_data(data):
    """ Drops all matches where we don't have data for both teams. """
    data = data.copy()
    data = _drop_unbalanced_matches(data)
    _check_data(data)
    return data


L1_ALPHA = 16.0
def build_model_logistic(target, data, acc=0.00000001, alpha=L1_ALPHA):
    """ Trains a logistic regresion model. target is the target.
        data is a dataframe of samples for training. The length of
        target must match the number of rows in data.
    """
    data = data.copy()
    data['intercept'] = 1.0
    logit = sm.Logit(target, data, disp=False)
    return logit.fit_regularized(maxiter=1024, alpha=alpha, acc=acc, disp=False)


def _drop_unbalanced_matches(data):
    """  Because we don't have data on both teams during a match, we
         want to drop any match we don't have info about both teams.
         This can happen if we have fewer than 10 previous games from
         a particular team.
    """
    keep = []
    index = 0
    data = data.dropna()
    while index < len(data) - 1:
        skipped = False
        for col in data:
            if isinstance(col, float) and math.isnan(col):
                keep.append(False)
                index += 1
                skipped = True

        if skipped:
            pass
        elif data.iloc[index]['matchid'] == data.iloc[index + 1]['matchid']:
            keep.append(True)
            keep.append(True)
            index += 2
        else:
            keep.append(False)
            index += 1
    while len(keep) < len(data):
        keep.append(False)
    results = data[keep]
    if len(results) % 2 != 0:
        raise Exception('Unexpected results')
    return results


def _check_data(data):
    """ Walks a dataframe and make sure that all is well. """
    i = 0
    if len(data) % 2 != 0:
        raise Exception('Unexpeted length')
    matches = data['matchid']
    teams = data['teamid']
    op_teams = data['op_teamid']
    while i < len(data) - 1:
        if matches.iloc[i] != matches.iloc[i + 1]:
            raise Exception('Match mismatch: %s vs %s ' % (
                            matches.iloc[i], matches.iloc[i + 1]))
        if teams.iloc[i] != op_teams.iloc[i + 1]:
            raise Exception('Team mismatch: match %s team %s vs %s' % (
                            matches.iloc[i], teams.iloc[i],
                            op_teams.iloc[i + 1]))
        if teams.iloc[i + 1] != op_teams.iloc[i]:
            raise Exception('Team mismatch: match %s team %s vs %s' % (
                            matches.iloc[i], teams.iloc[i + 1],
                            op_teams.iloc[i]))
        i += 2


def split(data, test_proportion=0.2):
    """ Splits a dataframe into a training set and a test set.
        Must be careful because back-to-back rows are expeted to
        represent the same game, so they both must go in the
        test set or both in the training set.
    """

    train_vec = []
    if len(data) % 2 != 0:
        raise Exception('Unexpected data length')
    while len(train_vec) < len(data):
        rnd = random.random()
        train_vec.append(rnd > test_proportion)
        train_vec.append(rnd > test_proportion)

    test_vec = [not val for val in train_vec]
    train = data[train_vec]
    test = data[test_vec]
    if len(train) % 2 != 0:
        raise Exception('Unexpected train length')
    if len(test) % 2 != 0:
        raise Exception('Unexpected test length')
    return (train, test)


def _extract_target(data, target_col):
    """ Removes the target column from a data frame, returns the target
        col and a new data frame minus the target. """
    target = data[target_col]
    train_df = data.copy()
    del train_df[target_col]
    return target, train_df


def _standardize_col(col):
    """ Standardizes a single column (subtracts mean and divides by std
        dev).
    """
    std = np.std(col)
    mean = np.mean(col)
    if abs(std) > 0.001:
        return col.apply(lambda val: (val - mean)/std)
    else:
        return col


def _standardize(data):
    """ Standardizes a dataframe. All fields must be numeric. """
    return data.apply(_standardize_col)


def get_expected_winner(row):
    if row['predicted'] > 0.5:
        return row['Team.data.name']
    else:
        return row['op_Team.data.name']


def get_winners(my_df, tie_prob):
    my_df['winner'] = my_df.apply(get_winner, axis=1)
    output = my_df.loc[my_df['is_home'] == 1].copy().reset_index(drop=True)
    temp = my_df.loc[my_df['is_home'] == 0].copy().reset_index(drop=True)
    output['op_predicted'] = temp['predicted']
    my_df = output.copy()
    my_df['expected_winner'] = my_df.apply(get_expected_winner, axis=1)
    my_df = normalize_predictions(my_df, tie_prob)
    my_df = my_df[result_variables()].rename(columns={'matchid': 'fixture_id'})
    my_df = my_df.sort_values(['time.starting_at.date', 'fixture_id'])
    return my_df


def get_accuracy(my_df, prefix):
    total = my_df.shape[0]
    expected_winner = prefix + 'expected_winner'
    good = np.sum([my_df[expected_winner] == my_df['winner']])
    return good / total


def result_variables():
    output = ['matchid', 'time.starting_at.date', 'Team.data.name', 'op_Team.data.name',
              'expected_winner', 'winner', 'predicted', 'op_predicted', 'tie_predicted']
    return output


def normalize_predictions(my_df, tie_prob):
    my_df = my_df.copy()
    my_df['tie_predicted'] = tie_prob
    my_df['a'] = my_df['predicted'] * (1-my_df['tie_predicted']) / (my_df['predicted'] + my_df['op_predicted'])
    my_df['b'] = my_df['op_predicted'] * (1-my_df['tie_predicted']) / (my_df['predicted'] + my_df['op_predicted'])
    my_df['predicted'] = my_df['a']
    my_df['op_predicted'] = my_df['b']
    del my_df['a']
    del my_df['b']
    return my_df


def squared_error(a, b):
    if len(a) != len(b):
        raise ValueError('Squared error fuction: vectors do not have same lenght')
    dif = a-b
    return dif.dot(dif)


def get_squared_error(my_df):
    my_df = my_df.copy()
    gs_local = squared_error(my_df['gs_expected_score'], my_df['score'])
    gs_visit = squared_error(my_df['gs_op_expected_score'], my_df['op_score'])
    google_local = squared_error(my_df['expected_goals'], my_df['score'])
    google_visit = squared_error(my_df['op_expected_goals'], my_df['op_score'])
    gs_output = (gs_local + gs_visit)/my_df.shape[0]
    google_output = (google_local + google_visit)/my_df.shape[0]
    return gs_output, google_output


