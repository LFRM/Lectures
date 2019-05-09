# Author: Bruno Esposito
# Forked from: https://github.com/tvl/python-sportmonks
# Last modified: 07/11/17

#!/usr/bin/python3
import requests
import json
import pandas as pd
from pandas.io.json import json_normalize

import config


logger = config.config_logger(__name__,10)
api_token = ''
api_url = 'https://soccer.sportmonks.com/api/v2.0/'

def init(token):
    """Before you are able to make requests to our API it's required that you have created an Account and API token. You can create API tokens via the settings page which is available when you are logged in. API token will only be shown to you when you create them. Please make sure that you store your token safely.

    To authorize request to our API you must add a parameter to your request called api_token. The value of this parameter is the actual token you received when you created it. SportMonks will measure the usage of all the tokens you have generated and will make them visible via your dashboard."""

    global api_token
    api_token = token


def get(endpoint, include=None, page=None, paginated=True):
    payload = {'api_token': api_token }
    if include:
        payload['include'] = include
    if page:
        paginated = True
        payload['page'] = page
    r = requests.get(api_url + endpoint, params=payload)
    parts = json.loads(r.text)
    data = parts.get('data')
    meta = parts.get('meta')
    if not data:
        return None
    pagination = meta.get('pagination')
    if pagination:
        pages = int(pagination['total_pages'])
    else:
        pages = 1
    if (not paginated) and (pages > 1):
        for i in range(2, pages+1):
            payload['page'] = i
            r = requests.get(api_url + endpoint, params=payload)
            next_parts = json.loads(r.text)
            next_data = next_parts.get('data')
            if next_data:
                data.extend(next_data)
    return data


def continents():
    """With this endpoint you are able to retrieve a list of continents."""
    return get('continents')


def continent(id):
    """With this endpoint you are able to retrieve details a specific continent."""
    return get('continents/{}'.format(id))


def countries():
    """With this endpoint you are able to retrieve a list of countries."""
    return get('countries')


def country(id):
    """With this endpoint you are able to retrieve details a specific country."""
    return get('countries/{}'.format(id))


def leagues():
    """With this endpoint you are able to retrieve a list of leagues."""
    return get('leagues')


def league(id, include=None, page=None, paginated=True):
    """With this endpoint you are able to retrieve details a specific league."""
    return get('leagues/{}'.format(id),include, page, paginated)


def seasons():
    """With this endpoint you are able to retrieve a list of seasons."""
    return get('seasons')


def season(id, last=None, include=None, page=None, paginated=True):
    """With this endpoint you are able to retrieve a specific season."""
    return get('seasons/{}'.format(id), include, page, paginated)


def fixtures(first, last=None, include=None, page=None, paginated=True):
    """With this endpoint you are able to retrieve all fixtures between 2 dates or retrieve all fixtures for a given date."""
    if last is None:
        return get('fixtures/date/{}/'.format(first), include, page, paginated)
    else:
        return get('fixtures/between/{}/{}/'.format(first, last), include, page, paginated)


def fixture(id, include=None, page=None, paginated=True):
    """With this endpoint you are able to retrieve a fixture by it's id. """
    return get('fixtures/{}'.format(id), include, page, paginated)


def todayscores():
    """With this endpoint you are able to retrieve all fixtures that are played on the current day."""
    return get('livescores')


def livescores(include=None, page=None, paginated=True):
    """With this endpoint you are able to retrieve all fixtures for are currently beeing played. This response will also contain games that are starting within 45 minutes and that are ended less then 30 minutes ago."""
    return get('livescores/now', include, page, paginated)


def standings(season):
    """With this endpoint you are able to retrieve the standings for a given season."""
    return get('standings/season/{}'.format(season))


def venue(id):
    """With this endpoint you can get more information about a venue."""
    return get('venues/{}'.format(id))


def teams(season):
    """It might be interesting to know what teams have played a game in a partisucal season. with this endpoint you are able to retrieve a list of teams that have at least played 1 game in it."""
    return get('teams/season/{}'.format(season))


def team(id):
    """With this endpoint you are able to retrieve basic team information. """
    return get('teams/{}'.format(id))


def rounds(season):
    """With this endpoint you are able to retrieve all rounds for a given season (if applicable)."""
    return get('rounds/season/{}'.format(season))


def round(id):
    """With this endpoint you are able to retrieve a round by a given id."""
    return get('rounds/{}'.format(id))


def id_variables():
    return ['league_id', 'season_id', 'fixture_id', 'localteam_id', 'visitorteam_id', 'team_id',
            'time.starting_at.date', 'localTeam.data.name', 'visitorTeam.data.name', 'scores.localteam_score',
            'scores.visitorteam_score']


def order_fixture_dataframe(fixt_df):
    """
    Put the id variables of fixt_df at the beginning of the dataframe
    :param fixt_df: a dataframe with the variables:
        - league_id
        - season_id
        - match_id
        - localteam_id
        - visitorteam_id
    :return: the ordered df
    """
    output_df = fixt_df.copy()
    first_vars = id_variables()
    cols = list(output_df.columns.values)
    new_cols = [i for i in cols if i not in first_vars]
    new_cols.sort()
    new_cols = first_vars + new_cols
    output_df = output_df[new_cols]
    return output_df


def fixture_into_dataframe(fixt_json, verbose=False):
    """
    Convert a fixture json into a pandas dataframe.
    :param fixt_json: Fixture json obtained through the fixture function.
    :param verbose: True if you want to display log information.
    :return: - a dataframe containing the fixture json information
             - a bool if the game has stats info
    """
    if verbose:
        logger.info('Converting fixture json: id {0}'.format(fixt_json['id']))

    game_df = json_normalize(fixt_json)
    game_df.drop('stats.data', axis=1, inplace=True)
    game_df.rename(columns={'id':'fixture_id'}, inplace=True)
    stats_json = fixt_json['stats']['data']
    if not stats_json:
        has_stats = False
        game_df['stats_available'] = 0
        return game_df, has_stats
    teams_df = pd.DataFrame({})
    has_stats = True
    for team_json in stats_json:
        temp = json_normalize(team_json)
        teams_df = teams_df.append(temp).reset_index(drop=True)
    teams_df = pd.merge(teams_df, game_df, on='fixture_id', how='left')
    teams_df['stats_available'] = 1
    teams_df = order_fixture_dataframe(teams_df)
    return teams_df, has_stats


def season_into_dataframe(season_json):
    """
    Generate a dataframe with all the games in the season.
    There will be two observations per game. One per team.
    :param season_json: Season json obtained through the season function.
    :return: a dataframe containing the dataframe json information
    """
    season_df = pd.DataFrame({})
    matches = season_json['fixtures']['data']
    logger.info('Converting season json: league_id {2} - name {0} - id {1}'.format(
        season_json['name'], season_json['id'], season_json['league_id']))
    logger.info('Matches found: {0} - {1}'.format(len(matches), season_json['name']))
    matches_with_stats = 0
    for match in matches:
        (temp, has_stats) = fixture_into_dataframe(match, verbose=False)
        season_df = season_df.append(temp).reset_index(drop=True)
        matches_with_stats += has_stats
    logger.info('Matches with stats info: {0} - {1}'.format(matches_with_stats, season_json['name']))
    #if len(matches) == len(season_df):
    #    logger.warning('No stats data: name {0} - id {1}'.format(
    #        season_json['name'], season_json['id']))
        #return pd.DataFrame({})
    #season_df.fillna(0, inplace=True)
    #season_df = order_fixture_dataframe(season_df)
    return season_df


def league_into_dataframe(league_json):
    """
    Generate a dataframe with all the games in the season.
    There will be two observations per game. One per team.
    :param league_json: League json obtained through the season function.
    :return: a dataframe containing the dataframe json information
    """
    league_df = pd.DataFrame({})
    seasons = league_json['seasons']['data']
    logger.info('Converting league json: league id {0} - name {1}'.format(league_json['id'], league_json['name']))
    logger.info('Seasons found: {0}'.format(len(seasons)))
    for season in seasons:
        temp = season_into_dataframe(season)
        league_df = league_df.append(temp).reset_index(drop=True)
    league_df = order_fixture_dataframe(league_df)
    logger.info('Total matches with stats data: {0}'.format(league_df['stats_available'].sum()/2))
    return league_df