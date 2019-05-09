import pandas as pd
import time
import pickle
import json

import code.algorithms.config as config
import code.algorithms.models as models


def main():
    t0 = time.time()
    logger = config.config_logger(__name__, 10)
    stats_path = './data/sportmonks/'
    output_path = './output/goldman_sachs/'
    champions_path = './data/sportmonks/league_winners/'
    open_wc_json = False
    stage = 'C'

    if open_wc_json:
        logger.info('Open WC json')
        with open(stats_path + 'russia2018.json') as f:
            wc_data = json.load(f)

        logger.info('Convert WC json into dataframe')
        wc_list = []
        for team in wc_data['teams']:
            team_name = team['contry']
            for match in team['matches']:
                temp = [match[k] for k in ['city', 'date', 'islocal', 'local', 'score', 'stadium', 'visitant']]
                temp = [team_name] + temp
                wc_list.append(temp)

        wc_df = pd.DataFrame(wc_list,
                             columns=['country', 'city', 'date', 'islocal', 'local', 'raw_score', 'stadium', 'visitant'])
        wc_df['score'] = wc_df['raw_score'].apply(lambda x: str.split(x, '-')[0])
        wc_df['op_score'] = wc_df['raw_score'].apply(lambda x: str.split(x, '-')[1])
        wc_df.to_csv(stats_path + 'wc2018.csv')

    months = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04', 'May': '05', 'Jun': '06',
              'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

    logger.info('Open champions dataframe')
    champions_df = pd.read_csv('{0}wc2018_final.csv'.format(champions_path),
                               header=0, index_col=None)

    logger.info('Open WC dataframe')
    wc_df = pd.read_csv(stats_path + 'wc2018_complete{0}.csv'.format(stage), index_col=0)
    wc_df['date'] = wc_df['date'].apply(lambda x: str.split(x, '-')[0])
    wc_df['date'] = wc_df['date'].apply(lambda x: '{0}-{1}-{2}'.format(x[7:-1],
                                                                       months[x[3:6]],
                                                                       x[:2]))
    wc_df = wc_df.rename(columns={'date': 'time.starting_at.date',
                                  'local': 'Team.data.name',
                                  'visitant': 'op_Team.data.name',
                                  'islocal': 'is_home'})
    wc_df['league_id'] = 'wc'
    wc_df['season_id'] = 2018
    wc_df['fixture_id'] = wc_df.index.values
    wc_df['team_id'] = wc_df['Team.data.name']
    wc_df['op_team_id'] = wc_df['op_Team.data.name']


    my_league = models.Fixture(fixture=wc_df, name='wc2018', local_fixture=False)
    #my_league = my_league.convert_2match_to_1match()
    #my_league = my_league.clean_fixture()
    my_league = my_league.generate_dataset(5, 2)
    my_league.add_champion_dummy(champions_df)


    logger.info('Open trained model')
    with open('{0}gs_model.pickle'.format(output_path), 'rb') as handle:
        model = pickle.load(handle)

    logger.info('Extract predictions')
    my_league.fixture = my_league.fixture.loc[my_league.fixture['city'] == 'Rusia']
    results_gs = my_league.get_matches_prediction(model)
    results_gs.clean_results()
    results_gs = results_gs.fixture
    results_gs.to_csv('{0}wc2018_{1}.csv'.format(output_path, stage), index=None)
    print(results_gs.to_string())

    config.time_taken_display(t0)


if __name__ == '__main__':
    main()