import pandas as pd
import time
import pickle

import code.algorithms.config as config
import code.algorithms.models as models


def main():
    t0 = time.time()
    logger = config.config_logger(__name__, 10)
    fixtures_path = './data/sportmonks/with_data/'
    champions_path = './data/sportmonks/league_winners/'
    output_path = './output/goldman_sachs/'
    stats_path = './data/sportmonks/'

    logger.info('Begin execution')
    target_league = '8Premier_League'
    stats_file = 'sportmonks_final'
    league_id = models.get_league_dictionary()[target_league]

    # Leagues:
    # 82Bundesliga
    # 8Premier_League
    # 564La_Liga
    # 301Ligue_1

    logger.info('Open matches dataframe: {0}'.format(target_league))
    raw_df = pd.read_csv(fixtures_path + target_league + '.csv', index_col=0)
    logger.info('Open champions dataframe: {0}'.format(target_league))
    champions_df = pd.read_csv(champions_path + stats_file + '.csv', index_col=None)
    logger.info('Open stats dataframe: {0}'.format(stats_file))
    stats_df = pd.read_csv(stats_path + stats_file + '.csv', index_col=0)

    logger.info('Get descriptive stats')
    my_league = models.Fixture(raw_df, target_league)
    my_league.clean_fixture()
    stats_season = models.get_results_frequency(my_league)
    tie_prob = stats_season['winner_mod_tie'].sum() / stats_season['total'].sum()
    stats_season.to_csv('./output/descriptives/{0}.csv'.format(target_league), index=None)

    logger.info('Google algorithm start')
    google_df = stats_df.rename(columns={'fixture_id': 'matchid',
                                         'team_id': 'teamid',
                                         'op_team_id': 'op_teamid',
                                         'team_name': 'Team.data.name',
                                         'op_team_name': 'op_Team.data.name',
                                         'goals': 'score',
                                         'op_goals': 'op_score'})

    objective = google_df.loc[google_df['league_id'] == league_id]
    google_df = google_df.loc[google_df['league_id'] != league_id]

    (google_model, google_test) = models.train_model(google_df, models.non_feature_cols())
    results_google = models.predict_model(google_model, objective, models.non_feature_cols())

    results_google = models.get_winners(results_google, tie_prob=tie_prob)
    logger.info('GOOGLE - Matches predicted: {0}'.format(results_google.shape[0]))
    accuracy = models.get_accuracy(results_google, '')
    logger.info('GOOGLE - Accuracy obtained: {0}'.format(accuracy))
    results_google.to_csv(output_path + 'google_predictions.csv', index=None)

    logger.info('GS algorithm start')
    logger.info('Preprocess data')
    stats_df = stats_df.rename(columns={'goals': 'score',
                                        'op_goals': 'op_score',
                                        'team_name': 'Team.data.name',
                                        'op_team_name': 'op_Team.data.name'})

    my_league = models.Fixture(fixture=stats_df, name=target_league, local_fixture=False)
    my_league = my_league.convert_2match_to_1match()
    my_league.clean_fixture()
    my_league = my_league.generate_dataset()
    my_league.add_champion_dummy(champions_df)

    #train, test = my_league.exclude_last_x_seasons(2)
    temp_fixture = my_league.fixture
    objective_fixt = temp_fixture.loc[temp_fixture['league_id'] == league_id]
    objective = models.Fixture(fixture=objective_fixt, name='objective', local_fixture=False)
    train_fixt = temp_fixture.loc[temp_fixture['league_id'] != league_id]
    train = models.Fixture(fixture=train_fixt, name='train', local_fixture=False)

    logger.info('Train model: poisson regression - obs: {0}'.format(train.fixture.shape[0]))
    gs_model = train.train_model()

    logger.info('Save trained model')
    with open('{0}gs_model.pickle'.format(output_path), 'wb') as handle:
        pickle.dump(gs_model, handle)

    logger.info('Extract predictions')
    results_gs = objective.get_matches_prediction(gs_model)
    logger.info('GS - Matches predicted: {0}'.format(results_gs.fixture.shape[0]))

    logger.info('Save results')
    results_gs.clean_results()
    results_gs.fixture.to_csv(output_path + 'gs_predictions.csv', index=None)

    accuracy = results_gs.get_accuracy()
    logger.info('GS - Accuracy obtained: {0}'.format(accuracy))

    # Merge Google and GS results:
    logger.info('Merge Google and GS results')
    results_google = results_google.rename(columns={'predicted': 'google_predicted',
                                                    'op_predicted': 'google_op_predicted',
                                                    'tie_predicted': 'google_tie_predicted',
                                                    'expected_winner': 'google_expected_winner'})
    results_gs = results_gs.fixture[['fixture_id', 'local_prob', 'tie_prob', 'visitor_prob', 'expected_winner']]
    results_gs = results_gs.rename(columns={'local_prob': 'gs_local_prob',
                                            'tie_prob': 'gs_tie_prob',
                                            'visitor_prob': 'gs_visitor_prob',
                                            'expected_winner': 'gs_expected_winner'})

    results = pd.merge(results_google, results_gs, how='inner', on=['fixture_id'])
    logger.info('Matches predicted: {0}'.format(results.shape[0]))
    accuracy_gs = models.get_accuracy(results, 'gs_')
    accuracy_google = models.get_accuracy(results, 'google_')
    logger.info('Final GS acc: {0}'.format(accuracy_gs))
    logger.info('Final Google acc: {0}'.format(accuracy_google))
    results.to_csv('{0}{1}_predictions.csv'.format(output_path, target_league), index=None)
    print(results.head().to_string())



    config.time_taken_display(t0)


if __name__ == '__main__':
    main()
