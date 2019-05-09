# Author: Bruno Esposito
# Last modified: 07/11/17

#!/usr/bin/python3
import pandas as pd
import time

import sportmonks
import config

def main():

    LOGGER_LEVEL = 20
    cfg_name = './config/_credentials.cfg'
    OUTPUT_PATH = './data/sportmonks/'
    GET_STATS = False
    league_id_list = [2,5,72,74,78,82,85,208,462,564,570,384,390,8,9,24,12,600,301,304,453,1114,292,654,651,444,573,579]
   
    t0 = time.time()
    pd.set_option('display.float_format', lambda x: '{0:.2f}'.format(x))
    logger = config.config_logger(__name__, level=LOGGER_LEVEL)

    cfg_parser = configparser.ConfigParser()
    cfg_parser.read(cfg_name)
    my_key = str(cfg_parser.get('Sportmonks', 'key'))

    logger.info('Beginning execution')
    sportmonks.init(my_key)

    logger.info('Available leagues:')
    leagues_dict = {}
    for l in sportmonks.leagues():
        print(l['id'], l['name'], l['country_id'])
        leagues_dict[l['id']] = l['name']

    for league_id in league_id_list:
        if GET_STATS:
            league_name = leagues_dict[league_id]
            logger.info('Sending query - Stats')
            logger.info('League selected: {0} - {1}'. format(league_id, league_name))
            league_json = sportmonks.league(league_id,
                include='seasons.fixtures.stats,seasons.fixtures.localTeam,seasons.fixtures.visitorTeam')

            logger.info('Processing package')
            league_df = sportmonks.league_into_dataframe(league_json)
            #print(league_df.head().to_string())
            logger.info('Dimensions of the dataframe: {0}'.format(league_df.shape))

            save_name = '{0}{1}'.format(league_id,league_name.replace(' ', '_'))
            logger.info('Saving CSV: {0}'.format(OUTPUT_PATH + save_name))
            league_df.to_csv(OUTPUT_PATH + save_name + '.csv')

    config.time_taken_display(t0)
    print(' ')


if __name__ == '__main__':
    main()

