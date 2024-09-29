
import os
import get_data.data_wranglers.csv_and_sql as csql



def transfer_market_to_sql():
    """This takes the csv files from the Kaggle dataset for transfermarkt and makes a 
    .db file ( a file where we can use Sqlite3)"""
    csv_files = ['appearances.csv', 'club_games.csv','clubs.csv', 'competitions.csv','game_events.csv','game_lineups.csv','games.csv','player_valuations.csv','players.csv','transfers.csv']

    dr = os.getcwd()
    dr = os.path.join(dr, 'data/transfermarket')
    

    name = 'transfermarket'

    csql.csvs_to_sql(csv_files=csv_files,name=name,path=dr)


def understats_to_sql():
    """This takes the various csv files from the understats website that we scraped
    and combines it into a .db file for easy sql querys"""

    csv_files = ['player_stats.csv','game_stats.csv']

    dr = os.getcwd()
    dr = os.path.join(dr,'data/understat')

    name = 'understat'

    csql.csvs_to_sql(csv_files=csv_files,name=name,path=dr)