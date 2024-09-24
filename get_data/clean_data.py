
import os
import get_data.data_wranglers.csv_and_sql as csql



def transfer_market_to_sql():
    """This takes the csv files from the Kaggle dataset for transfermarkt and makes a 
    .db file ( a file where we can use Sqlite3)"""
    csv_files = ['appearances.csv', 'club_games.csv','clubs.csv', 'competitions.csv','game_events.csv','game_lineups.csv','games.csv','player_valuations.csv','players.csv','transfers.csv']

    dr = os.getcwd()
    dr = os.path.join(dr, 'data/transfermarket')
    print(dr)

    name = 'transfermarket'

    csql.csvs_to_sql(csv_files=csv_files,name=name,path=dr)