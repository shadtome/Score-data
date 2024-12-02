
import get_data.Scrapers.get_kaggle as gk
import get_data.clean_data as clean_data
import get_data.merge_data.merge_transfer_sofascore as mts
import get_data.merge_data.merge_transfer_understat as mus
import get_data.merge_data.merge_sofa_under as msu
import get_data.splitters.train_test_splitter as tts
import sqlite3
import pandas as pd


def get_data_merge_split():
    """This is used to get all of the necessary data needed for this project and do all the data wrangling
    It first gets all the kaggle data, then starts to merge the data with each other.  Finally, it is able to make the train,
    test splits for the data we need."""

    print('Start Kaggle data downloads')
    get_all_kaggle_data()
    print('Done with kaggle data\n Start Merges')
    print('Start Merging of the Data')
    mts.merge()
    mus.merge()
    msu.merge_sofa_under()
    print('Done with merging of the data')
    print('Start making database')

    sql_file = 'data/main_data/main_data.db'
    con = sqlite3.connect(sql_file)

        
    df = pd.read_csv('data/main_data/main_data_sofa_under.csv')

    table_name = 'player_stats'
    df.to_sql(table_name,con,if_exists='replace',index=False)

    con.commit()
    con.close()
    print('Create train test splits')
    tts.train_test(train_size=0.8,seed=42)
    tts.train_test(train_size=0.8,seed=42,cutoff=1000)


def get_all_kaggle_data():
    """Grabs all the Kaggle data"""

    gk.get_transfermarkt_data()
    gk.get_understat_lineup_data()
    gk.get_sofascore_data()
    
    #Do any cleaning needed, specifically convert the transfermarket/understat data into a sql database files
    clean_data.transfer_market_to_sql()
    clean_data.understats_lineup_to_sql()







    