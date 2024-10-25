import get_data.Scrapers.get_kaggle as gk
import get_data.merge_data.merge_transfer_sofascore as mts
import get_data.merge_data.merge_transfer_understat as mus
import get_data.merge_data.merge_sofa_under as msu
import sqlite3
import pandas as pd
import os

# get all kaggle data
print('Start kaggle data download')
gk.get_transfermarkt_data()
gk.get_understat_lineup_data()
gk.get_sofascore_data()
print('Done with kaggle data\n Start Merges')
mts.merge()
mus.merge()
msu.merge_sofa_under()
print('Done with merging of the data')

sql_file = 'data/main_data/main_data.db'
con = sqlite3.connect(sql_file)

    
df = pd.read_csv('data/main_data/main_data_sofa_under.csv')

table_name = 'player_stats'
df.to_sql(table_name,con,if_exists='replace',index=False)

con.commit()
con.close()