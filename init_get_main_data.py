import get_data.Scrapers.get_kaggle as gk
import get_data.merge_data.merge_transfer_sofascore as mts
import sqlite3
import pandas as pd
import os

# get all kaggle data
print('Start kaggle data download')
gk.get_transfermarkt_data()
gk.get_sofascore_data()
print('Done with kaggle data\n Start Merge')
mts.merge()

sql_file = 'data/main_data/main_data.db'
con = sqlite3.connect(sql_file)

    
df = pd.read_csv('data/main_data/main_data_sofascore.csv')

table_name = 'player_stats'
df.to_sql(table_name,con,if_exists='replace',index=False)

con.commit()
con.close()