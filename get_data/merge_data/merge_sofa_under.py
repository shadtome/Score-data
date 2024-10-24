import pandas as pd
import os
from tqdm.auto import tqdm
import sqlite3 as sql
import numpy as np


class merge_sofa_under:
    """This is used to merge the data from transfermerkt x understat with transfermarket x sofascore
    to fill in the missing data that sofascore does not have, specifically for xG,xA."""

    def __init__(self):
        self.understat_df = self.get_under()
        self.sofascore_df = self.get_sofa()
        self.merged = self.combine()
        self.save()
        print("Done with Merging Transfermarkt x sofascore and Transfermarkt x understat")

    def get_sofa(self):
        fd = 'data/main_data/main_data_sofascore.csv'
        df = pd.read_csv(fd)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['name'] = df['name'].astype(str)
        return df
    
    def get_under(self):
        fd = 'data/main_data/main_data_understat.csv'
        df = pd.read_csv(fd)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['name'] = df['name'].astype(str)
        df = df[['name','date','xG','xA','yellow_card','red_card','xGChain','xGBuildup']]
        return df
    
    def combine(self):
        merged_df = pd.merge(self.sofascore_df,self.understat_df,on=['name','date'],how='left',suffixes=('_pg','_us'))
        merged_df['expectedGoals'] = np.where(merged_df['expectedGoals']==0,merged_df['xG'],merged_df['expectedGoals'])
        merged_df['expectedAssists'] = np.where(merged_df['expectedGoals'] == 0, merged_df['xA'],merged_df['expectedAssists'])
        merged_df = merged_df.drop(columns = ['xG','xA'])
        return merged_df
    

    def save(self):
        save_path = os.getcwd()
        save_path = os.path.join(save_path,'data/main_data')
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        save_path = os.path.join(save_path,'main_data_sofa_under.csv')
        self.merged.to_csv(save_path,index=False)