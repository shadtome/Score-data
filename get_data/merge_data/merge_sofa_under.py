import pandas as pd
import os
from tqdm.auto import tqdm
import sqlite3 as sql
import numpy as np

CPI = {'year': [2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],
       'cpi': [98.24,98.83,100.49,102.12,103.64,105.44,106.72,112.67,123.9,130.88]}
CPI_df = pd.DataFrame(CPI)

class merge_sofa_under:
    """This is used to merge the data from transfermerkt x understat with transfermarket x sofascore
    to fill in the missing data that sofascore does not have, specifically for xG,xA."""

    def __init__(self):
        self.understat_df = self.get_under()
        self.sofascore_df = self.get_sofa()
        self.merged = self.combine()
        self.change_MV_Inflation()
        self.save()
        print("Done with Merging Transfermarkt x sofascore and Transfermarkt x understat")

    def get_sofa(self):
        """Get the transfermarkt x sofascore data"""
        fd = 'data/main_data/main_data_sofascore.csv'
        df = pd.read_csv(fd)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['name'] = df['name'].astype(str)
        return df
    
    def get_under(self):
        """Get the transfermarkt x understat data"""
        fd = 'data/main_data/main_data_understat.csv'
        df = pd.read_csv(fd)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['name'] = df['name'].astype(str)
        df = df[['name','date','xG','xA','yellow_card','red_card','xGChain','xGBuildup']]
        return df
    
    def combine(self):
        """Combines the data together"""
        merged_df = pd.merge(self.sofascore_df,self.understat_df,on=['name','date'],how='left',suffixes=('_pg','_us'))
        merged_df['xG'] = merged_df['xG'].fillna(0)
        merged_df['xA'] = merged_df['xA'].fillna(0)
        merged_df['yellow_card'] = merged_df['yellow_card'].fillna(0)
        merged_df['red_card'] = merged_df['red_card'].fillna(0)
        merged_df['xGChain'] = merged_df['xGChain'].fillna(0)
        merged_df['xGBuildup'] = merged_df['xGBuildup'].fillna(0)
        merged_df['expectedGoals'] = np.where(merged_df['expectedGoals']==0,merged_df['xG'],merged_df['expectedGoals'])
        merged_df['expectedAssists'] = np.where(merged_df['expectedAssists'] == 0, merged_df['xA'],merged_df['expectedAssists'])
        merged_df = merged_df.drop(columns = ['xG','xA'])
        return merged_df
    
    def change_MV_Inflation(self):
        """Change the market values based on inflation"""
        self.merged['date'] = pd.to_datetime(self.merged['date'])
        self.merged['year'] = self.merged['date'].dt.year
        self.merged = pd.merge(self.merged,CPI_df,on='year')
        base_cpi = CPI_df[CPI_df['year'] == 2024]['cpi'].values[0]
        self.merged['adjusted_market_value_in_eur'] = (self.merged['market_value_in_eur'] * (base_cpi/self.merged['cpi'])).astype(int)
        self.merged = self.merged.drop(labels = ['year','cpi'],axis=1)

    def save(self):
        save_path = os.getcwd()
        save_path = os.path.join(save_path,'data/main_data')
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        save_path = os.path.join(save_path,'main_data_sofa_under.csv')
        self.merged.to_csv(save_path,index=False)