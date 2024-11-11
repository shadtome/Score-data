import pandas as pd
import sqlite3 as sql
from sklearn.model_selection import train_test_split
import os

columns = ['name', 'shortName', 'position_acronym', 'height', 'date_of_birth_ss',
       'team', 'date', 'league', 'season', 'minutesPlayed', 'rating',
       'accuratePass', 'totalLongBalls', 'accurateLongBalls', 'keyPass',
       'totalPass', 'totalCross', 'accurateCross', 'goalAssist',
       'savedShotsFromInsideTheBox', 'saves', 'totalKeeperSweeper',
       'accurateKeeperSweeper', 'goalsPrevented', 'touches',
       'possessionLostCtrl', 'dispossessed', 'expectedAssists', 'aerialLost',
       'aerialWon', 'duelLost', 'duelWon', 'challengeLost', 'outfielderBlock',
       'totalContest', 'interceptionWon', 'wonContest', 'totalTackle',
       'totalClearance', 'blockedScoringAttempt', 'hitWoodwork',
       'bigChanceCreated', 'bigChanceMissed', 'shotOffTarget',
       'onTargetScoringAttempt', 'goals', 'expectedGoals', 'wasFouled',
       'fouls', 'totalOffside', 'first_name', 'last_name', 'date_of_birth',
       'sub_position', 'position_full', 'foot', 'height_in_cm', 'transfered',
       'transfered_from', 'transfered_to', 'transfer_fee',
       'market_value_in_eur', 'yellow_card', 'red_card', 'xGChain',
       'xGBuildup']

columns_general = ['name','shortName', 'position_acronym', 'height', 'date_of_birth_ss',
       'team', 'date', 'league', 'season','foot']



columns_count = ['minutesPlayed','totalLongBalls','keyPass',
       'totalPass', 'totalCross','goalAssist',
       'savedShotsFromInsideTheBox', 'saves', 'totalKeeperSweeper',
       'goalsPrevented', 'touches',
       'possessionLostCtrl', 'dispossessed','aerialLost',
       'aerialWon', 'duelLost', 'duelWon', 'challengeLost', 'outfielderBlock',
       'totalContest', 'interceptionWon', 'wonContest', 'totalTackle',
       'totalClearance', 'blockedScoringAttempt', 'hitWoodwork',
       'bigChanceCreated', 'bigChanceMissed', 'shotOffTarget',
       'onTargetScoringAttempt', 'goals','wasFouled',
       'fouls', 'totalOffside','yellow_card', 'red_card']
columns_mean = ['rating','accuratePass','accurateLongBalls','accurateCross',
                'accurateKeeperSweeper','expectedAssists','expectedGoals',
                'xGChain','xGBuildup']



class train_test:
    def __init__(self,train_size,seed):
        """A class that gives a fixed train test split of our main data and transforms"""
        self.train, self.test = self.get_splits(train_size,seed)
        self.save()

    def get_splits(self,train_size,seed):
        fd = 'data/main_data/main_data_sofa_under.csv'
        
        data = pd.read_csv(fd)
        data['date'] = pd.to_datetime(data['date']).dt.date
        data['date_of_birth_ss'] = pd.to_datetime(data['date_of_birth_ss']).dt.date

        # first lets get rid of bad rows
        condition = False
        for gs in columns_count:
            condition = condition | (data[gs]!=0)
        for gs in columns_mean:
            condition = condition | (data[gs]!=0)

        data = data[condition]
        
        agg_six_info = {'date': 'last'}
        final_date = data.groupby(by=['name'],as_index=False).agg(agg_six_info)
        final_date['date'] = pd.to_datetime(final_date['date'])
        final_date = final_date.rename(columns={'date':'final_date'})
        final_date['6_months'] = final_date['final_date'] - pd.Timedelta(days = 180)
        
        temp = data.merge(final_date, on='name',how='left')
        temp_after = temp[temp['date']>=temp['6_months']]
        temp_before = temp[temp['date']<temp['6_months']]



        #get our aggregation info
        agg_info = {}

        agg_info['position_acronym'] = 'last'
        agg_info['height'] = 'last'
        agg_info['foot'] = 'last'
        agg_info['date'] = 'last'
        agg_info['market_value_in_eur'] = 'last'
        agg_info['adjusted_market_value_in_eur'] = 'last'
        agg_info['team'] = 'last'
        agg_info['league'] = 'last'

        for c in columns_count:
            agg_info[c] = 'sum'

        for c in columns_mean:
            agg_info[c] = 'mean'

        # agg before the last 6 months
        agg_data_before = temp_before.groupby(by=['name','date_of_birth_ss'],as_index=False).agg(agg_info)


        # Agg the past 6 months

        agg_data_after = temp_after.groupby(by=['name','date_of_birth_ss'],as_index=False).agg(agg_info)

        combine = agg_data_before.merge(agg_data_after,on = 'name',how='left')

        combine = combine.rename(columns={'date_of_birth_ss_x': 'dob',
                                          'height_x': 'height',
                                          'foot_x': 'foot',
                                          'date_x': 'date',
                                          'team_x': 'team',
                                          'league_x':'league',
                                             'position_acronym_x': 'pos', 
                                             'market_value_in_eur_x': 'market_value_before',
                                             'adjusted_market_value_in_eur_x': 'adjusted_market_value_before',
                                             'market_value_in_eur_y': 'target_market_value',
                                             'adjusted_market_value_in_eur_y': 'target_adjusted_market_value'})
        combine = combine.drop(labels=['date_of_birth_ss_y','height_y','date_y','team_y','league_y',
                                       'position_acronym_y'],axis=1)
        train,test = train_test_split(combine,train_size=train_size,random_state=seed,stratify=combine['pos'])
        return train,test
    
    
    def save(self):
        fd = 'data/main_data'
        test_fd = os.path.join(fd,'test')
        train_fd = os.path.join(fd,'train')

        if os.path.exists(test_fd)==False:
            os.mkdir(test_fd)
        if os.path.exists(train_fd)==False:
            os.mkdir(train_fd)

        test_fd = os.path.join(test_fd,'test_up_to_6_months.csv')
        train_fd = os.path.join(train_fd,'train_up_to_6_months.csv')

        self.test.to_csv(test_fd,index=False)
        self.train.to_csv(train_fd,index=False)