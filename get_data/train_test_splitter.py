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

columns_general = ['name', 'position_acronym', 'height', 'date_of_birth_ss',
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
        data = data.rename(columns={'date_of_birth_ss': 'dob'})

        #get our aggregation info
        agg_info = {}

        agg_info['position_acronym'] = 'last'
        agg_info['height'] = 'last'
        agg_info['foot'] = 'last'
        agg_info['date'] = 'last'
        agg_info['market_value_in_eur'] = 'last'

        for c in columns_count:
            agg_info[c] = 'sum'

        for c in columns_mean:
            agg_info[c] = 'mean'

        agg_data = data.groupby(by=['name','dob'],as_index=False).agg(agg_info)
        train,test = train_test_split(agg_data,train_size=train_size,random_state=seed)
        return train,test
    
    def save(self):
        fd = 'data/main_data'
        test_fd = os.path.join(fd,'test')
        train_fd = os.path.join(fd,'train')

        if os.path.exists(test_fd)==False:
            os.mkdir(test_fd)
        if os.path.exists(train_fd)==False:
            os.mkdir(train_fd)

        test_fd = os.path.join(test_fd,'test.csv')
        train_fd = os.path.join(train_fd,'train.csv')

        self.test.to_csv(test_fd,index=False)
        self.train.to_csv(train_fd,index=False)

        
    
     

