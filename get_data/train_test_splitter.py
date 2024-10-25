import pandas as pd
import sqlite3 as sql

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
    def __init__(self,train_split,seed):
        """A class that gives a fixed train test split of our main data and transforms"""
        self.test = self.get_splits(train_split,seed)
        #self.train, self.test = self.get_splits()
        self.save()

    def get_splits(self,train_split,seed):
        fd = '../data/main_data/main_data_sofa_under.csv'
        
        data = pd.read_csv(fd)
        data['date'] = pd.to_datetime(data['date']).dt.date
        data['date_of_birth_ss'] = pd.to_datetime(data['date_of_birth_ss']).dt.date

        #get our aggregation info
        agg_info = {}

        agg_info['name'] = self.agg_identity
        agg_info['date_of_birth_ss'] = self.agg_identity
        agg_info['position_acronym'] = self.agg_identity
        agg_info['height'] = self.agg_identity
        agg_info['foot'] = self.agg_identity

        for c in columns_count:
            agg_info[c] = 'sum'

        for c in columns_mean:
            agg_info[c] = 'mean'

        agg_data = data.agg(agg_info)
        return agg_data

    def agg_identity(self,column_data):
        return column_data
     

