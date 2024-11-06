# This file holds the baseline models we have, to test our other models agianst.
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,mean_absolute_error,mean_absolute_percentage_error

class Simple_Linear_Regression:
    """This is just simple linear regression with out any feature engineering and all of the features"""
    def __init__(self,data):
        # First we need to transform our data to conform with data
        self.data = data
        self.features = ['minutesPlayed', 'totalLongBalls', 'keyPass', 'totalPass',
                            'totalCross', 'goalAssist', 'savedShotsFromInsideTheBox', 'saves',
                            'totalKeeperSweeper', 'goalsPrevented', 'touches', 'possessionLostCtrl',
                            'dispossessed', 'aerialLost', 'aerialWon', 'duelLost', 'duelWon',
                            'challengeLost', 'outfielderBlock', 'totalContest', 'interceptionWon',
                            'wonContest', 'totalTackle', 'totalClearance', 'blockedScoringAttempt',
                            'hitWoodwork', 'bigChanceCreated', 'bigChanceMissed', 'shotOffTarget',
                            'onTargetScoringAttempt', 'goals', 'wasFouled', 'fouls', 'totalOffside',
                            'yellow_card', 'red_card', 'rating', 'accuratePass',
                            'accurateLongBalls', 'accurateCross', 'accurateKeeperSweeper',
                            'expectedAssists', 'expectedGoals', 'xGChain', 'xGBuildup', 'age',
                            'pos_D', 'pos_F', 'pos_G', 'pos_M', 'foot_both', 'foot_left',
                            'foot_right']
        self.target = 'adjusted_market_value'
        self.data = self.transform_data(self.data)
        self.model = self.fit()

    def transform_data(self,data):
        data = self.get_age(data)
        data = self.drop_date_name(data)
        data = self.indicator_functions(data)
        return data

    def get_age(self,data):
        data['date'] = pd.to_datetime(data['date'])
        data['dob'] = pd.to_datetime(data['dob'])
        data['age'] = np.floor((data['date'] - data['dob']).dt.days/365)
        return data
    
    def drop_date_name(self,data):
        data = data.drop(['name','dob','date'],axis=1)
        return data

    def indicator_functions(self,data):
        data = pd.get_dummies(data,columns=['pos','foot'])
        return data

    def fit(self):

        reg = LinearRegression()
        reg.fit(self.data[self.features],self.data[self.target])
        return reg
    
    def predict(self,X):
        X_data = X.copy()
        X_data = self.transform_data(X_data)
        return self.model.predict(X_data[self.features])

    def perform_CV(self):
        cv = KFold(n_splits=10,shuffle=True,random_state=42,)
        X = self.data[self.features]
        y = self.data[self.target]

        train_mses = []
        train_rmses = []
        train_R2 = []
        train_maes = []
        train_mapes = []


        test_mses = []
        test_rmses = []
        test_R2 = []
        test_maes = []
        test_mapes = []

        for train_index, val_index in cv.split(self.data):
            X_train, X_val = X.iloc[train_index] , X.iloc[val_index]
            y_train, y_val = y.iloc[train_index] , y.iloc[val_index]

            model = LinearRegression()

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_val)

            train_mses.append(mean_squared_error(y_train,y_train_pred))
            test_mses.append(mean_squared_error(y_val,y_test_pred))

            train_rmses.append(root_mean_squared_error(y_train,y_train_pred))
            test_rmses.append(root_mean_squared_error(y_val,y_test_pred))

            train_R2.append(r2_score(y_train,y_train_pred))
            test_R2.append(r2_score(y_val,y_test_pred))

            train_maes.append(mean_absolute_error(y_train,y_train_pred))
            test_maes.append(mean_absolute_error(y_val,y_test_pred))

            train_mapes.append(mean_absolute_percentage_error(y_train,y_train_pred))
            test_mapes.append(mean_absolute_percentage_error(y_val,y_test_pred))

        print(f'MSE for train: mean: {np.mean(train_mses)} std: {np.std(train_mses)}') 
        print(f'MSE for test:  mean: {np.mean(test_mses)}  std: {np.std(test_mses)}\n')

        print(f'RMSE for train: mean: {np.mean(train_rmses)} std: {np.std(train_rmses)}') 
        print(f'RMSE for test: mean: {np.mean(test_rmses)} std: {np.std(test_rmses)}\n')

        print(f'R^2 for train: mean: {np.mean(train_R2)} std: {np.std(train_R2)}') 
        print(f'R^2 for test: mean: {np.mean(test_R2)} std: {np.std(test_R2)}\n')

        print(f'MAE for train: mean: {np.mean(train_maes)} std: {np.std(train_maes)}') 
        print(f'MAE for test: mean: {np.mean(test_maes)} std: {np.std(test_maes)}\n')  

        print(f'MAPE for train: mean: {np.mean(train_mapes)} std: {np.std(train_mapes)}') 
        print(f'MAPE for test: mean: {np.mean(test_mapes)} std: {np.std(test_mapes)}\n')



class Poly_Linear_Regression(Simple_Linear_Regression):
    def __init__(self, data, quad_f,cubic_f):
        self.quad_f = quad_f
        self.cubic_f = cubic_f
        super().__init__(data)
        

    def transform_data(self, data):
        data = super().transform_data(data)
        data = self.quad_features(data)
        data = self.cubic_features(data)
        return data

    def quad_features(self,data):
        for f in self.quad_f:
            data[f+'2'] = np.pow(data[f],2)
            self.features += [f+'2']
        return data
    
    def cubic_features(self,data):
        for f in self.quad_f:
            data[f+'3'] = np.pow(data[f],3)
            self.features += [f+'3']
        return data
    

    


class Decision_Tree_Reg:
    def __init__(self,data, max_depth = None):
        # First we need to transform our data to conform with data
        self.max_depth = max_depth
        self.data = data
        self.features = ['minutesPlayed', 'totalLongBalls', 'keyPass', 'totalPass',
                            'totalCross', 'goalAssist', 'savedShotsFromInsideTheBox', 'saves',
                            'totalKeeperSweeper', 'goalsPrevented', 'touches', 'possessionLostCtrl',
                            'dispossessed', 'aerialLost', 'aerialWon', 'duelLost', 'duelWon',
                            'challengeLost', 'outfielderBlock', 'totalContest', 'interceptionWon',
                            'wonContest', 'totalTackle', 'totalClearance', 'blockedScoringAttempt',
                            'hitWoodwork', 'bigChanceCreated', 'bigChanceMissed', 'shotOffTarget',
                            'onTargetScoringAttempt', 'goals', 'wasFouled', 'fouls', 'totalOffside',
                            'yellow_card', 'red_card', 'rating', 'accuratePass',
                            'accurateLongBalls', 'accurateCross', 'accurateKeeperSweeper',
                            'expectedAssists', 'expectedGoals', 'xGChain', 'xGBuildup', 'age',
                            'pos_D', 'pos_F', 'pos_G', 'pos_M', 'foot_both', 'foot_left',
                            'foot_right']
        self.target = 'adjusted_market_value'
        self.data = self.transform_data(self.data)
        self.model = self.fit()

    def transform_data(self,data):
        data = self.get_age(data)
        data = self.drop_date_name(data)
        data = self.indicator_functions(data)
        return data

    def get_age(self,data):
        data['date'] = pd.to_datetime(data['date'])
        data['dob'] = pd.to_datetime(data['dob'])
        data['age'] = np.floor((data['date'] - data['dob']).dt.days/365)
        return data
    
    def drop_date_name(self,data):
        data = data.drop(['name','dob','date'],axis=1)
        return data

    def indicator_functions(self,data):
        data = pd.get_dummies(data,columns=['pos','foot'])
        return data

    def fit(self):

        reg = DecisionTreeRegressor(max_depth=self.max_depth)
        reg.fit(self.data[self.features],self.data[self.target])
        return reg
    
    def predict(self,X):
        X_data = X.copy()
        X_data = self.transform_data(X_data)
        return self.model.predict(X_data[self.features])

    def perform_CV(self):
        cv = KFold(n_splits=10,shuffle=True,random_state=42,)
        X = self.data[self.features]
        y = self.data[self.target]

        train_mses = []
        train_rmses = []
        train_R2 = []
        train_maes = []
        train_mapes = []


        test_mses = []
        test_rmses = []
        test_R2 = []
        test_maes = []
        test_mapes = []

        for train_index, val_index in cv.split(self.data):
            X_train, X_val = X.iloc[train_index] , X.iloc[val_index]
            y_train, y_val = y.iloc[train_index] , y.iloc[val_index]

            model = DecisionTreeRegressor(max_depth=self.max_depth)

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_val)

            train_mses.append(mean_squared_error(y_train,y_train_pred))
            test_mses.append(mean_squared_error(y_val,y_test_pred))

            train_rmses.append(root_mean_squared_error(y_train,y_train_pred))
            test_rmses.append(root_mean_squared_error(y_val,y_test_pred))

            train_R2.append(r2_score(y_train,y_train_pred))
            test_R2.append(r2_score(y_val,y_test_pred))

            train_maes.append(mean_absolute_error(y_train,y_train_pred))
            test_maes.append(mean_absolute_error(y_val,y_test_pred))

            train_mapes.append(mean_absolute_percentage_error(y_train,y_train_pred))
            test_mapes.append(mean_absolute_percentage_error(y_val,y_test_pred))

        print(f'MSE for train: mean: {np.mean(train_mses)} std: {np.std(train_mses)}') 
        print(f'MSE for test:  mean: {np.mean(test_mses)}  std: {np.std(test_mses)}\n')

        print(f'RMSE for train: mean: {np.mean(train_rmses)} std: {np.std(train_rmses)}') 
        print(f'RMSE for test: mean: {np.mean(test_rmses)} std: {np.std(test_rmses)}\n')

        print(f'R^2 for train: mean: {np.mean(train_R2)} std: {np.std(train_R2)}') 
        print(f'R^2 for test: mean: {np.mean(test_R2)} std: {np.std(test_R2)}\n')

        print(f'MAE for train: mean: {np.mean(train_maes)} std: {np.std(train_maes)}') 
        print(f'MAE for test: mean: {np.mean(test_maes)} std: {np.std(test_maes)}\n')  

        print(f'MAPE for train: mean: {np.mean(train_mapes)} std: {np.std(train_mapes)}') 
        print(f'MAPE for test: mean: {np.mean(test_mapes)} std: {np.std(test_mapes)}\n')
