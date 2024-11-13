# This file holds the baseline models we have, to test our other models agianst.
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LinearRegression,LogisticRegression, QuantileRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,mean_absolute_error,mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score,confusion_matrix

class quantile_Regression:
    """This model designed to predict the quantile class of players.  This will help with the regression problem
    later when trying to predict the market_value for players based on their quantile class"""

    def __init__(self,data,quantile=0.5):
        self.q = quantile
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
        self.model = self.fit()

    def the_model(self):
        model = QuantileRegressor(quantile=self.q)

        return model
    

    def transform_data(self,data):
        data = self.get_age(data)
        data = self.indicator_functions(data)
        return data
    
    def get_age(self,data):
        data['date'] = pd.to_datetime(data['date'])
        data['dob'] = pd.to_datetime(data['dob'])
        data['age'] = np.floor((data['date'] - data['dob']).dt.days/365)
        return data
    

    def indicator_functions(self,data):
        data = pd.get_dummies(data,columns=['pos','foot'])
        return data
    
    
    
    def fit(self):
        t_data = self.transform_data(self.data.copy())
        
        reg = self.the_model()
        reg.fit(t_data[self.features],t_data[self.target])
        return reg
    
    def predict(self,X):
        X_data = X.copy()
        X_data = self.transform_data(X_data)
        return self.model.predict(X_data[self.features])
    
    def predict_player(self,X,player):
        X_data = X.copy()
        X_data = self.transform_data(X_data)
        try:
            X_data = X_data[X_data['name'] == player]
            if X_data.empty:
                raise ValueError(f'Player "{player}" not found in the dataframe')
            return self.model.predict(X_data[self.features])
        except ValueError as e:
            print(e)

    def evaluate(self,test_data):
        t_test = self.transform_data(test_data.copy())
        t_train = self.transform_data(self.data.copy())
        train_pred = self.model.predict(t_train[self.features])
        test_pred = self.model.predict(t_test[self.features])

        MSE_train = mean_squared_error(train_pred,t_train[self.target])
        RMSE_train = root_mean_squared_error(train_pred, t_train[self.target])
        R2_train = r2_score(t_train[self.target], train_pred)
        MAE_train = mean_absolute_error(train_pred,t_train[self.target])
        MAPE_train = mean_absolute_percentage_error(train_pred, t_train[self.target])

        MSE_test = mean_squared_error(test_pred,t_test[self.target])
        RMSE_test = root_mean_squared_error(test_pred, t_test[self.target])
        R2_test = r2_score( t_test[self.target],test_pred)
        MAE_test = mean_absolute_error(test_pred,t_test[self.target])
        MAPE_test = mean_absolute_percentage_error(test_pred, t_test[self.target])

        print(f'MSE for train: {MSE_train}') 
        print(f'MSE for test:  {MSE_test}\n')

        print(f'RMSE for train: {RMSE_train}') 
        print(f'RMSE for test: {RMSE_test}\n')

        print(f'R^2 for train: {R2_train}') 
        print(f'R^2 for test: {R2_test}\n')

        print(f'MAE for train: {MAE_train}') 
        print(f'MAE for test: {MAE_test}\n')  

        print(f'MAPE for train: {MAPE_train}') 
        print(f'MAPE for test: {MAPE_test}\n')    

    def perform_CV(self,n_splits=10):
        cv = KFold(n_splits=n_splits,shuffle=True,random_state=42,)
        t_data = self.transform_data(self.data.copy())
        X = t_data[self.features]
        y = t_data[self.target]

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

        for train_index, val_index in cv.split(t_data):
            X_train, X_val = X.iloc[train_index] , X.iloc[val_index]
            y_train, y_val = y.iloc[train_index] , y.iloc[val_index]

            model = self.the_model()

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


