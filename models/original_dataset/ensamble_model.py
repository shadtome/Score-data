# This file holds the baseline models we have, to test our other models agianst.
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,mean_absolute_error,mean_absolute_percentage_error



class general_Regression:
    def __init__(self,data,type = 'LR',features=None,**kwargs):
        """General model for regression with our data
        Takes in the data we have directly and it will transform it in the appropriate way
        type: This input is for choosing which regression model to use,
            - LR: Linear Regression
            - LASSO: Lasso linear regression
            - RIDGE: Ridge linear regression
            - KNN: k nearest neighbors
            - DT: Decision tree regressor
            - RDR: Random forest regressor
            - GBR: Gradient boosted regressor
        **kwargs: this can take any arguments for each of the models above, example for Lasso and Ridge,
                  it can take alpha=0.5, or for Random forest regressor, it can take max_depth=3, ect..."""
        self.n_neighbors = kwargs.get('n_neighbors', 5)
        self.alpha = kwargs.get('alpha',1)
        if type == 'GBR':
            self.max_depth = kwargs.get('max_depth',3)
        else:
            self.max_depth = kwargs.get('max_depth',None)
        self.min_samples_leaf = kwargs.get('min_samples_leaf',1)
        self.min_samples_split = kwargs.get('min_samples_split',2)
        if type == 'RDR':
            self.max_features = kwargs.get('max_features',1)
        else:
            self.max_features = kwargs.get('max_features',None)
        self.random_state = kwargs.get('random_state',None)
        self.max_leaf_nodes = kwargs.get('max_leaf_nodes',None)
        self.n_estimators = kwargs.get('n_estimators',100)
        self.bootstrap = kwargs.get('bootstrap',True)
        self.learning_rate = kwargs.get('learning_rate',0.9)
        self.subsample = kwargs.get('subsample',1)
        
        self.data = data
        self.type = type
        self.features = features if features else self._get_features()
        self.target = 'adjusted_market_value'
        self.model = self._fit()

    def _get_features(self):
        features = ['minutesPlayed', 'totalLongBalls','keyPass', 'totalPass',
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
        return features

    def _the_model(self):
        
        if self.type == 'LR':
            return LinearRegression()
        if self.type == 'LASSO':
            return Lasso(alpha=self.alpha,random_state=self.random_state)
        if self.type == 'RIDGE':
            return Ridge(alpha=self.alpha)
        if self.type == 'KNN':
            return Pipeline([('scaler',StandardScaler()),('knn',KNeighborsRegressor(n_neighbors=self.n_neighbors))])
        if self.type == 'DT':
            return DecisionTreeRegressor(max_depth=self.max_depth,min_samples_split = self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf,max_features=self.max_features,
                                         random_state = self.random_state,max_leaf_nodes=self.max_leaf_nodes)
        if self.type == 'RFR':
            return RandomForestRegressor(n_estimators=self.n_estimators,max_depth=self.max_depth,
                                         min_samples_split = self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf,max_features=self.max_features,
                                         random_state = self.random_state,max_leaf_nodes=self.max_leaf_nodes,
                                         bootstrap=self.bootstrap)
        if self.type == 'GBR':
            return GradientBoostingRegressor(subsample = self.subsample,learning_rate = self.learning_rate,n_estimators=self.n_estimators,max_depth=self.max_depth,
                                         min_samples_split = self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf,max_features=self.max_features,
                                         random_state = self.random_state,max_leaf_nodes=self.max_leaf_nodes,
                                         bootstrap=self.bootstrap)

    def _transform_data(self,data):
        data = self._get_age(data)
        data = self._indicator_functions(data)
        return data

    def _get_age(self,data):
        data['date'] = pd.to_datetime(data['date'])
        data['dob'] = pd.to_datetime(data['dob'])
        data['age'] = np.floor((data['date'] - data['dob']).dt.days/365)
        return data
    

    def _indicator_functions(self,data):
        data = pd.get_dummies(data,columns=['pos','foot'])
        return data
    
    def _scale_target(self,data):
        return data
    
    def scale_target_back(self,x):
        return x
    
    def _fit(self):
        t_data = self._transform_data(self.data.copy())
        reg = self._the_model()
        reg.fit(t_data[self.features],t_data[self.target])
        return reg
    
    def predict(self,X):
        X_data = X.copy()
        X_data = self._transform_data(X_data)
        return self.scale_target_back(self.model.predict(X_data[self.features]))
    
    def predict_player(self,X,player):
        X_data = X.copy()
        X_data = self._transform_data(X_data)
        try:
            X_data = X_data[X_data['name'] == player]
            if X_data.empty:
                raise ValueError(f'Player "{player}" not found in the dataframe')
            return f'{float(self.scale_target_back(self.model.predict(X_data[self.features]))):f}'
        except ValueError as e:
            print(e)

    def evaluate(self,test_data):
        t_test = self._transform_data(test_data.copy())
        t_train = self._transform_data(self.data.copy())
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
        t_data = self._transform_data(self.data.copy())
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

            model = self._the_model()

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


class G_Pos(general_Regression):
    def __init__(self,data,type='LR',**kwargs):
        features = ['minutesPlayed', 'totalLongBalls','keyPass', 'totalPass','savedShotsFromInsideTheBox', 'saves',
                            'totalKeeperSweeper', 'goalsPrevented', 'touches','blockedScoringAttempt',
                            'yellow_card', 'red_card', 'rating', 'accuratePass',
                            'accurateLongBalls','accurateKeeperSweeper','age']

        super().__init__(data,type=type,features=features,**kwargs)

class D_Pos(general_Regression):
    def __init__(self,data,type='LR',**kwargs):
        features = ['minutesPlayed', 'totalLongBalls','keyPass', 'totalPass',
                            'totalCross', 'goalAssist', 'goalsPrevented', 'touches', 'possessionLostCtrl',
                            'dispossessed', 'aerialLost', 'aerialWon', 'duelLost', 'duelWon',
                            'challengeLost', 'outfielderBlock', 'totalContest', 'interceptionWon',
                            'wonContest', 'totalTackle', 'totalClearance', 'blockedScoringAttempt',
                            'hitWoodwork', 'bigChanceCreated', 'bigChanceMissed', 'shotOffTarget',
                            'onTargetScoringAttempt', 'goals', 'wasFouled', 'fouls', 'totalOffside',
                            'yellow_card', 'red_card', 'rating', 'accuratePass',
                            'accurateLongBalls', 'accurateCross',
                            'expectedAssists', 'expectedGoals', 'xGChain', 'xGBuildup', 'age']

        super().__init__(data,type=type,features=features,**kwargs)
    
class M_Pos(general_Regression):
    def __init__(self,data,type='LR',**kwargs):
        features = ['minutesPlayed', 'totalLongBalls','keyPass', 'totalPass',
                            'totalCross', 'goalAssist', 'savedShotsFromInsideTheBox',
                             'goalsPrevented', 'touches', 'possessionLostCtrl',
                            'dispossessed', 'aerialLost', 'aerialWon', 'duelLost', 'duelWon',
                            'challengeLost', 'outfielderBlock', 'totalContest', 'interceptionWon',
                            'wonContest', 'totalTackle', 'totalClearance', 'blockedScoringAttempt',
                            'hitWoodwork', 'bigChanceCreated', 'bigChanceMissed', 'shotOffTarget',
                            'onTargetScoringAttempt', 'goals', 'wasFouled', 'fouls', 'totalOffside',
                            'yellow_card', 'red_card', 'rating', 'accuratePass',
                            'accurateLongBalls', 'accurateCross',
                            'expectedAssists', 'expectedGoals', 'xGChain', 'xGBuildup', 'age']

        super().__init__(data,type=type,features=features,**kwargs)
    
class F_Pos(general_Regression):
    def __init__(self,data,type='LR',**kwargs):
        features = ['minutesPlayed', 'totalLongBalls','keyPass', 'totalPass',
                            'totalCross', 'goalAssist',
                             'goalsPrevented', 'touches', 'possessionLostCtrl',
                            'dispossessed', 'aerialLost', 'aerialWon', 'duelLost', 'duelWon',
                            'challengeLost', 'outfielderBlock', 'totalContest', 'interceptionWon',
                            'wonContest', 'totalTackle', 'totalClearance', 'blockedScoringAttempt',
                            'hitWoodwork', 'bigChanceCreated', 'bigChanceMissed', 'shotOffTarget',
                            'onTargetScoringAttempt', 'goals', 'wasFouled', 'fouls', 'totalOffside',
                            'yellow_card', 'red_card', 'rating', 'accuratePass',
                            'accurateLongBalls', 'accurateCross',
                            'expectedAssists', 'expectedGoals', 'xGChain', 'xGBuildup', 'age']

        super().__init__(data,type=type,features=features,**kwargs)

    
class ensamble_model:
    def __init__(self):
        
        # here put in our specifications for each model for each position to make these as best as possible.
        # Default is Linear regression
        self.model_setup = {'G': {'model': G_Pos, 'type': 'LR', 'parameters': {}},
                            'D':{'model': D_Pos, 'type': 'LR', 'parameters': {}},
                            'M': {'model': M_Pos, 'type': 'RFR', 'parameters': {'max_depth': 4}},
                            'F': {'model': F_Pos, 'type': 'RFR', 'parameters': {'max_depth':4}}}
        
        self.target = 'adjusted_market_value'

        self.G_model=None
        self.D_model=None
        self.M_model=None
        self.F_model=None

    def fit(self, data):
        self.G_model = self.get_model(data,'G')
        self.D_model = self.get_model(data,'D')
        self.M_model = self.get_model(data,'M')
        self.F_model = self.get_model(data,'F')


    def get_model(self, data, pos: str):
        X = data.copy()
        X = X.loc[X['pos'] == pos]
        model = self.model_setup[pos]['model']
        type = self.model_setup[pos]['type']
        parameters = self.model_setup[pos]['parameters']
        return model(X,type,**parameters)
    
    def predict(self,data):
        X = data.copy()
        X_G = X.loc[X['pos'] == 'G']
        X_D = X.loc[X['pos'] == 'D']
        X_M = X.loc[X['pos'] == 'M']
        X_F = X.loc[X['pos'] == 'F']

        if X_G.empty== False:
            X.loc[X['pos'] == 'G','prediction'] = self.G_model.predict(X_G)
        if X_D.empty == False:
            X.loc[X['pos'] == 'D','prediction'] = self.D_model.predict(X_D)
        if X_M.empty == False:
            X.loc[X['pos'] == 'M','prediction'] = self.M_model.predict(X_M)
        if X_F.empty == False:
            X.loc[X['pos'] == 'F','prediction'] = self.F_model.predict(X_F)
        
        return X[['prediction']]
    

    def perform_CV(self,data,n_splits=10):
        cv = KFold(n_splits=n_splits,shuffle=True,random_state=42,)
        data_c = data.copy()
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

        for train_index, val_index in cv.split(data_c):
            X_train, X_val = data_c.iloc[train_index] , data_c.iloc[val_index]

            
            model = ensamble_model()
            y_train = X_train[model.target]
            y_val = X_val[model.target]

            model.fit(X_train)

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

