# This file holds the baseline models we have, to test our other models agianst.
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

    def the_model(self):
        return LinearRegression()

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

        reg = self.the_model()
        reg.fit(self.data[self.features],self.data[self.target])
        return reg
    
    def predict(self,X):
        X_data = X.copy()
        X_data = self.transform_data(X_data)
        return self.model.predict(X_data[self.features])

    def perform_CV(self,n_splits=10):
        cv = KFold(n_splits=n_splits,shuffle=True,random_state=42,)
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
    

class Position_Linear_Regression(Simple_Linear_Regression):
    def __init__(self,data, pos):
        self.pos = pos
        super().__init__(data)

    def transform_data(self, data):
        data = super().transform_data(data)
        data = self.get_pos(data)
        return data
    
    def get_pos(self,data):
        data = data[data[f'pos_{self.pos}'] == True]
        return data
    

class PCA_Linear_Regression(Simple_Linear_Regression):
    def __init__(self,data,n_components=None):
        self.n_components = n_components
        super().__init__(data)
    def the_model(self):
        pipeline = Pipeline([('scale',StandardScaler()),('pca',PCA(self.n_components)),('lreg',LinearRegression())])
        return pipeline


    


class Tree_Reg:
    def __init__(self,data,parameters = {'n_estimators': 100,
                                            'max_depth': None,
                                            'max_features': None,
                                            'min_samples_leaf': 1,
                                            'min_samples_split': 2,
                                            'bootstrap': True},
                       type='DTR', model=None):
        # First we need to transform our data to conform with data
        self.parameters = parameters
        self.type = type
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
        self.model = None
        if model == None:
            self.model = self.fit()
        else:
            self.model = model




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
        t_data = self.transform_data(self.data.copy())
        X=t_data[self.features]
        y=t_data[self.target]
        reg=None
        if self.type == 'DTR':
            reg = DecisionTreeRegressor(max_depth=self.parameters['max_depth'],max_features=self.parameters['max_features'],
                                        min_samples_split=self.parameters['min_samples_split'],
                                        min_samples_leaf=self.parameters['min_samples_leaf'])
        if self.type == 'RFR':
            reg = RandomForestRegressor(max_depth=self.parameters['max_depth'],n_estimators=self.parameters['n_estimators'],
                                        max_features=self.parameters['max_features'],
                                        min_samples_split=self.parameters['min_samples_split'],
                                          min_samples_leaf=self.parameters['min_samples_leaf'],
                                        bootstrap=self.parameters['bootstrap'])
        if self.type == 'GBR':
            reg = GradientBoostingRegressor(max_depth=self.parameters['max_depth'],n_estimators=self.parameters['n_estimators'],
                                            max_features=self.parameters['max_features'],
                                            min_samples_split=self.parameters['min_samples_split'],
                                            min_samples_leaf=self.parameters['min_samples_leaf'])
        reg.fit(X,y)
        return reg
        
    
    def predict(self,X):
        X_data = X.copy()
        X_data = self.transform_data(X_data)
        return self.model.predict(X_data[self.features])
    
    def evaluate(self,test_data):

        train_pred = self.predict(self.data)

        test = test_data.copy()
        pred = self.predict(test)

        MSE_train = mean_squared_error(train_pred,self.data[self.target])
        RMSE_train = root_mean_squared_error(train_pred, self.data[self.target])
        R2_train = r2_score(train_pred, self.data[self.target])
        MAE_train = mean_absolute_error(train_pred,self.data[self.target])
        MAPE_train = mean_absolute_percentage_error(train_pred, self.data[self.target])

        MSE_test = mean_squared_error(pred,test[self.target])
        RMSE_test = root_mean_squared_error(pred, test[self.target])
        R2_test = r2_score(pred, test[self.target])
        MAE_test = mean_absolute_error(pred,test[self.target])
        MAPE_test = mean_absolute_percentage_error(pred, test[self.target])

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

        for train_index, val_index in cv.split(self.data):
            X_train, X_val = X.iloc[train_index] , X.iloc[val_index]
            y_train, y_val = y.iloc[train_index] , y.iloc[val_index]

            model=None
            if self.type == 'DTR':
                model = DecisionTreeRegressor(max_depth=self.parameters['max_depth'],max_features=self.parameters['max_features'],
                                        min_samples_split=self.parameters['min_samples_split'],
                                        min_samples_leaf=self.parameters['min_samples_leaf'])
            if self.type == 'RFR':
                model = RandomForestRegressor(max_depth=self.parameters['max_depth'],n_estimators=self.parameters['n_estimators'],
                                            max_features=self.parameters['max_features'],
                                            min_samples_split=self.parameters['min_samples_split'],
                                            min_samples_leaf=self.parameters['min_samples_leaf'],
                                            bootstrap=self.parameters['bootstrap'])
            if self.type == 'GBR':
                model = GradientBoostingRegressor(max_depth=self.parameters['max_depth'],n_estimators=self.parameters['n_estimators'],
                                                max_features=self.parameters['max_features'],
                                                min_samples_split=self.parameters['min_samples_split'],
                                                min_samples_leaf=self.parameters['min_samples_leaf'])

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


    def hyperparameter_Search(self):
        t_data = self.transform_data(self.data.copy())
        X = t_data[self.features]
        y = t_data[self.target]
        random_grid = {'n_estimators': [50,100,200, 400,800],
                       'max_depth': [None,1,2,3,4,5,10,30,60,90],
                       'max_features': [None,1,'log2','sqrt'],
                       'min_samples_leaf': [1,2,4],
                       'min_samples_split': [2,5,10],
                       'bootstrap': [True,False]}
        

        if self.type == 'DTR':
            random_grid.pop('n_estimators')
            random_grid.pop('bootstrap')
            reg = DecisionTreeRegressor()
            reg_random = RandomizedSearchCV(estimator = reg, param_distributions = random_grid,
                                            n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
            reg_random.fit(X,y)
            return Tree_Reg(self.data,reg_random.best_params_,type = self.type,model=reg_random.best_estimator_)
        if self.type == 'RFR':
            reg = RandomForestRegressor()
            reg_random = RandomizedSearchCV(estimator = reg, param_distributions = random_grid,
                                            n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
            reg_random.fit(X,y)
            return Tree_Reg(self.data,reg_random.best_params_,type = self.type,model=reg_random.best_estimator_)
        if self.type == 'GBR':
            random_grid.pop('bootstrap')
            reg = GradientBoostingRegressor()
            reg_random = RandomizedSearchCV(estimator = reg, param_distributions = random_grid,
                                            n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
            reg_random.fit(X,y)
            return Tree_Reg(self.data,reg_random.best_params_,type = self.type,model=reg_random.best_estimator_)