# This file holds the baseline models we have, to test our other models agianst.
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,mean_absolute_error,mean_absolute_percentage_error
import random
import os
import json


class general_Regression:
    def __init__(self,data,type = 'LR',features=None,scale=None,**kwargs):
        """General model for regression with our data
        Takes in the data we have directly and it will transform it in the appropriate way
        type: This input is for choosing which regression model to use,
            - LR: Linear Regression
            - LASSO: Lasso linear regression
            - RIDGE: Ridge linear regression
            - ELASTICR: Elastic linear regression
            - KNN: k nearest neighbors
            - DT: Decision tree regressor
            - RFR: Random forest regressor
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
        if type == 'RFR':
            self.max_features = kwargs.get('max_features',1)
        else:
            self.max_features = kwargs.get('max_features',None)
        self.random_state = kwargs.get('random_state',None)
        self.max_leaf_nodes = kwargs.get('max_leaf_nodes',None)
        self.n_estimators = kwargs.get('n_estimators',100)
        self.bootstrap = kwargs.get('bootstrap',True)
        self.learning_rate = kwargs.get('learning_rate',0.9)
        self.subsample = kwargs.get('subsample',1)

        self.scale = scale
        
        self.data = data
        self.type = type
        self.features = features if features else self._get_features()
        self.target = 'adjusted_market_value'
        self.model = self._fit()

    def _get_features(self):
        """The base features for this data.  If you want to use different features, you can inherit this 
        class and input the features you want."""
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
        if self.type == 'ELASTICR':
            return Pipeline([('scaler',StandardScaler()),('elastic',ElasticNet(alpha=self.alpha,random_state=self.random_state))])
        if self.type == 'LASSO':
            return Pipeline([('scaler',StandardScaler()),('lasso',Lasso(alpha=self.alpha,random_state=self.random_state))])
        if self.type == 'RIDGE':
            return Pipeline([('scaler',StandardScaler()),('ridge',Ridge(alpha=self.alpha))])
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
                                         random_state = self.random_state,max_leaf_nodes=self.max_leaf_nodes)

    def _transform_data(self,data):
        data = self._get_age(data)
        data = self._indicator_functions(data)
        data[self.target] = self.scale_target(data[self.target])
        return data

    def _get_age(self,data):
        data['date'] = pd.to_datetime(data['date'])
        data['dob'] = pd.to_datetime(data['dob'])
        data['age'] = np.floor((data['date'] - data['dob']).dt.days/365)
        return data
    

    def _indicator_functions(self,data):
        data = pd.get_dummies(data,columns=['pos','foot'])
        return data
    
    def scale_target(self,x):
        if self.scale == None:
            return x
        if self.scale == 'log':
            return np.log1p(x)
    
    def scale_target_back(self,x):
        if self.scale == None:
            return x
        if self.scale == 'log':
            return np.expm1(x)
    
    def _fit(self):
        t_data = self._transform_data(self.data.copy())
        reg = self._the_model()
        reg.fit(t_data[self.features],t_data[self.target])
        return reg
    
    def predict(self,X):
        """Takes in the dataframe of data, with the features and the target column in it and give
        a prediction on that data."""
        X_data = X.copy()
        X_data = self._transform_data(X_data)
        return self.model.predict(X_data[self.features])
    
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
        """Evalutes the data based on the train and test data"""
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
        """Performs K-fold cross-validation with n_splits"""
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

class hyperparameter_tuning_general:
    def __init__(self,data,n_iter, cv,scale=None, type = None,beta=0):
        """Takes in the data, number of iterations (n_iter) and number of cross validations (cv)
        and randomly looks through the possible values for each of the parameters and performs cross-validation.
        It finds the parameters with the best score"""
        self.scale=scale
        self.beta=beta
        self.data = data
        self.n_iter = n_iter
        self.cv = cv
        self.type=type
        # Here we will put all the possible combinations of parameters we would like to look at
        self.models = ['LR','LASSO','RIDGE','ELASTICR','KNN','DT','RFR','GBR']
        self.parameters = {'LR': {},
                           'LASSO': {'alpha' : np.linspace(0.5,10)},
                           'ELASTICR': {'alpha' : np.linspace(0.5,10)},
                           'RIDGE': {'alpha' : np.linspace(0.5,10)},
                           'KNN'  : {'n_neighbors': [2,4,6,8,10,20,30,40,50,60]},
                           'DT'   : {'max_depth' : [None,2,3,4,5,6,10,20],
                                     'max_features' : [None,0.25,0.5,0.75,1,'sqrt'],
                                     'min_samples_split': [2,5,10,15],
                                     'min_samples_leaf' : [1,2,4,8,10,12]},
                           'RFR'  : {'max_depth' : [None,2,3,4,5,6,10],
                                     'n_estimators': [10,20,30,40,50,60,70,80],
                                     'max_features' : [0.25,0.5,0.75,1,'sqrt'],
                                     'min_samples_split': [2,5,10],
                                     'min_samples_leaf' : [1,2,4,8],
                                     'bootstrap' : [True,False]},
                           'GBR'  : {'max_depth' : [None,2,3,4,5,6,10],
                                     'n_estimators': [10,20,30,40,50,60,70,80],
                                     'min_samples_split': [2,5,10],
                                     'min_samples_leaf' : [1,2,4,5,6,7,8],
                                     'bootstrap' : [True,False]}}

        self.best_model, self.best_params, self.best_score, self.best_RMSE_train,self.best_RMSE_test = self.perform_tuning()

    def perform_tuning(self):
        best_score = np.inf
        best_RMSE_train = None
        best_RMSE_test = None
        best_param = None
        best_model = None
        kf = KFold(n_splits=self.cv,shuffle=True,random_state=42)

        for _ in range(self.n_iter):
            
            if self.type==None:
                model_type = random.choice(self.models)
            else:
                model_type = self.type
            

            param = {key : random.choice(values) for key,values in self.parameters[model_type].items()}

            model=None

            cv_scores = []
            cv_train = []
            cv_test = []

            for train_index, val_index in kf.split(self.data):

                data_train = self.data.iloc[train_index]
                data_val = self.data.iloc[val_index]

                model = general_Regression(data_train,type=model_type,scale=self.scale,**param)
                
                y_pred_train = model.predict(data_train)
                y_pred_test = model.predict(data_val)
                target_train = model.scale_target(data_train[model.target])
                target_test = model.scale_target(data_val[model.target])
                score_train = root_mean_squared_error(target_train,y_pred_train)
                score_test = root_mean_squared_error(target_test,y_pred_test) 
                #score with penatly term
                score = score_test + self.beta*abs(score_train - score_test)
                cv_train.append(score_train)
                cv_test.append(score_test)
                cv_scores.append(score)
            mean_cv_score = np.mean(cv_scores)
            mean_cv_train = np.mean(cv_train)
            mean_cv_test = np.mean(cv_test)

            if mean_cv_score<best_score:
                best_score = mean_cv_score
                best_RMSE_train = mean_cv_train
                best_RMSE_test = mean_cv_test
                best_param= {'model' : model_type, 'param' : param}
                best_model = model
            
        return best_model, best_param, best_score,best_RMSE_train,best_RMSE_test


class G_Pos(general_Regression):
    def __init__(self,data,type='LR',scale = None, **kwargs):
        """This is the model for the goal keepers inherited from general_Regression()
        Takes in the data we have directly and it will transform it in the appropriate way
        type: This input is for choosing which regression model to use,
            - LR: Linear Regression
            - LASSO: Lasso linear regression
            - RIDGE: Ridge linear regression
            - ELASTICR: Elastic linear regression
            - KNN: k nearest neighbors
            - DT: Decision tree regressor
            - RFR: Random forest regressor
            - GBR: Gradient boosted regressor
        **kwargs: this can take any arguments for each of the models above, example for Lasso and Ridge,
                  it can take alpha=0.5, or for Random forest regressor, it can take max_depth=3, ect..."""
        features = ['minutesPlayed', 'totalLongBalls','keyPass', 'totalPass','savedShotsFromInsideTheBox', 'saves',
                            'totalKeeperSweeper', 'goalsPrevented', 'touches','blockedScoringAttempt',
                            'yellow_card', 'red_card', 'rating', 'accuratePass',
                            'accurateLongBalls','accurateKeeperSweeper','age']

        super().__init__(data,type=type,features=features,scale = scale,**kwargs)

class D_Pos(general_Regression):
    
    def __init__(self,data,type='LR',scale = None, **kwargs):
        """This is the model for the defenders inherited from general_Regression()
        Takes in the data we have directly and it will transform it in the appropriate way
        type: This input is for choosing which regression model to use,
            - LR: Linear Regression
            - LASSO: Lasso linear regression
            - RIDGE: Ridge linear regression
            - ELASTICR: Elastic linear regression
            - KNN: k nearest neighbors
            - DT: Decision tree regressor
            - RFR: Random forest regressor
            - GBR: Gradient boosted regressor
        **kwargs: this can take any arguments for each of the models above, example for Lasso and Ridge,
                  it can take alpha=0.5, or for Random forest regressor, it can take max_depth=3, ect..."""
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

        super().__init__(data,type=type,features=features,scale = scale, **kwargs)
    
class M_Pos(general_Regression):

    def __init__(self,data,type='LR',scale=None,**kwargs):
        """This is the model for the midfielders inherited from general_Regression()
        Takes in the data we have directly and it will transform it in the appropriate way
        type: This input is for choosing which regression model to use,
            - LR: Linear Regression
            - LASSO: Lasso linear regression
            - RIDGE: Ridge linear regression
            - ELASTICR: Elastic linear regression
            - KNN: k nearest neighbors
            - DT: Decision tree regressor
            - RFR: Random forest regressor
            - GBR: Gradient boosted regressor
        **kwargs: this can take any arguments for each of the models above, example for Lasso and Ridge,
                  it can take alpha=0.5, or for Random forest regressor, it can take max_depth=3, ect..."""
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

        super().__init__(data,type=type,features=features,scale=scale,**kwargs)
    
class F_Pos(general_Regression):

    def __init__(self,data,type='LR',scale=None,**kwargs):
        """This is the model for the forwards inherited from general_Regression()
        Takes in the data we have directly and it will transform it in the appropriate way
        type: This input is for choosing which regression model to use,
            - LR: Linear Regression
            - LASSO: Lasso linear regression
            - RIDGE: Ridge linear regression
            - ELASTICR: Elastic linear regression
            - KNN: k nearest neighbors
            - DT: Decision tree regressor
            - RFR: Random forest regressor
            - GBR: Gradient boosted regressor
        **kwargs: this can take any arguments for each of the models above, example for Lasso and Ridge,
                  it can take alpha=0.5, or for Random forest regressor, it can take max_depth=3, ect..."""
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

        super().__init__(data,type=type,features=features,scale=scale,**kwargs)

    
class ensamble_model:
    def __init__(self,scale=None):
        """ This is the ensamble model for each of the positions.
        To use this model, you need to input the type of model and parameters for those using:
        G_parameters, D_parameters, M_parameters, and F_parameters.  Then you can fit the data with .fit()
        and predict.  This class also has methods to do k-fold cross-validation based on the inputed models and parameters."""
        self.model_setup = {'G': {'model': G_Pos, 'type': 'LR', 'parameters': {}},
                            'D':{'model': D_Pos, 'type': 'LR', 'parameters': {}},
                            'M': {'model': M_Pos, 'type': 'LR', 'parameters': {}},
                            'F': {'model': F_Pos, 'type': 'LR', 'parameters': {}}}
        self.scale=scale
        self.target = 'adjusted_market_value'

        self.G_model=None
        self.D_model=None
        self.M_model=None
        self.F_model=None

    def G_parameters(self,type, **kwargs):
        """The parameters for the model on goal keepers
        type: This input is for choosing which regression model to use,
            - LR: Linear Regression
            - LASSO: Lasso linear regression
            - RIDGE: Ridge linear regression
            - ELASTICR: Elastic linear regression
            - KNN: k nearest neighbors
            - DT: Decision tree regressor
            - RFR: Random forest regressor
            - GBR: Gradient boosted regressor
        **kwargs: this can take any arguments for each of the models above, example for Lasso and Ridge,
                  it can take alpha=0.5, or for Random forest regressor, it can take max_depth=3, ect..."""
        self.model_setup['G']['type'] = type
        self.model_setup['G']['parameters'] = kwargs
        

    def D_parameters(self,type, **kwargs):
        """The parameters for the model on defenders
        type: This input is for choosing which regression model to use,
            - LR: Linear Regression
            - LASSO: Lasso linear regression
            - RIDGE: Ridge linear regression
            - ELASTICR: Elastic linear regression
            - KNN: k nearest neighbors
            - DT: Decision tree regressor
            - RFR: Random forest regressor
            - GBR: Gradient boosted regressor
        **kwargs: this can take any arguments for each of the models above, example for Lasso and Ridge,
                  it can take alpha=0.5, or for Random forest regressor, it can take max_depth=3, ect..."""
        self.model_setup['D']['type'] = type
        self.model_setup['D']['parameters'] = kwargs

    def M_parameters(self,type, **kwargs):
        """The parameters for the model on midfielders
        type: This input is for choosing which regression model to use,
            - LR: Linear Regression
            - LASSO: Lasso linear regression
            - RIDGE: Ridge linear regression
            - ELASTICR: Elastic linear regression
            - KNN: k nearest neighbors
            - DT: Decision tree regressor
            - RFR: Random forest regressor
            - GBR: Gradient boosted regressor
        **kwargs: this can take any arguments for each of the models above, example for Lasso and Ridge,
                  it can take alpha=0.5, or for Random forest regressor, it can take max_depth=3, ect..."""
        self.model_setup['M']['type'] = type
        self.model_setup['M']['parameters'] = kwargs

    def F_parameters(self,type, **kwargs):
        """The parameters for the model on forwards
        type: This input is for choosing which regression model to use,
            - LR: Linear Regression
            - LASSO: Lasso linear regression
            - RIDGE: Ridge linear regression
            - ELASTICR: Elastic linear regression
            - KNN: k nearest neighbors
            - DT: Decision tree regressor
            - RFR: Random forest regressor
            - GBR: Gradient boosted regressor
        **kwargs: this can take any arguments for each of the models above, example for Lasso and Ridge,
                  it can take alpha=0.5, or for Random forest regressor, it can take max_depth=3, ect..."""
        self.model_setup['F']['type'] = type
        self.model_setup['F']['parameters'] = kwargs

    def scale_target(self,x):
        return self.G_model.scale_target(x)
    
    def scale_target_back(self,x):
        return self.G_model.scale_target_back(x)
    
    def fit(self, data):
        """Fits the input data to the model"""
        
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
        return model(X,type,scale=self.scale,**parameters)
    
    def predict(self,data):
        """Predicts the target for input data"""
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
        """Performs cross-valdiation with the input parameters"""
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

            # transform the input data:
            
            
            model = ensamble_model(scale=self.scale)
            model.G_parameters(self.model_setup['G']['type'],**self.model_setup['G']['parameters'])
            model.D_parameters(self.model_setup['D']['type'],**self.model_setup['D']['parameters'])
            model.M_parameters(self.model_setup['M']['type'],**self.model_setup['M']['parameters'])
            model.F_parameters(self.model_setup['F']['type'],**self.model_setup['F']['parameters'])
            

            model.fit(X_train)

            
            y_train = model.scale_target(X_train[model.target])
            y_val = model.scale_target(X_val[model.target])

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




class hyperparameter_tuning:
    def __init__(self,data,n_iter, cv,scale=None,beta=0):
        """Takes in the data, number of iterations (n_iter) and number of cross validations (cv)
        and randomly looks through the possible values for each of the parameters and performs cross-validation.
        It finds the parameters with the best score"""
        self.scale=scale
        self.data = data
        self.n_iter = n_iter
        self.cv = cv
        self.beta=beta
        # Here we will put all the possible combinations of parameters we would like to look at
        self.models = ['LR','LASSO','RIDGE','ELASTICR','KNN','DT','RFR','GBR']
        self.parameters = {'LR': {},
                           'LASSO': {'alpha' : np.linspace(0.5,10)},
                           'ELASTICR': {'alpha' : np.linspace(0.5,10)},
                           'RIDGE': {'alpha' : np.linspace(0.5,10)},
                           'KNN'  : {'n_neighbors': [2,4,6,8,10,20,30,40,50,60]},
                           'DT'   : {'max_depth' : [None,2,3,4,5,6,10,20],
                                     'max_features' : [None,0.25,0.5,0.75,1,'sqrt'],
                                     'min_samples_split': [2,5,10,15],
                                     'min_samples_leaf' : [1,2,4,8,10,12]},
                           'RFR'  : {'max_depth' : [None,2,3,4,5,6,10],
                                     'n_estimators': [10,20,30,40,50,60,70,80],
                                     'max_features' : [0.25,0.5,0.75,1,'sqrt'],
                                     'min_samples_split': [2,5,10],
                                     'min_samples_leaf' : [1,2,4,8],
                                     'bootstrap' : [True,False]},
                           'GBR'  : {'max_depth' : [None,2,3,4,5,6,10],
                                     'n_estimators': [10,20,30,40,50,60,70,80],
                                     'min_samples_split': [2,5,10],
                                     'min_samples_leaf' : [1,2,4,5,6,7,8],
                                     'bootstrap' : [True,False]}}

        self.best_model, self.best_params, self.best_score = self.perform_tuning()

    def perform_tuning(self):
        best_score = np.inf
        best_param = {'G': None, 'D' : None, 'M': None, 'F': None}
        best_model = None
        kf = KFold(n_splits=self.cv,shuffle=True,random_state=42)

        for _ in range(self.n_iter):

            model_G_type = random.choice(self.models)
            model_D_type = random.choice(self.models)
            model_M_type = random.choice(self.models)
            model_F_type = random.choice(self.models)
            

            param_G = {key : random.choice(values) for key,values in self.parameters[model_G_type].items()}
            param_D = {key : random.choice(values) for key,values in self.parameters[model_D_type].items()}
            param_M = {key : random.choice(values) for key,values in self.parameters[model_M_type].items()}
            param_F = {key : random.choice(values) for key,values in self.parameters[model_F_type].items()}

            model = ensamble_model(scale=self.scale)
            model.G_parameters(type = model_G_type,**param_G)
            model.D_parameters(type = model_D_type,**param_D)
            model.M_parameters(type = model_M_type,**param_M)
            model.F_parameters(type = model_F_type,**param_F)

            cv_scores = []

            for train_index, val_index in kf.split(self.data):

                data_train = self.data.iloc[train_index]
                data_val = self.data.iloc[val_index]

                model.fit(data_train)
                
                y_pred_train = model.predict(data_train)
                y_pred_test = model.predict(data_val)

                target_train = model.scale_target(data_train[model.target])
                target_test = model.scale_target(data_val[model.target])

                score_train = root_mean_squared_error(target_train,y_pred_train)
                score_test = root_mean_squared_error(target_test,y_pred_test)
                score = score_test + self.beta*abs(score_test - score_train)
                cv_scores.append(score)

            mean_cv_score = np.mean(cv_scores)

            if mean_cv_score<best_score:
                best_score = mean_cv_score
                best_param['G'] = {'model' : model_G_type, 'param' : param_G}
                best_param['D'] = {'model' : model_D_type, 'param' : param_D}
                best_param['M'] = {'model' : model_M_type, 'param' : param_M}
                best_param['F'] = {'model' : model_F_type, 'param' : param_F}
                best_model = model
            
        return best_model, best_param, best_score
    

def save_general(fd,desc,params, score, RMSE_train,RMSE_test):
        
        if os.path.exists(fd)==False:
            os.mkdir(fd)

        dir = os.path.join(fd,'general_models.csv')
        data = {}

        if not os.path.exists(dir):
            # Create the file
            df_empty = pd.DataFrame(data)
            df_empty.to_csv(dir)

        # Saved the general score
        data['score'] = [score]
        data['desc'] = [desc]
        # Get type of model and the parameters saved
        data['model'] = params['model']
        for k,v in params['param'].items():
            data[k] = [v]
        # save any extra columns
        data['RMSE_train'] = [RMSE_train]
        data['RMSE_test'] = [RMSE_test]

        df = pd.DataFrame(data)
        old_df = pd.read_csv(dir)
        new_df = pd.concat([old_df,df])
        new_df.to_csv(dir,index=False)



        

                



