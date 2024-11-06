## This file is for constructing linear regression models for our data.
from sklearn.linear_model import LinearRegression
import numpy as np

class Position_Ensamble:
    def __init__(self,features,target):
        self.features = features
        self.target = target
        self.lreg_D = LinearRegression()
        self.lreg_M = LinearRegression()
        self.lreg_F = LinearRegression()
        self.lreg_G = LinearRegression()

    def fit(self,data):
        X_D = data[data['pos_D'] == True]
        y_D = X_D[self.target]
        X_D = X_D[self.features]
        self.lreg_D.fit(X_D,y_D)

        X_M = data[data['pos_M'] == True]
        y_M = X_M[self.target]
        X_M = X_M[self.features]
        self.lreg_M.fit(X_M,y_M)

        X_F = data[data['pos_F'] == True]
        y_F = X_F[self.target]
        X_F = X_F[self.features]
        self.lreg_F.fit(X_F,y_F)

        X_G = data[data['pos_G'] == True]
        y_G = X_G[self.target]
        X_G = X_G[self.features]
        self.lreg_G.fit(X_G,y_G)

    def predict(self,data):
        X_D = data[data['pos_D'] == True][self.features]
        X_M = data[data['pos_M'] == True][self.features]
        X_F = data[data['pos_F'] == True][self.features]
        X_G = data[data['pos_G'] == True][self.features]
        pred = np.concatenate((self.lreg_D.predict(X_D), self.lreg_M.predict(X_M), self.lreg_F.predict(X_F), self.lreg_G.predict(X_G)))
        return pred