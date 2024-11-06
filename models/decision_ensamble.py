from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class ensamble_XGBR_LR:
    def __init__(self,X,y, max_depth,alpha):
        self.tree = DecisionTreeRegressor(max_depth=max_depth)
        self.tree.fit(X,y)
        self.leafs = np.unique(self.tree.apply(X))
        self.leaf_models = {}
        self.gen_leaf_models(X,y,alpha)

    def gen_leaf_models(self,X,y,alpha):
        for l in self.leafs:
            leaf_nodes = self.tree.apply(X)
            leaf_indices = np.where(leaf_nodes == l)
            X_leaf,y_leaf = X.iloc[leaf_indices],y.iloc[leaf_indices]
        
            lin_model = Ridge(alpha=alpha)
            lin_model.fit(X_leaf,y_leaf)
            self.leaf_models[l] = lin_model
        
    def predict(self,X):
        leaf_ids = self.tree.apply(X)
        preds = np.zeros(shape=(len(X),))

        for i, l in enumerate(leaf_ids):
            preds[i] = self.leaf_models[l].predict(X.iloc[[i]])

        return preds