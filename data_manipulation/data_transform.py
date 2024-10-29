import pandas as pd
import os
import numpy as np

class transformer:
    def __init__(self,original_data):
        """Grabs our original train/test data and transforms it inorder
            original_data: this is the original data in our train or test at the end of the project"""

        self.data = original_data.copy()
        #sequence of operations on the data
        self.get_age()
        self.scale_market_value()
        

    def get_age(self):
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['dob'] = pd.to_datetime(self.data['dob'])
        self.data['age'] = np.floor((self.data['date'] - self.data['dob']).dt.days/365)


    def scale_market_value(self):
        self.data['market_value'] = np.log1p(self.data['market_value'])
