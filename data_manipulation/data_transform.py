import pandas as pd
import os

class transformer:
    def __init__(self,original_data):
        """Grabs our original train/test data and transforms it inorder
            original_data: this is the original data in our train or test at the end of the project"""
        self.data = original_data
