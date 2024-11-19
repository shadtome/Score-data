import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath('')))

print(parent_dir)
sys.path.insert(0,parent_dir)

import original_dataset.ensamble_model as em

class Linear_Regression_6_months(em.general_Regression):

    def __init__(self,data,type=type,**kwargs):
        features = ['minutesPlayed']


        super().__init__(data,type=type,features=features,**kwargs)

    def _transform_data(self, data):
        data = self.players_with_lots_of_playtime(data)
        data = super()._transform_data(data)
        return data

    def players_with_lots_of_playtime(self,data):
        data = data[data['minutesPlayed']>100]
        return data



class G_Pos(Linear_Regression_6_months):
    def __init__(self,ect)
        super().__init__(self.)




class ensamble_model_6_months(em.ensamble_model):
    def 
