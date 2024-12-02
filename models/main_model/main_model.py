
import pandas as pd
import models.main_dataset.ensamble_model as em


def main_model():
    # Train Data
    fd = 'data/main_data/train/train_cutoff.csv'
    train = pd.read_csv(fd)


    model = em.ensamble_model(scale='log')
    model.G_parameters(type ='GBR',alpha=8,max_depth=2,n_estimators=30,min_samples_split=2,min_samples_leaf=6,bootstrap=True)
    model.D_parameters(type ='GBR', alpha=8,max_depth=2,n_estimators=20,min_samples_split=10,min_samples_leaf=6,bootstrap=True)
    model.M_parameters(type='GBR',alpha=8,max_depth=2,n_estimators=20,min_samples_split=10,min_samples_leaf=7,bootstrap=True)
    model.F_parameters(type = 'GBR',alpha=8,max_depth=2,n_estimators=20,min_samples_split=10,min_samples_leaf=6,bootstrap=True)
    model.fit(train)

    return model