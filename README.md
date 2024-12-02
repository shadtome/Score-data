# SCORE: Soccer Club Optimization of Recruitment Expenses
### Predicting market values for players in major European soccer leagues
### A 2024 Erdös Institute Project

## Authors
- Cody Tipton &nbsp;<a href="https://www.linkedin.com/in/cody-tipton-21075417b/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="height: 1em; width:auto;"/></a> &nbsp; <a href="https://github.com/shadtome"> <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="height: 1em; width: auto;"/></a>
- Rafael Magaldi &nbsp;<a href="https://www.linkedin.com/in/rafaelmagaldi/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="height: 1em; width:auto;"/></a> &nbsp; <a href="https://github.com/rmmagaldi"> <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="height: 1em; width: auto;"/></a>

## Summary
We use in-game statistics to **predict the market value** of soccer players. This is useful information for soccer clubs when deciding how much to bid when trying to sign a player.

## Background
The soccer transfer market is highly competitive and volatile, with player valuations frequently fluctuating. Clubs have access to two transfer windows per season, when thousands of transfers take place worldwide, moving billions of dollars. Clubs often compete to sign the best talent within their budget, so having accurate player valuations is vital when determining how much to spend on a new player. These valuations allow clubs to make informed financial decisions and ensure sustainable investments that pay off in the field.

## Dataset
Our data is sourced from the following websites:

- Transfermarkt: https://www.transfermarkt.us/
    - This has the transfer fees, market values, and general stats about players worldwide.
- Sofascore: https://www.sofascore.com/
    - This contains all the player and team stats in every professional game, including a sofascore rating measuring how well a player did in a match.
- Understat: https://understat.com/
    - This contains some general stats about each player and game in the English Premier League, La Liga, Bundesliga, Ligue 1, Serie A, and Russian Premier League.

For the Transfermarkt data, we were able to use a dataset available on Kaggle (https://www.kaggle.com/datasets/davidcariboo/player-scores) that is kept up-to-date, so we had current information on the market values of all players.
As for the player stats, we performed web scraping to gather them from Understat (using Beautiful Soup) and Sofascore (using Selenium and Sofascore's API). We chose to gather data from the top 10 strongest soccer leagues in Europe (https://theanalyst.com/2024/10/strongest-leagues-world-football-opta-power-rankings), across as many seasons as they had stats for (ranging from 5 to 11 seasons depending on the league).

Finally, we combined these data sources, leaving us with over 10k players and all their in-game stats as well as their personal information like name, height and date of birth.

## Stakeholders

Our stakeholders are European soccer clubs across various leagues, including both large, established clubs and smaller, emerging teams. They have a vested interest in optimizing the buying and selling of players to enhance both sporting performance and financial sustainability. These clubs seek data-driven insights to avoid overpaying for talent, identify undervalued players, and make informed transfer decisions. Ultimately, their goal is to maximize returns on player investments, maintain competitive squads, and achieve long-term financial health within the volatile football transfer market.

Another potential niche for the models developed in this project are fantasy soccer players, who seek accurate player valuations to optimize their fantasy teams.

## Key Performance Indicators
We want to create a predictive model, so having good accuracy is vital. With that goal in mind, we measured the following quantities for all models tested:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- R²

We also want to make sure the model has generalization power when presented to new data, so we measure all of these on both the training and testing sets, and compare them. In general, we found that the RMSE gives us a good idea of how well a model is performing and how well it generalizes, so when discussing the results we will focus on this metric.

## Feature Engineering and Target Preprocessing
Given the player stats on each match, the approach we decided to follow was to aggregate the statistics up to the current date or whenever the player stopped playing (using sums or averages, depending on the feature). We also chose to use the last known market value for each player as the target, and adjusted it for inflation when it was not current (for instance, for players who stopped playing before the current season). We noted the target had high skewness and kurtosis, which was causing some negative market value predictions with linear regression, so to make the distribution closer to a normal one, we decided to apply ln(1+x) to the market values. This also has the benefit of helping reduce outliers, as now the range of the target is between the values of 10 and 20. We then split the dataset into training (80%) and testing (20%) sets, ensuring diversity for player profiles and making sure there are proportional amounts of players for each position in each set.

![image](https://github.com/user-attachments/assets/008ccae8-3276-4d96-8d48-3111530db12e)

## EDA
For our exploratory data analysis, we checked the influence of each feature on the market value, noting that players that play on different positions have different sets of features that matter most when determining their market values.

![image](https://github.com/user-attachments/assets/0a687690-bda4-43bf-a779-92b0a391561a)

![image](https://github.com/user-attachments/assets/12ada42c-113d-4de2-94e3-ee3e6ec7a07c)


## Model Testing and Evaluation
### Baseline Model:

For this model, we used all the features and all the players in the training set.

Simple Linear Regression:

- Train RMSE: 0.986
- Test RMSE: 1.017
- This model produces a very skewed distribution for the market values. There is a lot of error as seen by the RMSE, but this is the model we will try to beat with more refined approaches.

### First improved model:

Same dataset as before, but trying gradient boosting instead of linear regression.

Gradient Boosting:
- Train RMSE: 0.780
- Test RMSE: 0.859
- This is already a significant improvement over linear regression, but it is overfitting

### Minutes played threshold:

We tested some cutoffs on the minimum amount of minutes played for a player to be consired in the model. The idea was to remove outliers that had very few minutes played, as they do not have enough game time to generate stats that accurately represent them. We settled on a cutoff of 1000 minutes (equivalent to a little over 11 full matches played), which kept a reasonable amount of players (over 7500) while producing great results:

Linear Regression:
- Train RMSE: 0.828
- Test RMSE: 0.841

Gradient Boosting:
- Train RMSE: 0.687
- Test RMSE: 0.775

### Ensemble Models:
We split the players into four positions (goalkeeper, defender, midfield and forward) and used our domain knowledge, combined with the EDA analysis, to decide which features to consider for each position. For example, for goalkeepers, goals scored should not matter much, but saves are very important; the opposite is true for forward players. We then tested the following models for each position, performing extensive hyperparameter tuning on each one: Linear Regression with either L1 (Lasso) or L2 (Ridge) regularization, or both; K-Nearest Neighbors; Decision Trees Regression; Random Forest Regression; Gradient Boosting Regression.

What we found is that the best performing model for every position was gradient boosting, with a maximum depth of 2 and number of estimators ranging from 20 to 50, depending on the position:

- Train RMSE: 0.666
- Test RMSE: 0.842

However, comparing the training and test sets, we see this model has significant overfitting.
        
## Conclusions

Our modeling approach aimed to leverage aggregated career statistics to predict soccer player market values effectively. By testing various models and incorporating domain-specific insights, our goal was to provide accurate and actionable predictions for transfer market decisions.

The key ways we tried to improve our results compared to the baseline linear regression model were:
- Introducing a cutoff on the minimum amount of minutes played
- Using position-specific features
- Testing a range of models and performing hyperparameter tuning

Our best performing model was the gradient boosting with the cutoff of 1000 minutes played. which over all positions **improved upon the baseline by ~24%**. The table below shows the RMSE for the models discussed, as well as their percent improvement when comparing RMSE on the test set with the baseline.

![image](https://github.com/user-attachments/assets/3f9570cd-1adb-4d42-9651-89a3c94fa290)

The main difficulty we observed when trying to predict the market values are that outliers with incredibly high market values, or minimal minutes played, can reduce the generalization power of the models. So we end up with significant overfitting when using models other than linear regression.

Given these considerations, in the future we aim to expand this project by exploring the following avenues:

- Gathering data from more competitions:
    - Leagues in other countries
    - National and international cups
- Creating synthetic data for training
- Include information about "decisiveness" and "star power" of players:
    - Titles won
    - Individual awards
    - Popularity (jerseys sold, online following, etc)
- More hyperparameter tuning
- Create different models for players in each country
- Give more weight to most recent statistics
- Test different modeling approaches, like time series

## Instructions for navigating the repo

There is example code in the ```example_code.ipynb``` with more description about this repo.


### Conda Enviroment
Make the conda enviroment by running
```console
conda env create --file=environment.yml
```
### Downloading the data

To download the data, you just need to run the following code

```python
import get_data.get_all_data as gad 
gad.get_data_merge_split()

    # Get the data
train = pd.read_csv('data/main_data/train/train.csv')
train_cutoff = pd.read_csv('data/main_data/train/train_cutoff_1000.csv')

test = pd.read_csv('data/main_data/test/test.csv')
test_cutoff = pd.read_csv('data/main_data/test/test_cutoff_1000.csv')
```

### Creating different models
We constructed a couple of classes built to make different types of models and to ensemble them together into one 
big model.

To access these classes, import the following

```python
import models.main_dataset.ensamble_model as em

# Example of a generalized_Regression model
# This gives us a linear regression model
ex_LR = em.general_Regression(train,type='LR')

#This gives us a linear regression with L2 regularization and regularization factor of 4
ex_RIDGE = em.general_Regression(train,type='RIDGE',alpha=4)

# This gives us a random forest regression model with the various parameters
ex_RFR = em.general_Regression(train,type='RFR',scale='log',max_depth=4,n_estimators=20,min_sample_leaf=2 ,bootstrap=True) 

# This gives us a Gradient Boost regression model with the various parameters
ex_GBR = em.general_Regression(train,type='GBR',scale='log',max_depth=4,n_estimators=20,min_sample_leaf=2 ,bootstrap=True) 

# In each of these, you can perform a cross-validation
ex_LR.perform_CV()
ex_RIDGE.perform_CV()
ex_RFR.perform_CV()
ex_GBR.perform_CV()

```
Hyperparameter tuning on the ```general_Regression``` class

```python
# One can perform hyperparameter tuning on this generalized regression class. You can specify type="something"
# and it will only use parameters for that type of model, otherwise, it will randomly go through different models 
# and their corresponding parameters.

ex_hp = em.hyperparameter_tuning_general(train,n_iter=10,cv=3,model=em.general_Regression,scale='log',beta=1,type=None)

# Perform the tuning
ex_hp.perform_tuning()

# Outputs the best parameters it found and the score.
print(ex_hp.best_params)
print(ex_hp.best_score)

# This is the best model it found, this is a general_Regression class if model=em.general_Regression, otherwise
# it will be any model that is inherited from general_Regression.
ex_hp.best_model

```
Building models for each position

```python
# Build models for each position

#goalkeeper position model
g_pos = em.G_Pos(train,type='LR',scale='log')

# defender position model
d_pos = em.D_Pos(train,type='RIDGE',scale='log',alpha=4)

# Midfielder position model
m_pos = em.M_Pos(train,type='RFR',scale='log',max_depth=4,n_estimators=20,min_sample_leaf=2 ,bootstrap=True)

# Forward position model
f_pos = em.F_Pos(train,type='GBR',scale='log',max_depth=4,n_estimators=20,min_sample_leaf=2 ,bootstrap=True)

# Since these are inherited classes, they have the same methods as general_Regression class
```
Construct Ensemble models for each of the positions
```python

# Now we can talk about our ensemble model, which is essentially takes in each of the position models like above

en_model = em.ensamble_model(scale='log')

# Put the parameters for each position
en_model.G_parameters(type='LR')
en_model.D_parameters(type='RIDGE',alpha=4)
en_model.M_parameters(type='RFR',max_depth=4,n_estimators=20,min_sample_leaf=2 ,bootstrap=True)
en_model.F_parameters(type='GBR',max_depth=4,n_estimators=20,min_sample_leaf=2 ,bootstrap=True)

# Can perform cross-validation
en_model.perform_CV(train,n_splits=5)

#Fit the model
en_model.fit(train)

# Makes a predictions, but it is not scaled back
predictions = en_model.predict(test)


# This makes a prediction, but it scales it back to the original scale (before the ln(1+x))
predictions_scaled_back = en_model.predict_scaled(test)

```
Hyperparameter tuning for the ensemble model
```python

# To do hyperparameter tuning for the ensamble model, we use a specific class.  Note that beta is the penalizing constant

en_hp = em.hyperparameter_tuning(train,n_iter=10,cv=3,scale='log',beta=1)

# Perform the tuning
en_hp.perform_tuning()

# Outputs the best parameters it found and the score
print(en_hp.best_params)
print(en_hp.best_score)

# This is the best model that it outputs, it is a ensemble_model class and has all the usual methods for that class
en_hp.best_model
```

### Main Model

For our main model, you can either grab it from a certain function, or define it with the following parameters

```python
import models.main_model.main_model as mm
main_model = mm.main_model()
```

This is just a ensemble_model class as follows:

```python
import models.main_dataset.ensamble_model as em

main_model = em.ensamble_model(scale='log')

main_modelmodel.G_parameters(type ='GBR',alpha=8,max_depth=2,n_estimators=30,
                                    min_samples_split=2,min_samples_leaf=6,bootstrap=True)

main_model.D_parameters(type ='GBR', alpha=8,max_depth=2,n_estimators=20,
                                    min_samples_split=10,min_samples_leaf=6,bootstrap=True)

main_model.M_parameters(type='GBR',alpha=8,max_depth=2,n_estimators=20,
                                    min_samples_split=10,min_samples_leaf=7,bootstrap=True)

main_model.F_parameters(type = 'GBR',alpha=8,max_depth=2,n_estimators=20,
                                    min_samples_split=10,min_samples_leaf=6,bootstrap=True)

main_model.fit(train)
```




