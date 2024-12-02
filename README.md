# SCORE: Soccer Club Optimization of Recruitment Expenses
### Predicting market values for players in major European soccer leagues
### A 2024 Erdös Institute Project

## Authors
- Cody Tipton &nbsp;<a href="https://www.linkedin.com/in/cody-tipton-21075417b/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="height: 1em; width:auto;"/></a> &nbsp; <a href="https://github.com/shadtome"> <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="height: 1em; width: auto;"/></a>
- Rafael Magaldi &nbsp;<a href="https://www.linkedin.com/in/rafaelmagaldi/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="height: 1em; width:auto;"/></a> &nbsp; <a href="https://github.com/rmmagaldi"> <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="height: 1em; width: auto;"/></a>

## Summary
We use soccer player's in-game statistics to **predict their market value**. This is useful information for soccer clubs when deciding how much to bid when trying to sign a player.

## Background
The soccer transfer market is highly competitive and volatile, with player valuations frequently fluctuating. Clubs have access to two transfer windows per season, when thousands of transfers take place worldwide, moving billions of dollars. Clubs often compete to sign the best talent within their budget, so having accurate player valuations is vital when determining how much to spend on a new player. These valuations are what allows clubs to make informed financial decisions and ensure sustainable investments that pay off in the field.

## Dataset
Our data is sourced from the following websites:

- Transfermarkt: https://www.transfermarkt.us/
    - This has the transfer fees, market values, and general stats about players worldwide.
- Sofascore: https://www.sofascore.com/
    - This contains all the player and team stats in every professional game, including a sofascore rating measuring how well a player did in a match.
- Understat: https://understat.com/
    - This contains some general stats about each player and game in the English Premier League, La Liga, Bundesliga, Ligue 1, Serie A, and Russian Premier League.

For the Transfermarkt data, we were able to use a dataset available on Kaggle (https://www.kaggle.com/datasets/davidcariboo/player-scores) that is being automatically updated, so we have up-to-date information on the market values of all players.
As for the player stats, we performed web scraping to gather them from Understat (using Beautiful Soup) and Sofascore (using selenium and Sofascore's API). We chose to gather data from the top 10 strongest soccer leagues in Europe (https://theanalyst.com/2024/10/strongest-leagues-world-football-opta-power-rankings), across as many seasons as they had data for (ranging from 5 to 11 seasons depending on the league).

Therefore, when combining these data sources, we end up with over 10k players, and have all their in-game stats as well as their personal information like name, height and date of birth.

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
Given the player stats on each match, the approach we decided to follow was to aggregate the statistics up to the current date or whenenever the player stopped playing (using sums or averages, depending on the feature). We also chose to use the last known market value for each player as the target, and adjusted it for inflation when it was not current (for instance, for players who stopped playing before the current season). We noted the target had high skewness and kurtosis, which was causing some negative market value predictions with linear regression, so to make the distribution closer to a normal one, we decided to apply ln(1+x) to the market values. This also has the benefit of helping reduce outliers, as now the range of the target is between the values of 10 and 20. We then split the dataset into training (80%) and testing (20%) sets, ensuring diversity for player profiles and making sure there are proportional amounts of players for each position in each set.

## EDA
For our exploratory data analysis, we checked the influence of each feature on the market value, noting that players that play on different positions have different sets of features that matter most when determining their market values.

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

Our best performing model was the gradient boosting with the cutoff of 1000 minutes played. which over all positions **improved upon the baseline by ~24%**. The table below shows the RMSE for the models discussed, as well as their percent improvement over the baseline.

![image](https://github.com/user-attachments/assets/3f9570cd-1adb-4d42-9651-89a3c94fa290)

The main difficulty we observed when trying to predict the market values are that outliers with incredibly high market values, or minimal minutes played, can reduce the generalization power of the models. So we end up with significant overfitting when using models other than linear regression.

Given these considerations, in the future we aim to expand this project by exploring the following avenues:

- Gatheriong data from more competitions:
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
Cody, please change this and the following sections according to however you structured the repo.

## Conda Enviroment
Make the conda enviroment by running
```console
conda env create --file=environment.yml
```
## Downloading the data


## Sample Code

```python
import sample code
```




