# SCORE: Soccer Club Optimization of Recruitment Expenses
### Predicting market values for players in major European soccer leagues
### A 2024 Erdös Institute Project

## Authors
- Cody Tipton &nbsp;<a href="https://www.linkedin.com/in/cody-tipton-21075417b/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="height: 1em; width:auto;"/></a> &nbsp; <a href="https://github.com/shadtome"> <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="height: 1em; width: auto;"/></a>
- Rafael Magaldi &nbsp;<a href="https://www.linkedin.com/in/rafaelmagaldi/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="height: 1em; width:auto;"/></a> &nbsp; <a href="https://github.com/rmmagaldi"> <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="height: 1em; width: auto;"/></a>

## Summary
We use soccer player's in-game statistics to **predict their market value**. This is useful information for soccer clubs when deciding how much to bid when trying to sign a player.

## Background
The soccer transfer market is highly competitive and volatile, with player valuations frequently fluctuating. Clubs have access to two transfer windows per season, when thousands of transfers take place, moving billions of dollars. Clubs often compete to sign the best talent in their budget, so having accurate player valuations is vital when determining how much to spend on a new player. These valuations are what allows clubs to make informed financial decisions and ensure sustainable investments that pay off in the field.

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

Therefore, when combining these data sources, we end up with over 10k players, and have all their in-game stats as well as their personal information like height and date of birth.

## Stakeholders

Our stakeholders are European soccer clubs across various leagues, including both large, established clubs and smaller, emerging teams. They have a vested interest in optimizing the buying and selling of players to enhance both sporting performance and financial sustainability. These clubs seek data-driven insights to avoid overpaying for talent, identify undervalued players, and make informed transfer decisions. Ultimately, their goal is to maximize returns on player investments, maintain competitive squads, and achieve long-term financial health within the volatile football transfer market.

Another potential niche for the models developed in this project are fantasy soccer players, who seek accurate player valuations to optimize their fantasy teams.

## Key Performance Indicators
We want our model to be able to predict market values, so having good accuracy is vital. With that goal in mind, we use the following metrics for all models tested:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- R²

We also want to make sure the model has generalization power when met with new data, so we use these metrics on both the training and testing sets, and compare them.

## Data Preparation and Feature Engineering
Data Collection:
    Gather comprehensive player statistics, including performance metrics, physical attributes, and historical market values.

Approach 1: Aggregated Career Stats
    Aggregate each player's statistics up to the current date or when the player stopped playing.
    Use the most recent known market value as the target variable.
    Split the dataset into training (80%) and testing (20%) sets based on players, ensuring diversity in player profiles and making sure there is a proportional amount of players for each position in the train and test sets.

Approach 2: Time-Series Stock Model (Not pursued due to time constraints)
    Treat each player as a stock, using player stats and market valuations over time.
    Predict the next market valuation using historical data with a time interval of 6 months.

Approach 3: Semi-Aggregated Time-Series
    Initial idea was to aggregate player statistics every 6 months, creating features that reflect progression over time. We would construct a dataset with columns for each 6-month interval, capturing the evolution of player valuations.
    However, given the timeframe of the project, we chose a simpler approach of using the aggregate data for the final 6 months (or maybe also 12 months) as features.

## Model Testing and Evaluation
Baseline Models Tested:

    Simple Linear Regression:
        Initial RMSE of approximately 7 million EUR.
        Enhanced with quadratic and cubic features, slightly improving RMSE.

    Decision Tree:
        Achieved very low RMSE of approximately 6 thousand EUR, but it is overfitting.
        Identified the need for boosting to improve model robustness.

New Models/Approaches:

    Feature Selection:
        Identify significant features for each position.
        Tailor models for specific positions to enhance prediction accuracy.

    XGBoost Regression:
        Implement to reduce variance and improve upon decision tree performance.

    Ensemble Models:
        Develop ensembles of linear regression models based on position, league, and team, incorporating quadratic and cubic features when needed.
        Introduce penalties for age to account for market depreciation.

    Ensemble of XGBoost:
        Create position-based ensembles to leverage XGBoost's strengths in handling complex interactions.

    Extended Time-Series Features:
        Introduce datasets with aggregated results up to 6 months ago, adding new columns for the past 6 months.
        Experiment with 12-month aggregation to capture longer-term trends.
            
## Conclusion
This modeling approach aims to leverage both aggregated career statistics and temporal data to predict soccer player market values effectively. By testing various models and incorporating domain-specific insights, the goal is to provide accurate and actionable predictions for transfer market decisions.


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




