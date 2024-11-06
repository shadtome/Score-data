# Predicting Market-values for players in Major European Football Leagues
### SCORE

## Objective 
Our objective is to use player historical data in their career to predict their current market value for each transfer window. So that clubs can make financially sound investments for players they want to buy.

## Stakeholders

Our stakeholders are European football clubs across various leagues, including both large, established clubs and smaller, emerging teams. They have a vested interest in optimizing the buying and selling of players to enhance both sporting performance and financial sustainability. These clubs seek data-driven insights to avoid overpaying for talent, identify undervalued players, and make informed transfer decisions. Ultimately, their goal is to maximize returns on player investments, maintain competitive squads, and achieve long-term financial health within the volatile football transfer market.

## KPI's
- Prediction Accuracy
    - stuff
- ROI from transfers
    - stuff
- Player Performance post-transfer
    - stuff
- Financial impact on club's budget
    - stuff
- Stakeholder satistfaction 
- Determine player's ROI
- Determine players Market values
- Determine clubs financial risk in buying a player or not based on team success

## Possible Products Avenues

- data-driven recommendations
    - Generate recommendations for transfers based on the club's financial constriatins.

## Dataset
Our data is sourced from the following websites:

- Transfermarkt: https://www.transfermarkt.us/
    - This has the transfer fees, market values, and general stats about each player in the European football leagues.
- Sofascore: https://www.sofascore.com/
    - This contains all the player and team stats in every game, including a sofascore rating measuring how well a player is in a match.
- Understat: https://understat.com/
    - This contains some general stats about each player and game in the English Premier League, La Liga, Bundesliga, Ligue 1, Serie A, and Russian Premier League.


### Raw Data
Our raw data has the following features:
- name of player
- 


## Our train-test data
    Lets say we are working for a top English Premier League club right now in the 2024/2025 season and the January transfer season is coming up and the manager is looking at a certain player that could be helpful for a certain position.  What would be a good market value for this player given their career performance so far? This is what we want to determine using our data.

There are two ideas we have to predict market values for players and how we will set up our train and test data.
- We can aggregate each of the players stats all the way up to when either the player has stopped playing or the current date with their target variable being their most recent known market value.  Then we can create our train-test split with our collection of players, i.e., we have 80% of the players in our train dataset with their aggregated stats to train on.  We could build simple models based on this data with respect to their career stats
- Another way is to treat each player as a stock, i.e., we have their player stats and valuations over time and we try to predict the next market valuation using the past data.  Our $\Delta t$ would be every 6 months, which is where the valuations are made. This has some complications in terms of having enough data to even make a good prediction.  
- Lastly, one that combines both of the ideas above is to aggregate the stats every 6 months for each player, but put them in the columns, so it would be simlar to the first bullet point, but instead of aggregating the whole career, it is seperated out so one can see the progress of valuations of the player. This would create a lot of features in the data.




## Models

### Baseline Models
- Simple Linear Regression with all the features Has RMSE $\approx 7 million$
- Simple Linear regression with a few quadratic and cubic features.  Has RMSE $\approx 7 million$ (but slightly better then simple linear regression)
- Decision Tree with all the features.  Has RMSE $\approx 6 thousand EUR$ but it is overfitting. (Hence boosting will improve this)

### New Models
- Look at the significant features and use just those. (Do this for each position) (Also see if we can add awards/achievements)
- XGBoost Regression, this will improve the decision tree above decreasing the variance of the model.
- Ensamble of linear regression models based on position/league/team (include the quad and cubic features if it improves, have a heavy penatly to age).
    - Weight the features that are more important for each position
- Ensamble of XGBoost regression based on position
- Introduce new dataset with aggregated results upto 6 months ago and then new columns for the past 6 months.
    - Same thing, but do it for more then 6 months, like agg upto 12 months ago and new columns for the past 12 months.






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




