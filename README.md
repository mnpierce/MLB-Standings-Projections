# MLB Final Standings Projection Model
Machine Learning Modeling for MLB Season Standings  
In collaboration with https://github.com/t-robarge

# Objective
The original objective for this project was to predict how the MLB standings will finish using stats from about halfway through the season. In reality, we are predicting the number of _wins_ each team will end with. As a result we can compare the projected wins of each team to form our own "model rankings." This can be both league-wide standings, or the more standard divisional standings. Both results can be seen below.

# Results - Validating the use of additional stats on top of midseason wins
Statistics were gathered for seasons 2008-2024 (exluding 2020 due to COVID). Accuracy results were calculated by averaging simulated runs for every year 2008-2024, calculated with the Spearman Rank Correlation Coefficient. When simulating, the chosen year is excluded from the training data to avoid bias. In the following results, "Only using midseason wins" accounts for a dataset that includes midseason wins, while "Midseason wins + 10 additional stats" adds many additional offensive and pitching team statistics. Midseason wins are a very strong predictor for the end of season rankings, so we made sure to verify that there was an improvement when incorporating additional statistics. 

## Linear Models

### LinearRegression

    RMSE (wins only): 6.1396 wins
    RMSE (wins + 10 stats): 5.8587 wins

    League-wide Accuracy:  
        Only using midseason wins: 0.8585  
        Midseason wins + 10 additional stats: 0.8678
    Divisional Accuracy:  
        Only using midseason wins: 0.8155  
        Midseason wins + 10 additional stats: 0.8187

### RidgeCV

    RMSE (wins only): 6.1403 wins
    RMSE (wins + 10 stats): 5.8470 wins

    League-wide Accuracy:  
        Only using midseason wins: 0.8585
        Midseason wins + 10 additional stats: 0.8679 
    Divisional Accuracy:  
        Only using midseason wins: 0.8155 
        Midseason wins + 10 additional stats: 0.8171

### Linear Findings
For each of these linear regression models, there is noticeable improvement in the RMSE as well as the overall percentages. Every accuracy metric improved overall, showing that there is merit to training a model for this purpose, as midseason wins do not produce the best score alone.

## Ensemble Models

### RandomForestRegressor

    RMSE (wins only): 6.3305 wins
    RMSE (wins + 10 stats): 6.0543 wins

    League-wide Accuracy:  
        Only using midseason wins: 0.8592
        Midseason wins + 10 additional stats:  0.8534
    Divisional Accuracy:  
        Only using midseason wins: 0.8209 
        Midseason wins + 10 additional stats: 0.8037

### GradientBoostingRegressor

    RMSE (wins only): 6.3491 wins
    RMSE (wins + 10 stats): 6.3017 wins

    League-wide Accuracy:  
        Only using midseason wins: 0.8582
        Midseason wins + 10 additional stats: 0.8464 
    Divisional Accuracy:  
        Only using midseason wins: 0.8206
        Midseason wins + 10 additional stats: 0.8018

### Ensemble Findings
Because ensemble models are so powerful, we actually see a decrease in performance when providing the model with 10 select additional stats. These stats are too simple and not enough volume to take advantage of the complex patterns ensemble models can extract. It is worth noting that the RMSE still improved marginally. This shows that there is some value to using these stats on top of wins for these models.


# Results - Excluding feature: midseason wins 
_A more intriguing approach is to see how well we can predict without training on midseason wins at all. The stats used for this model are:_ 

```
('runs', 'pitching'), ('era', 'pitching'), ('whip', 'pitching'), ('strikeoutWalkRatio', 'pitching'), ('ops', 'pitching'), ('ops', 'hitting'), ('leftOnBase', 'hitting'), ('runs', 'hitting'), ('hits', 'hitting'), ('avg', 'hitting')
```   

## Linear Models
    
### LinearRegression

    RMSE: 6.6167 wins

    League-wide Accuracy: 0.8219  
    Divisional Accuracy: 0.7795  

### RidgeCV

    RMSE: 6.6016 wins

    League-wide Accuracy: 0.8196  
    Divisional Accuracy: 0.7732  

## Ensemble Models

### RandomForestRegressor

    RMSE: 6.8916 wins

    League-wide Accuracy: 0.8059  
    Divisional Accuracy: 0.7499  

### GradientBoostingRegressor

    RMSE: 7.1452 wins

    League-wide Accuracy: 0.7911  
    Divisional Accuracy: 0.7196

## Findings
For both linear and ensemble models, an accuracy of close to 80% is achieved, even after excluding the powerful "midseason wins" feature. This serves to show a strong predictive model can be built from these auxiliary stats. At this stage, given the small number of features, the simple linear models are still outperforming the more powerful ensemble models. The next step is to incorporate more stats to fully utilize the ensemble capabilities.

# Results - Large feature set
For this part of the project, we will be assessing the results after significantly increasing the number of features for training. Instead of using the select 10 features from the previous part, we will use 67 features.

They are as follows:

```python
[
            ('runs', 'pitching'),
            ('era', 'pitching'),
            ('whip', 'pitching'),
            ('strikeoutWalkRatio', 'pitching'),
            ('ops', 'pitching'),
            ('ops', 'hitting'),
            ('leftOnBase', 'hitting'),
            ('runs', 'hitting'),
            ('hits', 'hitting'),
            ('avg', 'hitting'),
            ('earnedRuns', 'pitching'),
            ('slg', 'hitting'),
            ('obp', 'hitting'),
            ('stolenBases', 'hitting'),
            ('blownSaves', 'pitching'),
            ('groundOuts', 'hitting'),
            ('airOuts', 'hitting'),
            ('doubles', 'hitting'),
            ('triples', 'hitting'),
            ('homeRuns', 'hitting'),
            ('strikeOuts', 'hitting'),
            ('baseOnBalls', 'hitting'),
            ('intentionalWalks', 'hitting'),
            ('hitByPitch', 'hitting'),
            ('atBats', 'hitting'),
            ('caughtStealing', 'hitting'),
            ('stolenBasePercentage', 'hitting'),
            ('groundIntoDoublePlay', 'hitting'),
            ('numberOfPitches', 'hitting'),
            ('plateAppearances', 'hitting'),
            ('totalBases', 'hitting'),
            ('rbi', 'hitting'),
            ('sacBunts', 'hitting'),
            ('sacFlies', 'hitting'),
            ('groundOuts', 'pitching'),
            ('airOuts', 'pitching'),
            ('doubles', 'pitching'),
            ('triples', 'pitching'),
            ('strikeOuts', 'pitching'),
            ('hitByPitch', 'pitching'),
            ('avg', 'pitching'),
            ('obp', 'pitching'),
            ('slg', 'pitching'),
            ('atBats', 'pitching'),
            ('caughtStealing', 'pitching'),
            ('stolenBases', 'pitching'),
            ('stolenBasePercentage', 'pitching'),
            ('groundIntoDoublePlay', 'pitching'),
            ('saveOpportunities', 'pitching'),
            ('holds', 'pitching'),
            ('battersFaced', 'pitching'),
            ('outs', 'pitching'),
            ('shutouts', 'pitching'),
            ('strikes', 'pitching'),
            ('strikePercentage', 'pitching'),
            ('hitBatsmen', 'pitching'),
            ('balks', 'pitching'),
            ('wildPitches', 'pitching'),
            ('pickoffs', 'pitching'),
            ('totalBases', 'pitching'),
            ('groundOutsToAirouts', 'pitching'),
            ('pitchesPerInning', 'pitching'),
            ('strikeoutsPer9Inn', 'pitching'),
            ('walksPer9Inn', 'pitching'),
            ('hitsPer9Inn', 'pitching'),
            ('runsScoredPer9', 'pitching'),
            ('homeRunsPer9', 'pitching')
    ]
```

## Linear Models

### LinearRegression
    RMSE: 34.1489 wins

    League-wide Accuracy: 0.3792
    Divisional Accuracy: 0.3449

### RidgeCV
    RMSE: 6.5293 wins

    League-wide Accuracy: 0.8329
    Divisional Accuracy: 0.7787

## Ensemble Models
### RandomForestRegressor
    RMSE: 6.8721 wins

    League-wide Accuracy: 0.8095
    Divisional Accuracy: 0.7509
### GradientBoostingRegressor
    RMSE: 6.6367 wins

    League-wide Accuracy: 0.8148
    Divisional Accuracy: 0.7600

## Final Findings
These results agree with what we would expect for a feature set of this size and correlation. Because many of these features are highly correlated and there is a large volume, simple linear regression performs extremely poorly. This is primarily due to multicollinearity, which is addressed well by RidgeCV, introducing L2 regularization and producing the best score excluding midseason wins.

Coming in just after RidgeCV are our ensemble methods, which are able to extract complex patterns from this large feature set.

### Credits
This project uses the mlbstatsapi library by toddrob, which is licensed under the GNU General Public License (GPL) Version 3.  
You can find the original library at: https://github.com/toddrob99/MLB-StatsAPI
