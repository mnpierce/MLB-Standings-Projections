# MLB Final Standings Projection Model
Machine Learning Modeling for MLB Season Standings  
In collaboration with https://github.com/t-robarge

# Objective
The original objective for this project was to predict how the MLB standings will finish using stats from about halfway through the season. In reality, we are predicting the number of _wins_ each team will end with. As a result we can compare the projected wins of each team to form our own "model rankings." This can be both league-wide standings, or the more standard divisional standings. Both results can be seen below.

# Results and Validation
Statistics were gathered for seasons 2008-2024 (exluding 2020 due to COVID). Accuracy results were calculated by averaging simulated runs for every year 2008-2024, calculated with the Spearman Rank Correlation Coefficient. When simulating, the chosen year is excluded from the training data to avoid bias. In the following results, "Only using midseason wins" accounts for a dataset that includes midseason wins, while "Midseason wins + 10 additional stats" adds many additional offensive and pitching team statistics. Midseason wins are a very strong predictor for the end of season rankings, so we made sure to verify that there was an improvement when incorporating additional statistics. At this point, only linear regression is used for modeling.

    League-wide rankings:  
        Only using midseason wins: 0.8584506628856584  
        Midseason wins + 10 additional stats: 0.8703498163546692  
    Divisional rankings (averaged):  
        Only using midseason wins: 0.8182271522410419  
        Midseason wins + 10 additional stats: 0.8327452559378887  

_While not a large difference, there is noticeable improvement in accuracy when incorporating additional stats to the model outside of midseason wins count._



_A more intriguing approach is to see how well we can predict without considering midseason wins at all. The stats used for this model are:_ 

```
('runs', 'pitching'), ('era', 'pitching'), ('whip', 'pitching'), ('strikeoutWalkRatio', 'pitching'), ('ops', 'pitching'), ('ops', 'hitting'), ('leftOnBase', 'hitting'), ('runs', 'hitting'), ('hits', 'hitting'), ('avg', 'hitting')
```   

With only the above stats:  
    
    League-wide rankings:  
        Overall Prediction Accuracy: 0.82015907332374  
    Divisional rankings (averaged):  
        Overall Prediction Accuracy: 0.7695890939024339  

_These results are over 75% accurate for each league-wide and divisional standings, showing a strong understanding of these weights by our model._

### Credits
This project uses the mlbstatsapi library by toddrob, which is licensed under the GNU General Public License (GPL) Version 3.  
You can find the original library at: https://github.com/toddrob99/MLB-StatsAPI
