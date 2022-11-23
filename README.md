# Chess-Game-Winner-Prediction


## Introduction


The objective of this project is to use the <b>database</b> from an online chess game (<a href="https://lichess.org/"> Lichess </a>), containing information about players and their rankings, to <b>predict a winner</b> between two players through <b>machine learning</b> technics, based on the players profile data.

In the folder /docs is avaible a <b>portuguese</b> <b>article</b> explaining and evaluating the implementation and its results.

## How it works

The dataset (<b>games.csv</b>) was taken from the website <a href="https://www.kaggle.com/datasets/datasnaek/chess">Kaggle</a>, and was the input to train the two learning models implemented: <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">K-Nearest Neighbors</a> (KNN) and <a href="https://en.wikipedia.org/wiki/Random_forest"> Random Forest</a> (RF). Hyper-parameterization was also used to improve the result, testing different parameters for the models functions, one being <b>Grid Search</b> on the KNN and the other being <b>Random Search</b> on the RF.

### Dataset

The dataset originally has 20056 rows and a total of 16 columns defined this way:

<ul>
<li>Game ID (Game ID);</li>
<li>Rated (T/F) (If it's a ranked match);</li>
<li>Start Time;</li>
<li>End Time;</li>
<li>Number of Turns;</li>
<li>Victory Status (Status of the end of the match);</li>
<li>Winner;</li>
<li>Time Increment (increment time of each turn);</li>
<li>White Player ID;</li>
<li>White Player Rating;</li>
<li>Black Player ID;</li>
<li>Black Player Rating;</li>
<li>All Moves in Standard Chess Notation (All moves in the game);</li>
<li>Opening Eco (Opening type id);</li>
<li>Opening Name;</li>
</ul>

## How to use

<b>It's required that the game.csv file stay in the same folder as the program.</b>

### Input

The set of parameters in the learning algorithms are the inputs that changes the efficiency in the results, in line <b>57</b> and <b>74</b>.

RF parameters:

```python
# números de amostra aleatória de 4 a 204
'n_estimators': [4, 200],
# max_fetures normalmente distribuídos, com média 0,25 stddev 0,1, limitado entre 0 e 1
'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
# distrubuição uniforme de 0.01 a 0.2 (0.01 + 0.199)
'min_samples_split': [2 ,5 ,10],
'max_depth':[10, 50, 100],
```

KNN parameters:

```python
param_grid_knn = [ {'n_neighbors': [1, 5, 10, 25, 50, 750], 'weights':
['distance','uniform'],'leaf_size': [1, 5, 10, 25, 50, 750], 'p':[1, 2]},]
```

### Output

The output will be a <a href= "https://en.wikipedia.org/wiki/F-score"/>F1-score</a> that measure the precision of the algorithm. An example of output using RF will be:

```
{'max_features': 0.3077729475157153, 'min_samples_split': 2, 'n_estimators': 200}
              precision    recall  f1-score   support

           0       0.68      0.48      0.56      2730
           1       0.97      0.96      0.96       271
           2       0.63      0.80      0.70      3016

    accuracy                           0.66      6017
   macro avg       0.76      0.74      0.74      6017
weighted avg       0.67      0.66      0.65      6017

```

### Dependency

The following libraries are required to use the program:

<ul>
<li> <a href="https://pandas.pydata.org/">Pandas</a> (Dataframe creation);</li>
<li><a href="https://matplotlib.org/">Matplotlib</a> (Graphics Plot);</li>
<li><a href="https://seaborn.pydata.org/">Seaborn</a> (Data Viewing);</li>
<li><a href="https://numpy.org/">Numpy</a> (Math operations);</li>
<li><a href="https://imbalanced-learn.org/">Imblearn</a> (Classification and balancing);</li>
<li><a href="https://scikit-learn.org/">Scikit</a> (Machine learning).</li>
</ul>
