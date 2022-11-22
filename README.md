# Chess-Game-Winner-Prediction


## Introduction


The objective of this project is to use the <b>database</b> from an online chess game (<a href="https://lichess.org/"> Lichess </a>), containing information about players and their rankings, to <b>predict a winner</b> between two players through <b>machine learning</b> technics, based on the players profile data.

## How it works

The dataset (<b>games.csv</b>) was taken from the website <a href="https://www.kaggle.com/datasets/datasnaek/chess">Kaggle</a>, and was the input to train the two learning models implemented: <a href="https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm">K-Nearest Neighbors</a> (KNN) and <a href="https://en.wikipedia.org/wiki/Random_forest"> Random Forest</a> (RF). Hyper-parameterization was also used to improve the result, testing different parameters for the models functions, one being <b>Grid Search</b> on the KNN and the other being <b>Random Search</b> on the RF.

## Data-set

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

## Input

The set of parameters in the learning algorithms are the inputs that changes the efficiency in the results
