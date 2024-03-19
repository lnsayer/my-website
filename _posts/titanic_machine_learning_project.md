# First Machine Learning Project: The Titanic Dataset

<img src="/files/titanic_project/Stöwer_Titanic.jpg" width="500" height="350">

This was my first end-to-end machine learning project on the famous Titanic dataset. It is a classification problem in which we predict which passengers survived the sinking of the Titanic. I have implemented several machine learning techniques during this project and gained a respectable position in the leaderboard, within the top 7% of users. 

This notebook also serves as a guide for people new to machine learning. Please do not hesitate to get in contact if you have any questions, or even more so if you spot anything incorrect!

### The plan:
1. Data exploration and visualisation
2. Data cleaning, feature selection and engineering
3. Cross Validation and Hyperparameter tuning
4. Testing different classifiers

First a little history. RMS (Royal Mail Ship) Titanic was the largest ocean liner at the time and was carrying around 2,200 people onboard. She sank four days into her maiden voyage from Southampton in 1912, after hitting an iceberg 370 miles off the coast of Newfoundland. More than 1500 people were estimated to have died in the disaster. Survival rates are starkly different between different passengers, with age, passenger class and sex playing key factors. 

In this project we have access to two datasets: the training set containing 891 passengers (whose survival or death we know) and the test set containing 418 passengers (whose survival or death we must predict). 

### 1. Data exploration and visualisation

``` js
# Load the data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
```
