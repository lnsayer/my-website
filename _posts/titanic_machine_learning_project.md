# First Machine Learning Project: Classification With The Titanic Dataset

<img src="/files/titanic_project/Stöwer_Titanic.jpg" width="500" height="350">

This was my first end-to-end machine learning project on the famous Titanic dataset. It is a classification problem in which we predict which passengers survived the sinking of the Titanic. I have implemented several machine learning techniques during this project and gained a respectable position in the leaderboard, within the top 7% of users. This project is particularly interesting because it demonstrates the biases which machine learning models can exhibit.

This notebook also serves as a guide for people new to machine learning. Please do not hesitate to get in contact if you have any questions, or even more so if you spot anything incorrect!

### The plan:
1. Data exploration and visualisation
2. Data cleaning, feature selection and engineering
3. Cross Validation and Hyperparameter tuning
4. Testing different classifiers

First a little history. RMS (Royal Mail Ship) Titanic was the largest ocean liner in 1912 when she famously sank 370 miles off the coast of Newfoundland, after hitting an iceberg. Out of the 2,200 people onboard, more than 1,500 are estimated to have died in the disaster. Survival rates were starkly different between different passengers, with age, passenger class and sex playing key factors. 

In this project we have access to two datasets: the training set containing 891 passengers (whose survival or death we know) and the test set containing 418 passengers (whose survival or death we must predict).

### 1. Data exploration and visualisation

Before anything let's import the necessary libraries and functions for our project 

``` python
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn
import re 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
This shows our input data files 

```python
/kaggle/input/titanic/train.csv
/kaggle/input/titanic/test.csv
/kaggle/input/titanic/gender_submission.csv
```

We've got three files: the training set, test set and gender submission which is a practice submission file (makes predictions based on the sex of the passenger). Let's load these data and take a look at our features.

``` python
# Load the data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

# Make dataframes
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
combined_df = pd.concat([train_df,test_df], ignore_index=True)

# Set how many rows are set in the dataframe
pd.set_option('display.min_rows', 10) 
pd.set_option('display.max_rows', 10) 

# Show the dataframe
display(combined_df)
```
<img src="/files/titanic_project/train_df_image.png" width="989" height="350"> 

We've got 12 columns with 10 usable features. 

Let's look at the data types and what we are missing:

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1309 entries, 0 to 1308
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  1309 non-null   int64  
 1   Survived     891 non-null    float64
 2   Pclass       1309 non-null   int64  
 3   Name         1309 non-null   object 
 4   Sex          1309 non-null   object 
 5   Age          1046 non-null   float64
 6   SibSp        1309 non-null   int64  
 7   Parch        1309 non-null   int64  
 8   Ticket       1309 non-null   object 
 9   Fare         1308 non-null   float64
 10  Cabin        295 non-null    object 
 11  Embarked     1307 non-null   object 
dtypes: float64(3), int64(4), object(5)
```
We mostly have integers and strings, with some objects which are strings. We are missing data in the Age, Cabin, Fare and Embarked columns. We will have to do some imputation (filling in missing values) and encoding (converting the strings to numerical data which the models can handle).
