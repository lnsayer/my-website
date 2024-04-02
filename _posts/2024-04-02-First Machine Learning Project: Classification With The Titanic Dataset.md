---
layout: post
title: First Machine Learning Project
subtitle: Classification with the Titanic dataset
cover-img: /files/titanic_project/Stöwer_Titanic.jpg
thumbnail-img: /files/titanic_project/Stöwer_Titanic.jpg
share-img: /assets/img/path.jpg
tags: [Machine Learning, Classification, Tutorial]
author: Louis Sayer
mathjax: true
full-width: true
---

[1]

This was my first end-to-end machine learning project on the famous Titanic dataset. It is a classification problem in which we predict which passengers survived the sinking of the Titanic. I have implemented several machine learning techniques during this project and gained a respectable position in the leaderboard, within the top 7.3% of users. This project is particularly interesting because it demonstrates the biases which machine learning models can exhibit.

This notebook also serves as a guide for people new to machine learning. Please do not hesitate to get in contact if you have any questions, or even more so if you spot anything incorrect!

### What we will cover:
1. Data exploration and visualisation
2. Data cleaning, feature selection and engineering
3. Quick Modelling to see feature importance
4. Cross Validation and Hyperparameter tuning
5. Testing different classifiers and submitting a prediction

First a little history. RMS (Royal Mail Ship) Titanic was the largest ocean liner in 1912 when she famously sank 370 miles off the coast of Newfoundland, after hitting an iceberg. Out of the 2,200 people onboard, more than 1,500 are estimated to have died in the disaster. Survival rates were starkly different between different passengers, with age, passenger class and sex playing key factors. We will predict from these factors, along with others, which passengers survived or died. 

In this project we have access to two datasets: the training set containing 891 passengers (whose survival or death we know) and the test set containing 418 passengers (whose survival or death we must predict).

### 1. Data exploration and visualisation

Before anything else let's import the necessary libraries and functions for this project 

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

We've got three files: the training set file, test set file and gender submission file which is a practice submission file (makes predictions based on the sex of the passenger). Let's load these data and take a look at our features.

``` python
# Load the data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

# Make dataframes
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
combined_df = pd.concat([train_df,test_df], ignore_index=True)

# Set how many rows are shown in the dataframe
pd.set_option('display.min_rows', 10) 
pd.set_option('display.max_rows', 10) 

# Show the dataframe
display(combined_df)
```
<img src="https://lnsayer.github.io/my-website/files/titanic_project/combined_df.png" alt="Untitled" width="844" height="350"/>

Note, NaN means not a number.

We've got 12 columns with 10 usable features. 
- PassengerId is an index
- Survived is 1 if they survived or 0 if they died
- PClass shows the passenger class (1st, 2nd or 3rd)
- Name gives their full name and title
- Sex, male or female
- Age in years
- SibSp is number of siblings/spouses on the ship
- Parch is number of parents/children on the ship
- Ticket shows their ticket code
- Fare in dollars
- Cabin, with deck classification
- Embarked gives their port of embarkation: S: Southampton, C: Cherbourg, Q: Queenstown

Let's look at the data types and what we are missing:

```python
combined_df.info()
```
returns 
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
We mostly have integers and objects, which are strings. We are missing data in the Age (263 missing), Cabin (1014), Fare (1) and Embarked (2) columns. We will have to do some imputation (filling in missing values) and encoding (converting the strings to numerical data which the models can handle).

Since we have combined the training and test set into one dataframe we can see all the missing data. If we had only looked at the training set we would miss the missing fare entry in the test set. Normally we would impute the training and test set separately since our test set represents future data we want to predict. However, as this is the same ship with the same distribution it is okay to impute altogether. It also makes our life easier to do so. 

Let's explore how different features might affect survival numbers. We can display the survival rate based on age and sex at the same time with some plots:
``` python
# Dataframes of male and female survivals
male_train_df = train_df.loc[train_df['Sex'] == 'male']
female_train_df = train_df.loc[train_df['Sex'] == 'female']

male_survived_df = male_train_df.loc[male_train_df['Survived'] == 1]
male_died_df = male_train_df.loc[male_train_df['Survived'] == 0]

female_survived_df = female_train_df.loc[female_train_df['Survived'] == 1]
female_died_df = female_train_df.loc[female_train_df['Survived'] == 0]

# Initiating the plots
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(14,6), sharex=True)

# Male and female survived histograms, respectively
histogram_male_survived = ax0.hist(male_survived_df['Age'], bins=20, alpha=0.5,  label='Survived', color='C0', zorder=1);
histogram_male_died = ax0.hist(male_died_df['Age'], bins=20, alpha=0.5,  label='Died', color='C1', zorder=0);

histogram_female_survived = ax1.hist(female_survived_df['Age'], bins=20, alpha=0.5,  label='Survived');
histogram_female_died = ax1.hist(female_died_df['Age'], bins=20, alpha=0.5,  label='Died');

# Add a legend to ax0
ax0.legend()
ax1.legend()
# Set titles
ax0.set(title='Male', xlabel='Age', ylabel='Number of people');
ax1.set(title='Female', xlabel='Age', ylabel='Number of people');
# Set labels
ax0.set_xlabel('Age (yrs)', fontsize = 12)
ax0.set_ylabel('Number of people', fontsize = 12)
ax1.set_xlabel('Age (yrs)', fontsize = 12)
ax1.set_ylabel('Number of people', fontsize = 12)
```

<img src="/my-website/files/titanic_project/survival_rate_age_sex.png" width="auto" height="450"> 

These are quite stark results with the men much more likely to die than women. Age does not appear to affect survival numbers amongst women, however young men are much more likely to survive than older men. The famous unofficial code of conduct 'Women and children first" is confirmed by these plots.

We can also plot the survival numbers according to fare price:

```python
# Plot histograms of the number who survived/died according to fare price
survived_df = train_df.loc[train_df['Survived'] == 1]
died_df = train_df.loc[train_df['Survived'] == 0]

# Initiating the figures
fig, ax = plt.subplots(figsize=(14,6))

# Histograms of the survived and died based on fare price
survived_hist = ax.hist(survived_df['Fare'], bins=20, alpha=0.5,  label='Survived', color='C0', zorder=0);
died_ = ax.hist(died_df['Fare'], bins=20, alpha=0.5,  label='Died', color='C1', zorder=1);

# Add a legend to ax0
ax.legend()
# Set labels
ax.set_xlabel('Fare ($)', fontsize = 12)
ax.set_ylabel('Number of people', fontsize = 12)
```
<img src="/my-website/files/titanic_project/survival_rate_fare_price.png" width="auto" height="450">  

The tail of the survivors is longer showing that people who paid a higher fare were more likely to survive. However, this is not strictly the case, some passengers who bought an expensive ticket also died. 

Finally let's look at the survival numbers amongst different passenger classes. 

```python
pclass1_survive = train_df.loc[(train_df['Survived'] == 1) & (train_df['Pclass'] ==1 )]
pclass1_die = train_df.loc[(train_df['Survived'] == 0) & (train_df['Pclass'] ==1 )]

pclass2_survive = train_df.loc[(train_df['Survived'] == 1) & (train_df['Pclass'] ==2 )]
pclass2_die = train_df.loc[(train_df['Survived'] == 0) & (train_df['Pclass'] ==2 )]

pclass3_survive = train_df.loc[(train_df['Survived'] == 1) & (train_df['Pclass'] ==3 )]
pclass3_die = train_df.loc[(train_df['Survived'] == 0) & (train_df['Pclass'] ==3 )]

survived = [len(pclass1_survive), len(pclass2_survive), len(pclass3_survive)]
died = [len(pclass1_die), len(pclass2_die), len(pclass3_die)]

fig, ax = plt.subplots(figsize=(10,4))

x = np.arange(3)
width = 0.4

ax.bar(x-0.2, survived, width) 
ax.bar(x+0.2, died, width) 
ax.set_xticks(x, ['1st Class', '2nd Class', '3rd Class'], size =14) 
ax.set_xlabel("Passenger Class", size =14) 
ax.set_ylabel("Number of people", size =14) 
ax.legend(["Survived", "Died"], fontsize =12)
```

Note, I have since realised you can compare different features by plotting directly with the Pandas' dataframe using Crosstab which is a lot easier, e.g. `crosstab = pd.crosstab(train_df.Survived, df.PClass)` then `plot = crosstab.plot(kind='bar', rot = 0, figsize=(10,6) )`

<img src="/my-website/files/titanic_project/survival_rate_passenger_class.png" width="auto" height="450">  

Again, better passenger class corresponds to a higher chance of survival. On the Titanic, you can put a price on life.

### 2. Data cleaning, feature selection and engineering

Let's try and make some other features useful and even create a new feature. 
First, we will extract all the titles of the passengers. These might be useful since it can explicity tell the model whether a passenger is an adult or child, as well as their marital status. This code makes a new column 'Title' which returns the string following a comma and space and then a full stop e.g. for `, Mr.` it returns `Mr`. We can see how many of the original entries have each title with `value_counts`. 

```python
pd.set_option("display.max_rows", None)
# New data frame to work with
new_train_df = train_df.copy(deep=True)

# Function to create a new column with the title of each passenger
new_train_df['Title'] = new_train_df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
print(new_train_df['Title'].value_counts())
```
```python
Title
Mr              517
Miss            182
Mrs             125
Master           40
Dr                7
Rev               6
Mlle              2
Major             2
Col               2
the Countess      1
Capt              1
Ms                1
Sir               1
Lady              1
Mme               1
Don               1
Jonkheer          1
Name: count, dtype: int64
```
We can see there are several unusual titles, let's transfer them to some of the more common titles with this function:

```python
# Function to allocate uncommon titles to broader title categories
def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:
        return 'Mr'
    elif title in ['the Countess', 'Mme', 'Lady', 'Dona']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
```
In hindsight, we could have considered putting some of these in a special column but we will leave that for further work. Next we will extract the deck of the passenger cabins with another function: 
```python
# Function to replace the cabin code with their deck section, denoted by a letter
def replace_cabin(x):
    x['Cabin'] = x['Cabin'].fillna("U0")
    x['Deck'] = x['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    x['Deck'] = x['Deck'].map(deck)
    x['Deck'] = x['Deck'].fillna("U")
    x.drop('Cabin',axis=1, inplace=True)
    
    return x
```
We will fill empty cabin entries with `U` for unknown. Perhaps some areas of the boat were able to evacuate more easily than others. The boat hit the iceberg at 11:40pm so it is likely the passengers were inside their cabins at this time. 

Finally, we will make a family_size attribute which gives the family size of each passenger:
```python
# Function to define a person's family size
def add_family(x):
    x['Familysize'] = x['SibSp']+x['Parch']+1
    return x
```
Let's check all these are working with our training dataset: 
```python
# Show the new altered dataframes with 'Title', 'Deck' and 'Familysize' columns
new_train_df['Title'] = new_train_df.apply(replace_titles, axis=1)
new_train_df = replace_cabin(new_train_df)
new_train_df = add_family(new_train_df)

new_train_df
```
 <img src="/my-website/files/titanic_project/new_features.png" width="auto" height="400">   

 Next we are going to impute our data and one hot encode the categorical data. For this I created a function `prepare_dataframe` which can do this to any chosen dataframe and can also drop columns we are not interested in using. It does the following:
 - Converts the sex column into binary (0 and 1s)
 - Replaces the Name column with Title
 - Replaces the Cabin column with Deck
 - Adds family size
 - Imputes Fare with the median value
 - One hot encodes Embarked, Title, Deck and Pclass
 - Imputes the Age column using a KNN algorithm
 - Renames engineered columns
 - Drops any columns which are requested

```python
def prepare_dataframe(df, drop_columns):
    # Copying dataframe to manipulate
    new_df = df.copy(deep=True)
    
    # Binary mapping the sex column
    binary_mapping = {"male" : 0, "female": 1}
    new_df["Sex"] = new_df["Sex"].map(binary_mapping)
    
    # Creating the new Title and Deck columns
    new_df['Title'] = new_df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
    new_df['Title'] = new_df.apply(replace_titles, axis=1)
    
    # Add a column with their deck section
    new_df = replace_cabin(new_df)
    # Add a column with their family size
    new_df = add_family(new_df)
    
    # Numeric and categorical features to encode
    numeric_features = ["Fare"]
    categorical_features = ["Embarked", "Title", "Deck", 'Pclass']
    
    # Strategies for transforming these features
    numeric_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy="median"))])
    
    categorical_transformer = Pipeline(steps = [ ("imputer", SimpleImputer(strategy = "constant", 
                                                                           fill_value="missing")),
                                               ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    # Transforming these features    
    preprocessor = ColumnTransformer(transformers = [("num", numeric_transformer, numeric_features),
                                                    ("cat", categorical_transformer, categorical_features)])
    
    preprocessor.fit(new_df)
    
    transformed_data = preprocessor.transform(new_df)
    
    # Getting transformed data and creating new columns to put them in 
    numeric_data = transformed_data[:, :len(numeric_features)].toarray()
    categorical_data = transformed_data[:, len(numeric_features):].toarray()
        
    categorical_encoded_features = preprocessor.named_transformers_['cat']['onehot'] \
                                    .get_feature_names_out(input_features=categorical_features)
    
    # Replace the columns with transformed data
    new_df[categorical_encoded_features] = categorical_data
    new_df[numeric_features] = numeric_data
    
    # Impute the missing age data using a KNN algorithm utilising the following features 
    X = new_df[['SibSp', 'Fare', 'Age', 'Title_Master', 'Title_Miss',
       'Title_Mr', 'Title_Mrs', 'Pclass', 'Sex']]

    impute_knn = KNNImputer()
    X_imputed = impute_knn.fit_transform(X)

    X_df = pd.DataFrame(X_imputed)
    age_column = X_df.iloc[:,2]
    new_df['Age'] = age_column
    
    # Removing obsolete features which have been transformed 
    if "Embarked_missing" in new_df.columns:
        new_df.drop("Embarked_missing", axis=1, inplace=True)
    if "Title" in new_df.columns:
        new_df.drop("Title", axis=1, inplace=True)
    if "Deck" in new_df.columns:
        new_df.drop("Deck", axis=1, inplace=True)
    if "Pclass" in new_df.columns:
        new_df.drop("Pclass", axis=1, inplace=True)
        
    # Dropping custom columns according to which features we want to include in a model
    new_df.drop(drop_columns,axis =1, inplace=True)
    
    return pd.DataFrame(new_df)
```

This gave a final dataset of: 
```python
# Custom columns to drop in the function prepare_dataframe
drop_columns = ["Embarked", "Ticket", "Name"]
new_train_df = prepare_dataframe(train_df, drop_columns)
print(new_train_df.columns)
new_train_df['Age'].isna().sum()
new_train_df
```
 <img src="/my-website/files/titanic_project/prepared_dataset.png" width="auto" height="300"> 

Some notes:
 - The custom dropped columns could have been dropped automatically in the `prepare_dataframe` function since all those features had been imputed or one-hot encoded. When I was experimenting with using different features I used this quite a lot.
 -  The cause of the missing age values is not immediately clear (missing at random or not at random [2]). This is something we could have investigated further. I did not want to remove this column as it is such a useful predictor. Imputing using the median age would not give the best estimate for age since the passengers' ages vary depending on their passenger class, fare etc. There are other variables we can use to predict the missing values such as their title or number of siblings (children are more likely to travel with siblings than adults are). For this reason I decided to use a k-nearest neighbour algorithm because it is accurate, simple and fast. See my possible improvements at the bottom of the article for more information. Imputing the missing ages gave a similar distribution compared to the original non-missing data: 

```python
# Columns to drop in preparing the dataframes
drop_columns = ["Embarked", "Ticket", "Name", "PassengerId"]

new_train_df = prepare_dataframe(train_df, drop_columns)
new_test_df = prepare_dataframe(test_df, drop_columns)

#Plotting histograms of the feature variables
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize = (8,5))

# Create histograms
ax0.hist(train_df["Age"], bins=20);
ax1.hist(new_train_df["Age"], bins=20);

# Set labels
ax0.set_xlabel('Age (yrs)', fontsize = 12)
ax0.set_ylabel('Number of people', fontsize = 12)
ax1.set_xlabel('Age (yrs)', fontsize = 12)
ax0.set_title('Original dataframe')
ax1.set_title('Imputed dataframe')

print(new_train_df["Age"].isna().sum())
print(train_df["Age"].isna().sum())
```

 <img src="/my-website/files/titanic_project/imputed_age.png" width="auto" height="450">  

 - Another option for imputing the age would have been to set the missing values as 0. It might have been the case that passengers had a missing age *because* they had died and their records were not kept. Setting the missing values as 0 might have been able to capture this. Further work could explore the effect of this. 

- For the deck code we decided to keep the missing values as a new column because like age the missing values might have been caused by a passenger dying. We will see how this column turned out to be important in the models' feature importances.
- Since there is only one missing feature from the Fare column we imputed this with the median.
- We decided to one-hot encode the Pclass column but in fact this was unnecessary and only made the dataframe more sparse. We could have kept this in its original form. 

### 3. Quick Modelling to see feature importance

To train our first model we have to split up the data into the independent features `x_train` (all the independent variables such as age, sex columns etc.) and the dependent target `y_train` (the survived column). In this case we have imputed the data separately but we will change this later on.

```python
# Setup the random seed
np.random.seed(42)

# Prepare dataframes
new_train_df = prepare_dataframe(train_df, drop_columns)
new_test_df = prepare_dataframe(test_df, drop_columns)

# Split up into feature variables and target variables
x_train = new_train_df.drop(["Survived"], axis=1)
y_train = new_train_df["Survived"]
```

We can quickly model the data using a Random forest classifier, which aggregates the predictions of many decision trees (for example 100). A single decision tree looks like this: 

<img src="/my-website/files/titanic_project/Decision_Tree.jpg" width="auto" height="450">  
[3]


Random forests are a good choice for several reasons:
- They are accurate
- They are less influenced by outliers
- They are less prone to overfitting by using enough trees
- They automatically perform feature selection, both ignoring less useful features and handling features with colinearity well. If two features are strongly correlated then the tree will pick the feature with the most information gain, and this in turn will decrease further information gain from the other. Linear models like linear regression or logistic regression can have varying solutions with colinear variables. Although we did not investigate this explicity it is likely some of our features are colinear. For example Age and titles (e.g. Mr) or fare price and passenger class. We will see later how our feature importances change.

Random forests have a few drawbacks however: 
- Although a single tree is easy to interpret, it is not so easy for a whole forest, particularly when using many trees
- They are time consuming since each decision tree has to be trained or run to make a prediction
  
```python
# Instantiate the classifier
clf = RandomForestClassifier()
clf.fit(x_train, y_train)

# Predictions of the training data
y_preds = clf.predict(x_train)

print(accuracy_score(y_preds, y_train))
```
```python
0.9854096520763187
```
This is a very high accuracy score but is to be expected for the training set. Here we have used the default parameters which actually perform reasonably well. 

We can visualise the feature importances of the model easily:
```python
# Gives the importance of different features of the model
importance = clf.feature_importances_

# Shortened columns to appear on one plot
columns = x_train.columns

# The importance of the different feautures according to the model
importance_dictionary = {columns[i] : importance[i] for i in range(len(importance)) }
importance

keys = importance_dictionary.keys()
values = importance_dictionary.values()

# Plotting the feature importance
plt.figure(figsize=(16, 6))
plt.bar(keys, values)
plt.xlabel('Features', size=14)
plt.ylabel('Feature Importances', size=14)
plt.xticks(rotation=45)
plt.show()
```
 <img src="/my-website/files/titanic_project/default_clf_feature_importances.png" width="auto" height="450"> 

We can see the feature selection at play with the random forest clearly prioritising some features over others. As expected, some of the most important features were those we investigated at the start. Age, Fare, Title_Mr and Sex are the most important. Most of the Decks are unimportant but Deck_U is more important than the rest which partially supports our theory that a missing deck may have been been caused by the death of a passenger. 

**How is feature importance calculated?** Whenever a decision is made by the tree, we work out how well the decision splits the data into survived and died. This is done using the Gini index which measures this impurity (splitting up efficacy) based on the probabilities of the outcomes from a split. For example a perfect decision will split the data into survived and died perfectly, however the worst decision will split the data into a perfect mix (50/50). The feature_importances measures how much, on average, each attribute contributes to decreasing the impurity - this is why it is called MDI (Mean Decrease in Impurity)

**Note:** I have since learnt that Sklearn's `feature_importances_` is biased towards features with high cardinality (number of possible values) such as the Age column. This is because the more splitting points we have, the higher the probability that a split could reduce the impurity. A more effective measure of feature importance is permutation importance which iteratively scrambles one of the features and measures the corresponding decrease in accuracy. However, this is more computationally expensive since each feature needs to be scrambled and the model needs to be scored several times. In future work we could implement this instead. 

We will go into scoring the test set later but for now let's see how this model performs with the test set. For reference:
- Submitting all dead gives a score of 0.622
- Submitting based on gender (predicting male = dead, female = survived) gives an accuracy score of 0.766

Originally I tested this classifier without the newly engineered columns and imputing the age with the median age. This gave a score of 0.768, only a slight improvement!
A lot of work for 0.02 increase in accuracy score. Scoring with the newly engineered columns gives 0.746, a disappointing and confusing score considering we can do better with just the gender submission. This might be down to overfitting, or even a bug. Let's perform some cross validation and hyperparameter tuning to find out if we can improve these scores. We still do not have a lot of features (at least compared to the dataset) and we know that random forests are quite good at performing feature selection. 

### 4. Cross Validation and Hyperparameter tuning

The hyperparameters of the network are like the dials on our models which we can tune to improve the learning process. An analogy is using a microwave to heat up your food: you can change the power of the microwave, the time period of heating, whether the food turns inside the microwave etc. The model will decide for itself all the decisions of the tree but we can tune this to make better decisions. For example, we could increase the number of trees in our forest, or the minimum number of passengers needed to perform a further split based on one of the features. 

Cross validation is a way to choose between these models but in a way that prevents the model from overperforming on the training set and underperforming on the test. It does this by splitting our training data into a smaller training set and 'validation' set. The model trains on the training set but is scored on the validation set. We split up the whole training set *k* number of times to do this, so that the validation set is different each time, and calculate an average of the accuracy score of those *k* iterations. This is called k-folds cross-validation and it is much easier to visualise like this: 

 <img src="/my-website/files/titanic_project/k-folds_cross_validation.png" width="auto" height="300"> 

 [4]

The advantage of cross validation is that we can pick which model from hyperparameter tuning is best whilst knowing it will not overfit the data since the accuracy is scored on unseen data. Normally the test set is meant to represent unseen data so we do not want to improve our models on it otherwise we risk overfitting. This project is unusual in that the test data is actually from the same distribution (i.e. the same ship).

Luckily Scikit-learn has some functions so that we can do this automatically with `GridSearchCV` and `RandomizedSearchCV`. `GridSearchCV` takes a dictionary of hyperparameters and exhaustively tries every single one to find the best cross validation score (which we will do with k-folds). `RandomizedSearchCV` also takes a dictionary of hyperparameters but randomly tries a pre-set number of different combinations. It has been shown that `RandomizedSearchCV` performs better with fewer iterations but that `GridSearchCV` will find the optimal hyperparameters given enough iterations [5]. 

``` python
# Setup the random seed 
np.random.seed(42)

# # Random forest model discluding all the features below
drop_columns_decks = ["Embarked", "Ticket", "Name", 'PassengerId']

# Print which columns we are including 
print(set(train_df.columns)-set(drop_columns_decks))

# Grid of hyperparameters to sample from
grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [None, 10, 20, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 10],
    'max_features': ['sqrt'],
    'bootstrap': [True, False]
}

# Creating dataframes discluding the titles
x_train_decks = prepare_dataframe(train_df, drop_columns_decks).drop(["Survived"], axis=1)
y_train = new_train_df["Survived"]
x_test_decks = prepare_dataframe(test_df, drop_columns_decks)

# Instantiating the Random Forest Classifier
clf = RandomForestClassifier(n_jobs = 1)

# Setting up randomised search of hyperparameters (considers 10 combinations) with cross validation
'''
rs_clf_decks = RandomizedSearchCV(estimator = clf, param_distributions=grid,
                       n_iter = 10, # number of models to try
                       cv = 5, # Setting the test set as the validation set
                       verbose =2 # Prints out information as it is running
                       )
'''

# Setting up exhaustive grid search of hyperparameters (considers 288 combinations) with cross validation
gs_clf_decks = GridSearchCV(estimator = clf, param_grid=grid,
                       cv = 5, # Setting the test set as the validation set
                       verbose =1 # Prints out information as it is running
                       )

# Fit the classifier
gs_clf_decks.fit(x_train_decks, y_train);
# Best parameters of the 10 iterations
best_params_decks = gs_clf_decks.best_params_
# Dataframe of the results of each hyperparameter combination
cv_results_decks = gs_clf_decks.cv_results_
```
This gives a trained model `gs_clf_decks` with the best hyperparameters. We will see some results shortly. 

Initially we ran some randomised searches to find good general hyperparameters and then searched the space exhaustively. 

These are our cross validated results: 

| Model | Default hyperparameters | Best hyperparameters |
| ------ | ----------- | ---|
| Age imputed with median | 0.802 | 0.816 |
| Age imputed with KNN    | 0.800 | 0.829 |
| Age imputed with KNN along with test data  | 0.805 | 0.826 |

As we can see the cross validated accuracy scores improves with using the KNN imputation technique. Imputing with the test set data appears to give very similar results to imputing with just the training set. However, when evaluating the model on the test set this is almost certainly going to give an advantage, as we will see later. As mentioned before, normally imputing with the test set data is poor practice but since we are making predictions on the same distribution (i.e. the same ship) we are fitting better to the required distribution. 

We can see the best hyperparameters, results for all the hyperparameter combinations and feature importance of the best performing classifier like so: 
```python
# Results of the cross validation
print(best_params_decks)
print(cv_results_decks["mean_test_score"].mean())

# Dataframe results of the 10 iterations
cv_results_decks_df = pd.DataFrame(cv_results_decks)
display(cv_results_decks_df)

importance = clf_decks.best_estimator_.feature_importances_

# Custom columns to fit in the plot (previous labels too long)
columns = x_train_decks.columns

# Feature importances according to the classifier
importance_dictionary = {columns[i] : importance[i] for i in range(len(importance)) }
importance

keys = importance_dictionary.keys()
values = importance_dictionary.values()

# Plot the feature importance
plt.figure(figsize=(17, 6))
plt.bar(keys, values)
plt.xlabel('Features', size=12)
plt.ylabel('Feature Importances', size=12)
plt.title('Feature importance', size=12)
plt.xticks(rotation=45)
plt.show()
```

 <img src="/my-website/files/titanic_project/best_hyperparameters.png" width="1010" height="auto">  

 <img src="/my-website/files/titanic_project/best_hyperparameter_feature_importances.png" width="auto" height="450">  

 As you can see in the top image we have the best hyperparameters in a dictionary and the results are shown in a dataframe. It turns out the best hyperparameters are pretty similar to the default hyperparameters, only two hyperparameters were changed minimally, but this gives quite an improvement. We are also given the accuracy score for the test set of each of the five folds. The highest score (0.854) is 5.55% more accurate than the lowest score (0.809)! The cross validation modules in scikit-learn are pretty useful.

It is also interesting to look at how the feature importances change by selecting the hyperparameters. For the default hyperparameters Age was the most important feature with a value of 0.21 but this decreases to 0.07 for the best hyperparameters and the Title_Mr becomes the most important feature with a value of 0.21. This shows how random forests can deal with colinear variables: they are good at selecting the most important features, and naturally other features encapsulating similar information (e.g. Age) become less important. The Fare and Sex which probably are not as strongly colinear to other variables remain important to both forests. 

### Metrics 

We can compare different metrics for our models as well. Accuracy is very important since it is the proportion of correct identifications but we can also look at precision or recall. Firstly, it is useful to understand the four categories our predictions can fall into:

 <img src="/my-website/files/titanic_project/ConfusionMatrixRedBlue.png" width="auto" height="200">

 [6]

Accuracy is given by 

$$ Accuracy = \frac{TP+TN}{TP+FP+TN+FN}$$

which is the most important metric for our project since the submission score uses accuracy. However, we may be interested in how well our models fair with different metrics. 

Two other important metrics in machine learning are precision and recall, which are important in different scenarios. Precision indicates the proportion of positive predictions which were actually correct:

$$Precision = \frac{TP}{TP+FP}$$

Precision is important when you want to minimise the number of false positives or maximise the number of true positives. For example, a Youtube recommendation algorithm would want to minimise the number of bad videos recommended (false positives) and maximise the recommendation of very good videos (true positives).

Recall on the other hand is defined as: 

$$Recall = \frac{TP}{TP+FN}$$

Recall is important when you want to minimise the number of false negatives or maximise the number of true negatives. For example, a rare cancer screening model would want to minimise the number of true cancer cases it categorises as negative (false negatives). Missing a cancer case could prove lethal for the patient. We would really want to be sure the patient does not have cancer if we were to predict negative. This is at the expense of increasing the number of false positives but the cost of life is higher than the cost of a second cancer screening.

Luckily for us, SciKit-Learn has an in-built function for calculating these metrics (along with some others). We cannot evaluate these metrics for the test set since we do not know their actual values. I wanted to use the validation sets from the GridSearchCV but I could not access the individual predictions, therefore I made my own cross-validation function:

```python
# Setup random seed
np.random.seed(42)

# Function to divide a dataframe into validation and training sets with cross validation (k-folds). Returns the desired fold  
def cross_val_index(k_folds, dataframe, fold_number):
    dataframe.sample(frac=1)
    x_dataframe = dataframe.drop(["Survived"], axis=1)
    y_dataframe = dataframe["Survived"]
    
    index = round(len(x_dataframe)/k_folds)
    start_index, end_index = [], []
    for i in range(k_folds):
                start_index.append(i * index)
                end_index.append((i + 1) * index if i < k_folds - 1 else len(x_dataframe))
    # print(start_index[fold_number-1], end_index[fold_number-1])
    X_train = pd.concat([x_dataframe[:start_index[fold_number-1]], x_dataframe[end_index[fold_number-1]:]])
    y_train = pd.concat([y_dataframe[:start_index[fold_number-1]], y_dataframe[end_index[fold_number-1]:]])
    X_valid = x_dataframe[start_index[fold_number-1]:end_index[fold_number-1]]
    y_valid = y_dataframe[start_index[fold_number-1]:end_index[fold_number-1]]
    
    return X_train, y_train, X_valid, y_valid
```
We can split up the data with as many ```k-folds``` as desired and can ask for any folds of the training and validation sets of a dataset with the parameter ```fold_number```. 

Then we can train the best model and worst model (from the random forest hyperparameter tuning) on the respective training set and predict on the validation set. In retrospect I should have compared the models in closer contention such as the default vs best hyperparameters or differing imputing methods. However comparing these models gives clearer differences in their metrics. 

```python
# Columns to drop
drop_columns = ["Embarked", "Ticket", "Name", "PassengerId"]

# Setting up validation dataframes
new_train_df = prepare_dataframe(train_df, drop_columns)

# Calling the function to get the folded dataframes
cross_val_dataframes = cross_val_index(5, new_train_df, 5)

# Calculating the probabilities of prediction with the best RandomForestClassifier
best_clf = RandomForestClassifier( **{'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt',
                                      'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100})
best_clf.fit(cross_val_dataframes[0], cross_val_dataframes[1])
best_y_valid_preds = best_clf.predict(cross_val_dataframes[2])
best_y_valids_proba = best_clf.predict_proba(cross_val_dataframes[2])
best_y_valids_proba_pos = best_y_valids_proba[:, 1]
# print(accuracy_score(best_y_valid_preds, cross_val_dataframes[3]))

# Calculating the probabilities of prediction with the worst RandomForestClassifier
worst_clf = RandomForestClassifier( **{'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 40, 'bootstrap': False})
worst_clf.fit(cross_val_dataframes[0], cross_val_dataframes[1])
worst_y_valid_preds = worst_clf.predict(cross_val_dataframes[2])
worst_y_valids_proba = worst_clf.predict_proba(cross_val_dataframes[2])
worst_y_valids_proba_pos = worst_y_valids_proba[:, 1]
# print(accuracy_score(worst_y_valid_preds, cross_val_dataframes[3]))

# print classification reports for the best and worst validation predictions
print(classification_report(best_y_valid_preds, cross_val_dataframes[3]))
print(classification_report(worst_y_valid_preds, cross_val_dataframes[3]))
```

```
Best model report: 
               precision    recall  f1-score   support

           0       0.90      0.88      0.89       117
           1       0.78      0.81      0.79        62

    accuracy                           0.85       179
   macro avg       0.84      0.84      0.84       179
weighted avg       0.86      0.85      0.86       179

Worst model report: 
               precision    recall  f1-score   support

           0       0.84      0.87      0.85       112
           1       0.77      0.73      0.75        67

    accuracy                           0.82       179
   macro avg       0.80      0.80      0.80       179
weighted avg       0.81      0.82      0.81       179
```
For now ignore the f1-score, macro avg and weighted avg. We will focus on the accuracy, precision and recall scores. Normally, we consider recall and precision for the positive case (predicting 1) but we can also calculate it for the negative case. Scikit-Learn's classification reports gives both. 

The accuracy for the best model is 0.85, 0.03 higher than the worst model with an accuracy of 0.82. I am unsure as to why these are higher than Scikit-Learn's own cross validation, I am shuffling the data. Let me know if you can see why.

The precision and recall are quite evenly matched, which is fine since we have no preference for one or the other for this project. However, the metrics are higher for the negative case (i.e. when a passenger died) than the positive case (i.e. when a passenger survived). This shows the model may be a little biased towards the negative cases (with higher observations) and treating the survived cases more as noise. Precision and recall are more important when there are higher benefits or costs to one of the classification types. 

### ROC curves and the AUC

Another way in which we can compare our models is by using a ROC (Receiver Operator Characteristic) Curve and calculating the AUC (area under the curve). 
These are ways in which we can visualise how the true positive rate and false negative rate are affected by changing the classification threshold of our predictions. 
 
When we pass a passenger's information through our models, the model returns a number between 0 and 1 which indicates how likely it thinks the passenger has survived. The classification threshold is a number which determines whether the model then predicts the passenger as surviving or not. For example, if the classification threshold is 0.5 then a passenger with a predicted probability of 0.4 would be predicted as having died and passenger with a predicted probability of 0.6 would be predicted as having survived. 

Ideally, we would like the true positive rate to be as high as possible and the false positive rate to be as low as possible. There is a trade off between these however, e.g. decreasing the classification threshold leads to all the actual positive values being predicted correctly but all the actual negative values being predicted incorrectly. An ROC curve shows the TP and FP rates for different classification thresholds and we can compare models this way:

 <img src="/my-website/files/titanic_project/ideal_roc_curve.png" width="auto" height="450">   

 [7]

The area under the curve (AUC) indicates the quality of the model and a perfect model will have a score of 1 while a random model will have a score of 0.5. Let's calculate our prediction probabilities and the area under the curve: 
```python
# Calculating the rates and thresholds for the two classifiers
best_fpr, best_tpr, best_thresholds = roc_curve(cross_val_dataframes[3].values, best_y_valids_proba_pos)
worst_fpr, worst_tpr, worst_thresholds = roc_curve(cross_val_dataframes[3].values, worst_y_valids_proba_pos)

print(roc_auc_score(cross_val_dataframes[3].values, best_y_valids_proba_pos))
print(roc_auc_score(cross_val_dataframes[3].values, worst_y_valids_proba_pos))
```
```python
0.9203125000000001
0.8805027173913044
```
The area under the curve for the best model is 0.92 and for the worst model it is 0.88. The true positive rates and false positive rates are calculated for various classification thresholds and we can plot these for each respective model:

```python
# Plotting the ROC curves for the best and worst classifiers 
plt.figure(figsize=(6,4))
plt.plot(best_fpr, best_tpr, color='orange', label='Best model')
plt.plot(worst_fpr, worst_tpr, color='green', label='Worst model')
plt.xlabel('False positive rate (fpr)', size=12)
plt.ylabel('True positive rate (tpr)', size=12)
plt.title('Receiver Operating Characteristic (ROC) curve')
plt.legend()
plt.show()
```
 <img src="/my-website/files/titanic_project/roc_curve_best_and_worst_rf_classifier.png" width="auto" height="450"> 

As we can see, the best model is almost always above the worst model, particularly for a false positive rate at around 0.2. Visualising the ROC and using the AUC as a metric is important for two reasons:
- They are scale-invariant: they measure how well predictions are ranked, rather than their absolute values
- They are classification-threshold-invariant: They measure the quality of a model's predictions irrespective of what classification is chosen.
It is good to see that our grid search is finding better hyperparameters and therefore better models.

### 5. Testing different classifiers and submitting predictions

Although I was happy with my results using the Random Forest Classifier, I also hyperparameter tuned a Logistic Regression classifier. Luckily for us, Scikit-Learn's models are called, tuned and fitted in very similar ways so it only required looking at which hyperparameters we wanted to tune. I will not go into the details of this but will show the process for hyperparameter tuning which is almost exactly the same as for the Random Forest:

```python
# Setup random seed 
np.random.seed(42)

# Discluding these features
drop_columns_decks = ["Embarked", "Ticket", "Name", "PassengerId"]

# Logistic regression hyperparameters to sample from 
log_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.5, 1, 2, 3],
    'solver': ['liblinear'],
    'class_weight': [None, 'balanced'],
    'max_iter': [2000, 4000],
    'tol': [0.0001, 0.001, 0.01],
    'multi_class': ['ovr']
}

# Creating dataframes
x_train_decks = prepare_dataframe(train_df, drop_columns_decks).drop(["Survived"], axis=1)
y_train = prepare_dataframe(train_df, drop_columns_decks)['Survived']
x_test_decks = prepare_dataframe(test_df, drop_columns_decks)

# Instantiating the Logistic regression classifier
log_clf = LogisticRegression()

# Setting up randomised search of hyperparameters (considers 10 combinations) with cross validation
gs_log_clf_decks = GridSearchCV(estimator = log_clf, param_grid=log_grid,
                       cv = 5, # Setting the test set as the validation set
                       verbose =1 # Prints out information as it is running
                       )
```
The main differences are the hyperparameters, which are completely different. It is a *generalised* linear model and uses an algorithm in its optimisation, in our case we have chosen 'liblinear' which works well for small datasets [8]. 

We found the optimal hyperparameters in exactly the same way with GridSearchCV and we were then able to submit some predictions!

### Submitting predictions

I initially used the Random Forest Classifier to make predictions and found that imputing the data altogether gave a better accuracy score: 
```python
# Dropping only unnecessary columns
drop_columns = ["Embarked", "Ticket", "Name", "PassengerId"]

# Preparing dataframes with combined imputing
new_train_df = prepare_dataframe(combined_df, drop_columns)
x_train = new_train_df.drop(["Survived"], axis=1)[:891]
y_train = new_train_df["Survived"][:891]
x_test = new_train_df.drop(["Survived"], axis=1)[891:]

# Instantiate Random Forest Classifier with best hyperparameters
best_clf = RandomForestClassifier( **{'bootstrap': True, 'max_depth': 10, 'max_features': 'sqrt',
                                      'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100})

# Fit the classifier and make predictions
best_clf.fit(x_train, y_train)
best_y_preds = best_clf.predict(x_test).astype(int)

# Create a csv file with the predictions
output = gender_submission.copy(deep=True)
output['Survived']=best_y_preds
print(output)
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
```
which returns 
```python
PassengerId  Survived
0            892         0
1            893         0
2            894         0
3            895         0
4            896         1
..           ...       ...
413         1305         0
414         1306         1
415         1307         0
416         1308         0
417         1309         1

[418 rows x 2 columns]
Your submission was successfully saved!
```
This saves the submission file to our ```/kaggle/working``` directory where we can download our submission file in csv format and submit on the Titanic Machine Learning page. We can change the hyperparameters used and also submit scores for our logistic regression model. After submitting the results of these models I received the following scores:

| Model | Default hyperparameters | Best hyperparameters |
| ------ | ----------- | ---|
| Random Forest, Age imputed with KNN along with test data  | 0.775 | 0.792 |
| Logistic Regression | 0.773 | 0.775 |


The highest score was 0.792, which I was happy with ranked me within the top 7.3% of submissions. 

There were several areas I could have explored to try to improve my score:
- Impute the missing ages with 0. A friend mentioned this to me after discussing the dataset and it could have informed the network about the missing data. Or even impute the data but keep a binary column indicating whether the value had been imputed or not.
- I did not use the ticket feature at all, perhaps this could have been useful after performing some feature engineering.
- I should have left the Deck column as the class categories and not one-hot encoded it since this only made the dataset more sparse.
- My workflow of the project could have been improved, I should have spent more time at the beginning performing the EDA since this would have saved time from not implementing features and scoring often.
- I could have tried selecting features and eliminating ones which were less important. Simple models are quicker and better (Occam's razor https://en.wikipedia.org/wiki/Occam%27s_razor)
- I have recently learnt that XGBoost is particularly effective for machine learning competitions. Using it may not have significantly my understanding but it would have improved my score!

If you have reached this point of my article then thank you so much for taking the time to read through it all! Do not hesitate to send me a message with questions, any thoughts and especially if you see something wrong! 

Some more resources for this projects:

I learnt how to use Sklearn using the Complete A.I. & Machine Learning, Data Science Bootcamp from Andrei Neagoie and Daniel Bourke on udemy https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/ . Shoutout to Daniel for his enthusiasm and making me passionate about machine learning. 
You can also check out some other Titanic dataset tutorials with Python:
- https://www.kaggle.com/competitions/titanic/code
- https://www.ahmedbesbes.com/blog/kaggle-titanic-competition
- https://www.ultravioletanalytics.com/blog/kaggle-titanic-competition-part-i-intro/
- The Sklearn documentation is fantastic and a great place to learn more about machine learning: https://scikit-learn.org/stable/

### References: 

[1] Wikimedia Commons contributors, "File:Stöwer Titanic.jpg," Wikimedia Commons, https://commons.wikimedia.org/w/index.php?title=File:St%C3%B6wer_Titanic.jpg&oldid=845574585 (accessed April 2, 2024)

[2] Effective Strategies for Handling Missing Values in Data Analysis by Nasima Tamboli, https://www.analyticsvidhya.com/blog/2021/10/handling-missing-value/

[3] Wikimedia Commons contributors, "File:Decision Tree - survival of passengers on the Titanic.jpg," Wikimedia Commons, https://commons.wikimedia.org/w/index.php?title=File:Decision_Tree_-_survival_of_passengers_on_the_Titanic.jpg&oldid=469537604 (accessed April 2, 2024).

[4] Wikimedia Commons contributors, "File:K-fold cross validation EN.svg," Wikimedia Commons, https://commons.wikimedia.org/w/index.php?title=File:K-fold_cross_validation_EN.svg&oldid=852533300 (accessed April 2, 2024). Image edited by me. 

[5] Intro to Model Tuning: Grid and Random Search by Will Koerhsen, https://www.kaggle.com/code/willkoehrsen/intro-to-model-tuning-grid-and-random-search

[6] Wikimedia Commons contributors, "File:ConfusionMatrixRedBlue.png," Wikimedia Commons, https://commons.wikimedia.org/w/index.php?title=File:ConfusionMatrixRedBlue.png&oldid=505414509 (accessed April 2, 2024).

[7] Wikimedia Commons contributors, "File:Roc curve.svg," Wikimedia Commons, https://commons.wikimedia.org/w/index.php?title=File:Roc_curve.svg&oldid=790154745 (accessed April 2, 2024).

[8] Sklearn's Logistic Regression classifier, https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html



