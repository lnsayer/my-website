# First Machine Learning Project: Classification With The Titanic Dataset

<img src="/files/titanic_project/Stöwer_Titanic.jpg" width="500" height="350">

This was my first end-to-end machine learning project on the famous Titanic dataset. It is a classification problem in which we predict which passengers survived the sinking of the Titanic. I have implemented several machine learning techniques during this project and gained a respectable position in the leaderboard, within the top 7% of users. This project is particularly interesting because it demonstrates the biases which machine learning models can exhibit.

This notebook also serves as a guide for people new to machine learning. Please do not hesitate to get in contact if you have any questions, or even more so if you spot anything incorrect!

### The plan:
1. Data exploration and visualisation
2. Data cleaning, feature selection and engineering
3. Quick Modelling to see feature importance
4. Cross Validation and Hyperparameter tuning
5. Testing different classifiers

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
<img src="/files/titanic_project/combined_df.png" width="844" height="350"> 

We've got 12 columns with 10 usable features. 
- PassengerId is an index
- The Survived is 1 if they survived or 0 if they died
- The PClass columnn shows the passenger class (1st, 2nd or 3rd)
- The name column gives their full name and title
- Sex is either male or female
- Age in years
- SibSp is number of siblings/spouses on the ship
- Parch is number of parents/children on the ship
- Ticket shows their ticket code
- Fare in dollars
- Cabin, with deck classification
- Embarked gives their port of departure: S: Southampton, C: Cherbourg, Q: Queenstown

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
We mostly have integers and strings, with some objects which are strings. We are missing data in the Age (263), Cabin (1014), Fare (1) and Embarked (2) columns. We will have to do some imputation (filling in missing values) and encoding (converting the strings to numerical data which the models can handle).

Since we have combined the training and test set into one dataframe we can see all the missing data, e.g. if we had only looked at the training set we would miss the missing fare entry in the test set. Normally we want to impute the training and test set separately since our test set represents future data we want to predict. However, as this is the same ship with the same distribution it is okay to impute altogether. It also makes our life easier to do so. 

Let's explore how different features might affect survival rates. We can display the survival rate based on age and sex at the same time. 
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

<img src="/files/titanic_project/survival_rate_age_sex.png" width="auto" height="450"> 

These are quite stark results with the men much more likely to die than women. Age does not appear to affect survival rates amongst adults, however children are more likely to survive than die. The famous unofficial code of conduct 'Women and children first" is confirmed in use by these plots.

We can also plot the survival rates according to fare price:

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
<img src="/files/titanic_project/survival_rate_fare_price.png" width="auto" height="450">  

The tail of the survivors is longer showing that people who paid a higher fare were more likely to survive. However, this is not strict, some passengers who bought an expensive ticket also died. 

Finally let's look at the survival rates amongst different passenger classes. 

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
<img src="/files/titanic_project/survival_rate_passenger_class.png" width="auto" height="450">  

Again, better passenger class corresponds to a higher chance of survival. On the Titanic, you can put a price on life.

### 2. Data cleaning, feature selection and engineering

Now let's try and make some other features useful and even create a new feature. 
First we will extract all the titles of the passengers. These might be useful since it can explicity tell the model the age and sex of a passenger, as well as their marital status. This code makes a new column 'Title' which returns the string following a comma and space and then a full stop e.g. for `, Mr.` it returns `Mr`. We can see how many entries have each title with `value_counts`. 

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
We can see there are several unusual titles, let's transfer them to some of the more common titles with this function which can appply it to any dataframe:

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
In hindsight we could have considered putting these in a special column but for now we will leave it. Next we are going to extract the deck of their cabin with another function: 
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
We will fill empty cabin entries with `U`. Perhaps some areas of the boat were able to evacuate more easily than others. The boat hit the iceberg at 11:40pm so it is likely the passengers were inside their cabins at this time. 

Finally we will make a family_size attribute which takes gives the family size of each passenger:
```python
# Function to define a person's family size
def add_family(x):
    x['Familysize'] = x['SibSp']+x['Parch']+1
    return x
```
Let's check all these are working with our training dataset: 
```python
# Show the new altered dataframes with 'Title', 'Deck' and 'Familysize' columns
new_train_df['Title']=new_train_df.apply(replace_titles, axis=1)
new_train_df= replace_cabin(new_train_df)
new_train_df = add_family(new_train_df)

new_train_df
```
 <img src="/files/titanic_project/new_features.png" width="auto" height="400">   

 Next we are going to impute our data and one hot encode categorical data. For this I created a function `prepare_dataframe` which can do this to any dataframe we like and can also drop columns we are not interested in using. It does the following:
 - Converts the sex column into binary (0 and 1s)
 - Replace the Name column with Title
 - Replaces the Cabin column with Deck
 - Add family size
 - Imputes Fare with the median value
 - One hot encodes Embarked, Title, Deck and Pclass
 - Imputes the Age column using a KNN algorithm
 - Renames engineered columns
 - Drop any columns which are requested

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
# display(new_train_df)
print(new_train_df.columns)
new_train_df['Age'].isna().sum()
new_train_df
```
 <img src="/files/titanic_project/prepared_dataset.png" width="auto" height="300"> 
Some notes:

- It is not immediately clear of the cause for the missing age values. This is something we could have investigated further. We do not want to remove this column as it is such a useful predictor. Imputing using the median age would not give the best estimate for age since there are other variables which we can use to predict the missing values such as their title or number of siblings (children are more likely to travel with siblings than adults are). For this reason I decided to use a k-nearest neighbour algorithm because it is accurate, simple and fast. Imputing the missing ages gave a similar distribution compared to the original non-missing data: 

```python
# Columns to drop in preparing the dataframes
drop_columns = ["Embarked", "Ticket", "Name", 'PassengerId']

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

 <img src="/files/titanic_project/imputed_age.png" width="auto" height="450">  

 - Another option for imputing the age would have been to set the missing values as 0. It might have been the case that passengers had a missing age *because* they had died and their records were not kept. Setting the missing values as 0 might have been able to capture this. Further work could explore the effect of this. 

- For the deck code we decided to keep the missing values as a new column because like age the missing values might have been caused a passenger dying. We will see this column turned out to be pretty important in the models' feature importances.
- Since there is only one missing feature from the Fare column we imputed this with the median.
- We decided to one-hot encode the Pclass column but in fact this was unnecessary and only made the dataframe more sparse. We could have kept this in its original form. 

### 3. Quick Modelling to see feature importance

We can quickly model the data using a Random forest classifier. Random forests are a good choice for several reasons:
- They are accurate
- They are less influenced by outliers
- Less prone to overfitting by using enough trees
- Automatically perform feature selection, both ignoring less useful features and handling features with colinearity well. If two features are strongly correlated then the tree will pick the feature with the most information gain, and this in turn will decrease further information gain from the other. Linear models like linear regression or logistic regression can have varying solutions with colinear variables. Although we did not investigate this explicity it is likely some of our features are colinear. For example Age and titles (e.g. Mr) or fare price and passenger class. We will see later how our feature importances change.

Random forests have a few drawbacks however: 

  
```python
# Setup the random seed
np.random.seed(42)

# Prepare dataframes
new_train_df = prepare_dataframe(train_df, drop_columns)
new_test_df = prepare_dataframe(test_df, drop_columns)

# First RandomForestClassifier 

# Split up into feature variables and target variables
x_train = new_train_df.drop(["Survived"], axis=1)
y_train = new_train_df["Survived"]
x_test = new_test_df

# Instantiate the classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train, y_train)

# Predictions of the training data
y_preds = clf.predict(x_train)

print(accuracy_score(y_preds, y_train))
print(clf.get_params())
```

