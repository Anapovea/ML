# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O
#The Machine learning alogorithm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression

print("TITANIC ML MODEL")
    
# Input data files are available in the "../input/" directory.
# This function will list all files under the input directory
# Any results you write to the current directory are saved as output.
def available_files():
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

def model_solution():
    train = pd.read_csv('/kaggle/input/titanic/train.csv')
    
    #The algorithms in Sklearn (the library we are using), does not work missing values, so lets first check the data for missing values. Also, they work only work with numbers.
    train.isnull().sum()

    #print(train.head(3)) #show the first 5 rows of the data.
    #print(train.describe()) # command to get to know the data in a summarized way.
    
    # In this case ‘Survived’ Column is output column and rest all are input columns.
    # Selecting only 2 columns for ease
    train_x = train[["Pclass", "Sex"]]
    train_x.head()
    
    # Selecting the output / target variable
    train_y = train[["Survived"]]
    train_y.head()
    
    # Making Male/ Female to integer numbers called Label Encoding
    train_x["Sex"].replace("male", 1, inplace = True)
    train_x["Sex"].replace("female", 0, inplace = True)
    train_x.head()
    
    # We will use the train_test_split function to create the test/ train 
    # (cross-validation) split. We will use 70% of the data to train and 
    # model and 30% of the data to check accuracy.
    # Making dataset for validation
    # tr_x & tr_y are the training input and output 
    # cv_x & cv_y are cross-validation input and output.
    tr_x, cv_x, tr_y, cv_y   = train_test_split(train_x, train_y, test_size = 0.30)
    tr_x.head()
    tr_y.head()
    
    
    # Call the Machine Learning Algorithm
    rf = RandomForestClassifier()
    
    # Fitting and training the above called algorithm
    rf.fit(tr_x, tr_y)
    
    Accuracy_RandomForest = rf.score(cv_x, cv_y)
    print("Accuracy = {}%".format(Accuracy_RandomForest * 100))
    
    lgr = LogisticRegression()
    lgr.fit(tr_x, tr_y)
    
    Accuracy_LogisticRegression = lgr.score(cv_x, cv_y)
    print("Accuracy2 = {}%".format(Accuracy_LogisticRegression * 100))
    
    
    #Predict the data
    test = pd.read_csv('/kaggle/input/titanic/test.csv')    
    test.head()
    test_x = test[["Pclass", "Sex"]]
    test_x.head()
    test_x["Sex"].replace("male", 1, inplace = True)
    test_x["Sex"].replace("female", 0, inplace = True)
    test_x.head()
    prd = rf.predict(test_x)
    print(prd)
    # Check the format of the Submission file, there are 2 columns PassengerId and Survived. So lets convert our predicted output in the same format.
    op = test[['PassengerId']]
    op['Survived'] = prd
    print(op.head())
    op.to_csv("submission.csv",index=False)
    print("Your submission was successfully saved!")
#####################################################################    
model_solution()

TITANIC ML MODEL

/tmp/ipykernel_20/2989979964.py:42: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  train_x["Sex"].replace("male", 1, inplace = True)
/tmp/ipykernel_20/2989979964.py:43: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  train_x["Sex"].replace("female", 0, inplace = True)
/tmp/ipykernel_20/2989979964.py:61: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  rf.fit(tr_x, tr_y)

Accuracy = 81.34328358208955%
Accuracy2 = 81.34328358208955%
[0 1 0 0 1 0 1 0 1 0 0 0 1 0 1 1 0 0 1 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 0 1
 1 0 0 0 0 0 1 1 0 0 0 1 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 0 1 1 0 0 1 1 0 1 0
 1 0 0 1 0 1 0 0 0 0 0 0 1 1 1 0 1 0 1 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0
 1 1 1 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0
 0 0 1 0 0 1 0 0 1 1 0 1 1 0 1 0 0 1 0 0 1 1 0 0 0 0 0 1 1 0 1 1 0 0 1 0 1
 0 1 0 1 0 0 0 0 0 0 0 0 1 0 1 1 0 0 1 0 0 1 0 1 0 0 0 0 1 1 0 1 0 1 0 1 0
 1 0 1 1 0 1 0 0 0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 0 1
 0 0 0 1 1 0 0 0 0 1 0 0 0 1 1 0 1 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0
 1 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0
 1 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 1 0 0 0 1 0 1 0 0 1 0 1 1 0 1 1 0 1 1 0
 0 1 0 0 1 1 1 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0
 0 1 1 1 1 1 0 1 0 0 0]
   PassengerId  Survived
0          892         0
1          893         1
2          894         0
3          895         0
4          896         1
Your submission was successfully saved!

/opt/conda/lib/python3.10/site-packages/sklearn/utils/validation.py:1143: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
/tmp/ipykernel_20/2989979964.py:78: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  test_x["Sex"].replace("male", 1, inplace = True)
/tmp/ipykernel_20/2989979964.py:79: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  test_x["Sex"].replace("female", 0, inplace = True)
/tmp/ipykernel_20/2989979964.py:85: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  op['Survived'] = prd

