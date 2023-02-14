#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline
import seaborn as sns

#Model Export
import joblib

import dataset
df = pd.read_csv("D://Users//xkllkx//Desktop//all_program//NCU_ML//titanic_survive_predict//train_data_titanic.csv")


#observing dataset
#df.head()
#df.describe().T
#df.info()

#畫分布圖
# sns.distplot(df['Price'])
# sns.jointplot(df['Avg. Area Income'],df['Price'])
# sns.pairplot(df)

#Remove the columns model will not use
#不要'Name','Ticket'
df.drop(['Name','Ticket'],axis=1,inplace=True)
df.head()
df.info()
#sns.pairplot(df[['Survived','Fare']], dropna=True)

#data observing
df.groupby('Survived').mean()

#data observing
df['SibSp'].value_counts()
df['Parch'].value_counts()
df['Sex'].value_counts()

#Handle missing values
df.isnull().sum()
len(df)
len(df)/2
df.isnull().sum()>(len(df)/2)

#Cabin has too many missing values
df.drop('Cabin',axis=1,inplace=True)
df.head()
df['Age'].isnull().value_counts()
df['Age'].isnull().sum()
#Age is also have some missing values
df.groupby('Sex')['Age'].median().plot(kind='bar')
#缺失值男生就用男生的中位數(29)、女生就用女生的中位數(27)來填補
df['Age'] = df.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.median()))

df.isnull().sum()
#發現還有Embarked還有缺2個
df['Embarked'].value_counts()
#找出第一個次數最多的，發現是S
df['Embarked'].value_counts().idxmax()
df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(),inplace=True)
df['Embarked'].value_counts()

df.isnull().sum()

#將Sex, Embarked進行轉換
#Sex轉換成是否爲男生、是否爲女生，Embarked轉換爲是否爲S、是否爲C、是否爲Q
df = pd.get_dummies(data=df, columns=['Sex','Embarked'])
df.head()
#是否爲男生與是否爲女生只要留一個就好，留下是否爲男生
df.drop(['Sex_female'], axis=1, inplace=True)
df.head()

df.corr()

#Prepare training data

#把Survived, Pclass丟掉
X = df.drop(['Survived','Pclass'],axis=1)
y = df['Survived']

#split to training data & testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)

from sklearn.ensemble import GradientBoostingClassifier
lr = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
lr.fit(X, y)
predictions = lr.predict(X_test)
predictions

#Evaluate
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
print(accuracy_score(y_test, predictions))
print(recall_score(y_test, predictions))
print(precision_score(y_test, predictions))
pd.DataFrame(confusion_matrix(y_test, predictions), columns=['Predict not Survived',
'Predict Survived'], index=['True not Survived', 'True Survived'])

#Model Export
joblib.dump(lr,'titanic_survive_predict_20220317.pkl',compress=3)