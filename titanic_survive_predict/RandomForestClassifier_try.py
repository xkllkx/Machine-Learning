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
df_test = pd.read_csv("D://Users//xkllkx//Desktop//all_program//NCU_ML//titanic_survive_predict//test.csv")

#observing dataset
#df.head()
#df.describe().T
#df.info()

#畫分布圖
# sns.distplot(df['Price'])
# sns.jointplot(df['Avg. Area Income'],df['Price'])
# sns.pairplot(df)

#Remove the columns model will not use



df['Name_Title'] = df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
df['Name_Title'] = df_test['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
df['Name_Title'].value_counts()

# df['Survived'].groupby(df['Name_Title']).mean()
df = pd.get_dummies(data=df, columns=['Name_Title'])


df['Ticket_Len'] = df['Ticket'].apply(lambda x: len(x))
df['Ticket_Len'] = df_test['Ticket'].apply(lambda x: len(x))
df['Ticket_Len'].value_counts()
# df.groupby(['Ticket_Len'])['Survived'].mean()
df = pd.get_dummies(data=df, columns=['Ticket_Len'])

df['Ticket_Lett'] = df['Ticket'].apply(lambda x: str(x)[0])
df['Ticket_Lett'] = df_test['Ticket'].apply(lambda x: str(x)[0])
df['Ticket_Lett'].value_counts()
# df.groupby(['Ticket_Lett'])['Survived'].mean()
df = pd.get_dummies(data=df, columns=['Ticket_Lett'])



#不要'Name','Ticket'
df.drop(['Name','Ticket'],axis=1,inplace=True)

df.info


# df.head()
# df.info()
#sns.pairplot(df[['Survived','Fare']], dropna=True)

#data observing
df.groupby('Survived').mean()

#data observing
df['SibSp'].value_counts()
df['Parch'].value_counts()
df['Sex'].value_counts()

#Handle missing values 確認空格數
df.isnull().sum()
len(df)
len(df)/2
df.isnull().sum()>(len(df)/2)

#Cabin has too many missing values
# df.drop('Cabin',axis=1,inplace=True)

df['Cabin'] = pd.Series(['X' if pd.isnull(ii) else ii[0] for ii in df['Cabin']])
df_test['Cabin'] = pd.Series(['X' if pd.isnull(ii) else ii[0] for ii in df_test['Cabin']])

plt.figure(figsize=(12,5))
plt.title('Box Plot of Temperatures by Modules')
sns.boxplot(x='Cabin',y='Fare',data=df, palette='Set2')
plt.tight_layout()

print('Mean Fare of Cabin B {}'.format(df[df['Cabin']=='B']['Fare'].mean()))
print('Mean Fare of Cabin C {}'.format(df[df['Cabin']=='C']['Fare'].mean()))
print('Mean Fare of Cabin D {}'.format(df[df['Cabin']=='D']['Fare'].mean()))
print('Mean Fare of Cabin E {}'.format(df[df['Cabin']=='E']['Fare'].mean()))

def reasign_cabin(cabin_fare):
    
    cabin = cabin_fare[0]
    fare = cabin_fare[1]
    
    if cabin=='X':
        if (fare >= 113.5):
            return 'B'
        if ((fare < 113.5) and (fare > 100)):
            return 'C'
        if ((fare < 100) and (fare > 57)):
            return 'D'
        if ((fare < 57) and (fare > 46)):
            return 'D'
        else:
            return 'X'
    else:
        return cabin

df['Cabin'] = df[['Cabin', 'Fare']].apply(reasign_cabin, axis=1)
df_test['Cabin'] = df_test[['Cabin', 'Fare']].apply(reasign_cabin, axis=1)

plt.figure(figsize=(12,5))
plt.title('Box Plot of Temperatures by Modules')
sns.boxplot(x='Cabin',y='Fare',data=df, palette='Set2')
plt.tight_layout()

df = pd.get_dummies(data=df, columns=['Cabin'])



#df.head()
df['Age'].isnull().value_counts()
df['Age'].isnull().sum()


#Age is also have some missing values

#df.groupby('Sex')['Age'].median().plot(kind='bar')

#試圖用其他種方法填補Age

#1.Sex中位數填補
#缺失值男生就用男生的中位數(29)、女生就用女生的中位數(27)來填補
# df['Age'] = df.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.median()))


#2.Pclass平均數填補
# df.groupby(['Pclass'])['Age'].mean()
# # print(df.groupby(['Pclass'])['Age'].mean()[1])
# df['Age'] = df.groupby(['Pclass'])['Age'].apply(lambda x: x.fillna(x.mean()))

#3.Pclass中位數填補
# df.groupby(['Pclass'])['Age'].median()
# df['Age'] = df.groupby(['Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

#4.Sex平均數填補
df.groupby(['Sex'])['Age'].mean()
df['Age'] = df.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.mean()))

#5.Sex、Pclass年齡平均數填補
# table = df.pivot_table(values='Age',index='Pclass',columns='Sex',aggfunc=np.mean)
# def fage(x):
#     return table.loc[x['Pclass'],x['Sex']]
# print(table)
# df['Age'].fillna(df.apply(fage, axis=1),inplace=True)


#df.isnull().sum()
#發現還有Embarked還有缺2個
df['Embarked'].value_counts()
#找出第一個次數最多的，發現是S
df['Embarked'].value_counts().idxmax()
df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(),inplace=True)
df['Embarked'].value_counts()

#df.isnull().sum()

#將Sex, Embarked進行轉換
#Sex轉換成是否爲男生、是否爲女生，Embarked轉換爲是否爲S、是否爲C、是否爲Q
df = pd.get_dummies(data=df, columns=['Sex','Embarked'])
# df.head()
#是否爲男生與是否爲女生只要留一個就好，留下是否爲男生
df.drop(['Sex_female'], axis=1, inplace=True)
# df.head()

df.corr()

#Prepare training data

#把Survived, Pclass丟掉
#X = df.drop(['Survived','Pclass'],axis=1)
X = df.drop(['Survived'],axis=1)
print(X)
y = df['Survived']

for i in range(1):
    print(f'---第{i+1}次測試---')
    #split to training data & testing data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=67)

    from sklearn.ensemble import RandomForestClassifier
    # lr = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    lr = RandomForestClassifier(criterion='gini', 
                            n_estimators=700,
                            min_samples_split=10,
                            min_samples_leaf=1,
                            max_features='auto',
                            oob_score=True,
                            random_state=1,
                            n_jobs=-1)
    lr.fit(X, y)
    predictions = lr.predict(X_test)
    # predictions

    #Evaluate
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
    print(accuracy_score(y_test, predictions))
    print(recall_score(y_test, predictions))
    print(precision_score(y_test, predictions))
    pd.DataFrame(confusion_matrix(y_test, predictions), columns=['Predict not Survived',
    'Predict Survived'], index=['True not Survived', 'True Survived'])

#Model Export
# joblib.dump(lr,'titanic_survive_predict_20220317.pkl',compress=3)

pd.concat((pd.DataFrame(df.iloc[:, 1:].columns, columns = ['variable']), 
        pd.DataFrame(lr.feature_importances_, columns = ['importance'])), 
        axis = 1).sort_values(by='importance', ascending = False)[:20]