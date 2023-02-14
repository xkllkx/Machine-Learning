#引入模型
import joblib
import numpy as np
model_pretrained = joblib.load('titanic_survive_predict_20220317.pkl')

#拿到題目卷
import pandas as pd
import dataset

df = pd.read_csv("D://Users//xkllkx//Desktop//all_program//NCU_ML//titanic_survive_predict//train_data_titanic.csv")
df_test = pd.read_csv("D://Users//xkllkx//Desktop//all_program//NCU_ML//titanic_survive_predict//test.csv")

#----------------------------------------------------------------

df_test['Name_Title'] = df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
df_test['Name_Title'] = df_test['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

df_test = pd.get_dummies(data=df_test, columns=['Name_Title'])


df_test['Ticket_Len'] = df['Ticket'].apply(lambda x: len(x))
df_test['Ticket_Len'] = df_test['Ticket'].apply(lambda x: len(x))

df_test = pd.get_dummies(data=df_test, columns=['Ticket_Len'])

df_test['Ticket_Lett'] = df['Ticket'].apply(lambda x: str(x)[0])
df_test['Ticket_Lett'] = df_test['Ticket'].apply(lambda x: str(x)[0])

df_test = pd.get_dummies(data=df_test, columns=['Ticket_Lett'])


#不要'Name','Ticket'
df_test.drop(['Name','Ticket'],axis=1,inplace=True)

#----------------------------------------------------------------

# df_test.drop('Cabin',axis=1,inplace=True)

df['Cabin'] = pd.Series(['X' if pd.isnull(ii) else ii[0] for ii in df['Cabin']])
df_test['Cabin'] = pd.Series(['X' if pd.isnull(ii) else ii[0] for ii in df_test['Cabin']])

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

df_test['Cabin'] = df[['Cabin', 'Fare']].apply(reasign_cabin, axis=1)
df_test['Cabin'] = df_test[['Cabin', 'Fare']].apply(reasign_cabin, axis=1)

df_test = pd.get_dummies(data=df_test, columns=['Cabin'])



#1.Sex中位數填補
#缺失值男生就用男生的中位數(29)、女生就用女生的中位數(27)來填補
# df_test['Age'] = df_test.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.median()))


#2.Pclass平均數填補
# df_test.groupby(['Pclass'])['Age'].mean()
# # print(df_test.groupby(['Pclass'])['Age'].mean()[1])
# df_test['Age'] = df_test.groupby(['Pclass'])['Age'].apply(lambda x: x.fillna(x.mean()))

#3.Pclass中位數填補
# df_test.groupby(['Pclass'])['Age'].median()
# df_test['Age'] = df_test.groupby(['Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

#4.Sex平均數填補
df_test.groupby(['Sex'])['Age'].mean()
df_test['Age'] = df_test.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.mean()))

#5.Sex、Pclass年齡平均數填補
# table = df_test.pivot_table(values='Age',index='Pclass',columns='Sex',aggfunc=np.mean)
# def fage(x):
#     return table.loc[x['Pclass'],x['Sex']]
# print(table)
# df_test['Age'].fillna(df_test.apply(fage, axis=1),inplace=True)

df_test.isnull().sum()
df_test['Fare'].value_counts() #Fare有缺 
df_test['Fare'].fillna(df['Fare'].value_counts().idxmax(),inplace=True) #補值

df_test = pd.get_dummies(data=df_test, columns=['Sex','Embarked'])
df_test.drop('Sex_female',axis=1,inplace=True)
#df_test.drop('Pclass',axis=1,inplace=True)

predictions2 = model_pretrained.predict(df_test)
predictions2

#Preare submit file
forSubmissionDF = pd.DataFrame(columns=['PassengerId','Survived'])
forSubmissionDF['PassengerId'] = range(892,1310)
forSubmissionDF['Survived'] = predictions2
forSubmissionDF

#Dataframe to CSV
forSubmissionDF.to_csv('for_submission_20220317.csv', index=False)
