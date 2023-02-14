#Basic Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Preprocessing Module
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
#ML model Module
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#遺漏值確認
train.isnull().sum()
test.isnull().sum()
train.isnull().values.any() #有缺就True
test.isnull().values.any() #有缺就True

#簡介
train.info()
test.info()

#移除受測者的id
train.drop('subject',axis=1, inplace=True)
test.drop('subject',axis=1, inplace=True)

#將欄位名稱另存一個List
rem_cols2 = train.columns.tolist()

type(train["Activity"][0])

#找到object
is_object_type_feature = train.dtypes == object
print(is_object_type_feature)
train.columns[is_object_type_feature]

#Activity的數值分布
train['Activity'].value_counts() #or train.Activity.value_counts()
test['Activity'].value_counts()

#將Activity轉換成數值型式
le = LabelEncoder()
for x in [train, test]:
    x['Activity'] = le.fit_transform(x['Activity'])

train['Activity'].value_counts()

#關聯性觀察
corr_val = train.corr()
corr_val_activity_abs = corr_val['Activity'].abs()
corr_val_activity_abs_sort = corr_val_activity_abs.sort_values(ascending=False)
corr_val_activity_abs_sort[corr_val_activity_abs_sort>0.84]

#切資料
feature_cols = train.columns[:-1] #不包含Activity

split_data = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
train_idx, val_idx = next(split_data.split(train[feature_cols],train.Activity)) #切訓練資料與測試資料

X_train = train.loc[train_idx, feature_cols]
y_train = train.loc[train_idx, 'Activity']

X_val = train.loc[val_idx, feature_cols]
y_val = train.loc[val_idx, 'Activity']

y_train.value_counts()
y_train.value_counts(normalize=True)
y_val.value_counts(normalize=True)

#分析
lr = LogisticRegression(max_iter=1500)
dt = DecisionTreeClassifier(max_depth=5)
rf = RandomForestClassifier()

models = [lr, dt, rf]
heatmaps = {}

score_board = pd.DataFrame(columns=['Accuracy','Precision','Recall','F1'])

def train_predict_evaluate(this_model, this_model_name):
    this_model_trained = this_model.fit(X_train, y_train)
    this_model_predict = this_model_trained.predict(X_val)
    this_model_accuracy_score = accuracy_score(y_val, this_model_predict)
    this_model_precision_score = precision_score(y_val, this_model_predict, average='weighted')
    this_model_recall_score = recall_score(y_val, this_model_predict, average='weighted')
    this_model_F1_score = f1_score(y_val, this_model_predict, average='weighted')
    score_board.loc[this_model_name]= {'Accuracy':this_model_accuracy_score,'Precision':this_model_precision_score,'Recall':this_model_recall_score,'F1':this_model_F1_score}
    heatmaps[this_model_name] = sns.heatmap(confusion_matrix(y_val,this_model_predict),annot=True,fmt='d')

# Logistic Regression
train_predict_evaluate(lr,'lr')
# Decision Tree Classifier
train_predict_evaluate(dt,'dt')
# Random Forest Classifier
train_predict_evaluate(rf,'rf')

score_board

















