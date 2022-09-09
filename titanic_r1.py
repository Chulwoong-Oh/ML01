#!/usr/bin/env python
# coding: utf-8

# In[226]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'last' #all last

import pandas as pd
import numpy as np
import os, sys, random
sys.path.append('F:\my_documnet_F')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.cm as cm
from importlib import reload
plt=reload(plt)
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import datetime
import seaborn as sns
from scipy import sparse
# import mglearn

import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import lightgbm as lgb

#Auto reloads notebook when changes are made
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.ticker as ticker

import imp
import Library.Rawdata_stack as rawstack
imp.reload(rawstack)


# In[150]:


path = './data/'


# In[151]:


df_train = pd.read_csv(path + 'train.csv')
df_test = pd.read_csv(path + 'test.csv')
df_submission = pd.read_csv(path + 'submission.csv')


# In[152]:


np.random.seed(1234)
random.seed(1234)


# In[153]:


df_embarked = df_train[['Embarked', 'Survived', 'PassengerId']].dropna().groupby(['Embarked', 'Survived']).count().unstack()
df_embarked.plot.bar(stacked = True)


# In[154]:


df_sex = df_train[['Sex', 'Survived', 'PassengerId']].dropna().groupby(['Sex', 'Survived']).count().unstack()
df_sex.plot.bar(stacked = True)


# In[155]:


df_class = df_train[['Pclass', 'Survived', 'PassengerId']].dropna().groupby(['Pclass', 'Survived']).count().unstack()
df_class.plot.bar(stacked = True)


# In[156]:


plt.hist(x=[df_train['Age'][df_train.Survived == 0], df_train['Age'][df_train.Survived == 1]], bins = 8, histtype = 'barstacked', label = ['Death', 'Survived'])
plt.legend()


# In[157]:


df_train_corr = pd.get_dummies(df_train, columns = ['Sex'], drop_first = True)
df_train_corr = pd.get_dummies(df_train_corr, columns = ['Embarked'])


# In[158]:


train_corr = df_train_corr.corr()


# In[159]:


plt.figure(figsize = (9, 9))
sns.heatmap(train_corr, vmax = 1, vmin = -1, center = 0, annot=True, cmap ='RdBu')


# In[198]:


df_all = pd.concat([df_train, df_test], sort=False).reset_index(drop=True)
df_all.loc[df_all.Fare.isnull(), 'Fare'] = df_all[['Pclass', 'Fare']].groupby('Pclass').transform(np.mean)
df_all1 = pd.concat([df_all, df_all['Name'].str.split("[.,]", 2, expand = True).rename(columns = {0:'family_name', 1:'honorific', 2:'name'})], axis = 1)
df_all1.loc[df_all1.Age.isnull(), 'Age'] = df_all1[['honorific', 'Age']].groupby('honorific').transform(np.mean)

df_all1['honorific'] = df_all1['honorific'].str.strip()


# In[199]:


plt.figure(figsize=(18, 5))
sns.boxplot(x='honorific', y='Age', data = df_all1)


# In[200]:


df_train1 = df_all1.iloc[:len(df_train), :]
df_test1 = df_all1.iloc[len(df_train):, :]


# In[201]:


df_all1['family_num'] = df_all.Parch + df_all.SibSp

df_all1.loc[df_all1.family_num ==0, 'alone'] = 1
df_all1.alone.fillna(0, inplace = True)


# In[202]:


df_train1[['honorific', 'Survived', 'PassengerId']].dropna().groupby(['honorific', 'Survived']).count().unstack().plot.bar(stacked=True)


# In[205]:


df_all1.loc[df_all.Age.isnull(), 'Age'] = df_all1[['honorific', 'Age']].groupby('honorific').transform(np.mean)
df_all1.drop(['PassengerId', 'Name', 'family_name', 'name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
df_all1.loc[~(df_all1['honorific'].isin(['Mr', 'Miss', 'Mrs', 'Master'])), 'honorific'] = 'others'


# In[208]:


df_all1['Embarked'].fillna('missing', inplace = True)


# In[220]:


for cat in df_all1.columns[df_all1.dtypes == 'object']:
    le = LabelEncoder()
    if df_all1[cat].dtypes == 'object':
        le = le.fit(df_all1[cat])
        df_all1[cat] = le.transform(df_all1[cat])


# In[222]:


X_train = df_all1[~df_all1.Survived.isnull()].drop('Survived', axis = 1).reset_index(drop=True)
Y_train = df_all1[~df_all1.Survived.isnull()]['Survived']


# # Making test set

# In[229]:


X_test = df_all1[df_all1.Survived.isnull()].drop('Survived', axis = 1).reset_index(drop=True)

X_train1, X_valid1, y_train1, y_valid1 = train_test_split(X_train, Y_train, test_size = 0.2)

categories = ['Embarked', 'Pclass', 'Sex', 'honorific', 'alone']


# # Light GBM Method

# In[234]:


lgb_train = lgb.Dataset(X_train1, y_train1, categorical_feature = categories)
lgb_eval = lgb.Dataset(X_valid1, y_valid1, categorical_feature = categories, reference = lgb_train)

lgbm_params = {
    'objective' : 'binary',
    'random_seed':1234,
}

model_lgb = lgb.train(lgbm_params,
                      lgb_train,
                      valid_sets = lgb_eval,
                      num_boost_round = 100,
                      early_stopping_rounds = 20,
                      verbose_eval = 10)


# In[260]:


df_impor = pd.DataFrame(model_lgb.feature_importance(), index=X_train1.columns, columns=['importance']).sort_values(by='importance', ascending=True)

plt.rcParams['figure.figsize'] = [8,4]
fig, ax = plt.subplots()
plt.barh(df_impor.index, df_impor.importance, color = 'g')
ax.set_axisbelow(True)
plt.grid(True)


# In[264]:


y_pred = model_lgb.predict(X_valid1, num_iteration=model_lgb.best_iteration)
accuracy_score(y_valid1, np.round(y_pred))


# # Linear Regression

# In[267]:


rf = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', max_depth = 5, oob_score = True, random_state = 10)
lr = LogisticRegression(random_state = 0)
lr.fit(X_train1, y_train1)
y_pred = lr.predict(X_valid1)
y_pre_prob = lr.predict_proba(X_valid1)[:, 1]

accuracy_score(y_pred, y_valid1)


# In[279]:


cols = X_train1.columns.tolist()
y_pos = np.arange(len(cols))
df_impor1 = pd.DataFrame(abs(lr.coef_[0]), index=cols, columns=['lr']).sort_values(by='lr', ascending=True)

plt.rcParams['figure.figsize'] = [5,4]
fig, ax = plt.subplots()
ax.barh(df_impor1.index, df_impor1.lr, align = 'center', color = 'g', ecolor = 'k')
ax.set_yticks(y_pos)
ax.set_yticklabels(df_impor1.index)
ax.set_xlabel('Coef')
ax.set_title('Each Feature Coef')
ax.set_axisbelow(True)
plt.grid(True)
plt.show()


# # Cross Validation with K Fold

# In[282]:


folds = 3
kf = KFold(n_splits = folds)


# In[283]:


models = []


# In[289]:


for train_index, val_index in kf.split(X_train):
    # print(val_index)
    X_train2 = X_train.iloc[train_index]
    X_valid2 = X_train.iloc[val_index]
    y_train2 = Y_train.iloc[train_index]
    y_valid2 = Y_train.iloc[val_index]
    
    lgb_train = lgb.Dataset(X_train2, y_train2, categorical_feature = categories)
    lgb_eval = lgb.Dataset(X_valid2, y_valid2, categorical_feature = categories, reference = lgb_train)
    
    model_lgb = lgb.train(lgbm_params,
                          lgb_train,
                          valid_sets=lgb_eval,
                          num_boost_round = 100,
                          early_stopping_rounds = 20,
                          verbose_eval =10,
                         )
    y_pred = model_lgb.predict(X_valid2, num_iteration = model_lgb.best_iteration)
    print(accuracy_score(y_valid2, np.round(y_pred)))
    models.append(model_lgb)


# In[293]:


preds = []

for model in models:
    pred = model.predict(X_valid2)
    preds.append(pred)


# In[296]:


preds_array = np.array(preds)
preds_mean = np.mean(preds_array, axis = 0)


# In[297]:


preds_int = (preds_mean > 0.5).astype(int)


# # # of persons each ticket class

# In[314]:


plt.rcParams['figure.figsize'] = [8,4]
fig, ax = plt.subplots()
plt.bar(df_all.Pclass.value_counts().index.astype(str), df_all.Pclass.value_counts(), color = 'g')
ax.set_axisbelow(True)
plt.grid(True)

