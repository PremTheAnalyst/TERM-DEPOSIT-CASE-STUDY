#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler


# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier


# In[4]:


get_ipython().system(' pip install xgboost')
from xgboost import XGBClassifier 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import roc_auc_score ,mean_squared_error,accuracy_score,classification_report,roc_curve,confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from scipy.stats.mstats import winsorize
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
pd.set_option('display.max_columns',None)
import six
import sys
sys.modules['sklearn.externals.six'] = six


# In[5]:


df = pd.read_excel('Historical_data.xlsx')
y = df['term_deposit_subscribed']
X = df.drop(['customer_id','term_deposit_subscribed'], axis=1)


# In[6]:


data = df.drop('customer_id', axis=1)
plt.figure(figsize=(20,10))
sns.heatmap(data.corr(),annot=True,center = 0 , cmap ='PuRd_r');


# In[7]:


#Converting string data to numerical data using one-hot encoding
categ = []
for col, value in X.iteritems():
    if value.dtype == 'object':
        categ.append(col)
df_cat = df[categ]
df_cat = pd.get_dummies(df_cat)
df = pd.concat([df, df_cat],  axis = 1)


# In[8]:


for col in df.columns:
    print(col)


# In[9]:


#As it is evident that new encoded features have been added to the dataframe (and are all numeric data) but the original ones 
#are not removed. 
#So we drop those columns.
df = df.drop(['job_type','marital','education','default','housing_loan','personal_loan','communication_type','month','prev_campaign_outcome'], axis=1)
print("Shape of the dataframe is", df.shape)
df.head()


# In[10]:


#Removing rows with null values
df = df.dropna(axis=0)
y = df['term_deposit_subscribed']
X = df.drop(['customer_id','term_deposit_subscribed'], axis=1)
X.shape, y.shape


# In[11]:


for col in df.columns:
    print(col)


# In[12]:


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.30, random_state=24)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[13]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[14]:


y_pred = model.predict(X_test)

print("Accuracy:",sklearn.metrics.accuracy_score(y_test, y_pred))
print("Precision:",sklearn.metrics.precision_score(y_test, y_pred))
print("Recall:",sklearn.metrics.recall_score(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[15]:


m1 = DecisionTreeClassifier(random_state=24)
m1.fit(X_train, y_train)


# In[16]:


y_pred = m1.predict(X_test)

print("Accuracy:",sklearn.metrics.accuracy_score(y_test, y_pred))
print("Precision:",sklearn.metrics.precision_score(y_test, y_pred))
print("Recall:",sklearn.metrics.recall_score(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[17]:


m2 = GradientBoostingClassifier(learning_rate=0.5, n_estimators=100)
m2.fit(X_train, y_train)


# In[18]:


y_pred = m2.predict(X_test)
import sklearn
print("Accuracy:",sklearn.metrics.accuracy_score(y_test, y_pred))
print("Precision:",sklearn.metrics.precision_score(y_test, y_pred))
print("Recall:",sklearn.metrics.recall_score(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[19]:


m3 = XGBClassifier(learning_rate=0.01, n_estimators = 1000, random_state=24)
m3.fit(X_train, y_train)


# In[20]:


y_pred = m3.predict(X_test)
import sklearn
print("Accuracy:",sklearn.metrics.accuracy_score(y_test, y_pred))
print("Precision:",sklearn.metrics.precision_score(y_test, y_pred))
print("Recall:",sklearn.metrics.recall_score(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[21]:


from scipy import stats


# In[22]:


#One way of using XGBClassifier
clf_xgb = XGBClassifier(objective = 'binary:logistic')
param_dist = {'n_estimators': stats.randint(150, 1000),
              'learning_rate': stats.uniform(0.01, 0.59),
              'subsample': stats.uniform(0.3, 0.6),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.4),
              'min_child_weight': [1, 2, 3, 4]
             }

numFolds = 5
kfold_5 = KFold(n_splits = numFolds)

clf = RandomizedSearchCV(clf_xgb, 
                         param_distributions = param_dist,
                         cv = kfold_5,  
                         n_iter = 5, # you want 5 here not 25 if I understand you correctly 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 3, 
                         n_jobs = -1)


# In[23]:


clf.fit(X_train, y_train)


# In[24]:


y_pred = clf.predict(X_test)

print("Accuracy:",sklearn.metrics.accuracy_score(y_test, y_pred))
print("Precision:",sklearn.metrics.precision_score(y_test, y_pred))
print("Recall:",sklearn.metrics.recall_score(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[25]:


m4 = XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.2, learning_rate = 0.005, subsample = 1, random_state=24,
                max_depth = 15, alpha = 10, n_estimators = 2000, base_score=0.7)

m4.fit(X_train, y_train)


# In[26]:


y_pred = m4.predict(X_test)

print("Accuracy:",sklearn.metrics.accuracy_score(y_test, y_pred))
print("Precision:",sklearn.metrics.precision_score(y_test, y_pred))
print("Recall:",sklearn.metrics.recall_score(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:


get_ipython().system('pip install lightgbm')
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier


# In[ ]:


#making an instance of the classifier
m5 = LGBMClassifier(objective='binary', n_estimators=1500, colsample_bytree=0.35, reg_alpha=0.2,reg_lambda=0.2, 
                        max_depth=2, learning_rate=0.01, random_state=24)

#fitting the model on train data
m5.fit(X_train, y_train)


# In[ ]:


y_pred = m5.predict(X_test)

print("Accuracy:",sklearn.metrics.accuracy_score(y_test, y_pred))
print("Precision:",sklearn.metrics.precision_score(y_test, y_pred))
print("Recall:",sklearn.metrics.recall_score(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:


dft = pd.read_excel("New_customer_list_data.xlsx", index_col = "customer_id")
print("Shape of the dataframe is", dft.shape)
dft.head()


# In[ ]:


tcateg = []
for tcol, tvalue in dft.iteritems():
    if tvalue.dtype == 'object':
        tcateg.append(tcol)
dft_cat = dft[tcateg]
dft_cat = pd.get_dummies(dft_cat)
dft = pd.concat([dft, dft_cat], axis = 1)


# In[ ]:


dft = dft.drop(['job_type','marital','education','default','housing_loan','personal_loan','communication_type','month','prev_campaign_outcome'], axis=1)
dft.head()


# In[ ]:


print(dft.shape)
output= m5.predict_proba(dft)
print(output)
output
dft['term_deposit_subscribed'] = m5.predict(dft)
dft['term_deposit_subscribed']


# In[ ]:


new_dft=  pd.read_excel("New_customer_list_data.xlsx")
final_list= []
cus_id_list= new_dft['customer_id']
for i in range(len(output)):
    temp_list= []
    temp_list.append(cus_id_list[i])
    temp_list.append(output[i][1])
    final_list.append(temp_list)
    
# print(len(final_list), final_list[0])

def Sort(sub_li):
    return(sorted(sub_li, key = lambda x: x[1], reverse= True))

answer= Sort(final_list)
new_ans= []
for i in range(len(answer)):
    new_ans.append(answer[i][0])
    
print(new_ans[0:5])
new_dft['term_deposit_subscribed']=new_ans


# In[ ]:


#Add customer_id to the exported csv
#dft['term_deposit_subscribed'].to_csv (r'{write your own path where ever you want the excel sheet to be}', index = True, header=True)


# In[ ]:


new_dft['term_deposit_subscribed'][0:1000].to_csv(('new_customer_output.csv'), index=False, header=False)


# print(dft.shape)
# dft.head()
# dft= dft.drop(['term_deposit_subscribed'], axis=1)
# # output= m5.predict_proba(dft)
# # dft['term_deposit_subscribed'] = m5.predict(dft)
# # dft['term_deposit_subscribed']
