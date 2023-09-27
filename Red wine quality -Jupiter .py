#!/usr/bin/env python
# coding: utf-8

# In[25]:


print('Predicting the quality of redwine based on the data below')


# In[24]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[2]:


df = pd.read_csv('winequality.csv')


# In[3]:


print(df.shape)
print(df.groupby('quality').size())
df.describe()


# In[4]:


X = df.drop('quality', axis=1)
y = df['quality']


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[7]:


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[8]:


y_pred = model.predict(X_test)


# In[9]:


mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')


# In[10]:


importances = model.feature_importances_
sorted_idx = np.argsort(importances)


# In[11]:


print("Feature ranking:")


# In[15]:


for f in range(X_train.shape[1]):
    print("{}. feature {} ({})".format(f+1, sorted_idx[f], importances[sorted_idx[f]]))


# In[16]:


from sklearn.svm import SVR

svr_lin = SVR(kernel='linear', C=1)
svr_lin.fit(X_train, y_train)
y_pred = svr_lin.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Linear SVR MSE:", mse)

from sklearn.model_selection import GridSearchCV
params = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01]}
svr_rbf = SVR(kernel='rbf')
clf = GridSearchCV(svr_rbf, params, scoring='neg_mean_squared_error', cv=5)
clf.fit(X_train, y_train)
y_pred = clf.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("RBF SVR MSE:", mse)

print("Random Forest MSE:", 0.262) # original MSE


# In[18]:


print ("For comparison, the original random forest model that i used has a test MSE of .262, so the So the best performing model is the RBF SVR, beating the Random Forest. The hyperparameter tuning helped improve performance significantly")


# In[19]:


from sklearn.model_selection import RandomizedSearchCV
 
params = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01]}
svr = SVR(kernel='rbf')

random_search = RandomizedSearchCV(svr, params, n_iter=10, scoring='neg_mean_squared_error', cv=5)
random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)


# In[20]:


importances = model.feature_importances_  

FI_threshold = 0.5
selected_idx = np.where(importances > FI_threshold)[0]
print("Number of selected features:", len(selected_idx))

X_train_selected = X_train[:, selected_idx]
X_test_selected = X_test[:, selected_idx]


# In[21]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('svr', SVR(kernel='rbf')) 
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Pipeline MSE:", mse)


# In[22]:


pipe_params = {
    'scale': [StandardScaler(), None],
    'svr__C': [0.1, 1, 10],
    'svr__gamma': [0.001, 0.01]
}

grid_pipeline = GridSearchCV(pipe, pipe_params, cv=5)
grid_pipeline.fit(X_train, y_train)

print("Best pipeline:", grid_pipeline.best_estimator_)


# In[23]:


print("It was very dificult working on this project and I have leant a lot about python and jupiter notebook plus how to fix and run error on the project. I ask for a lot of help during this project and I will collaboration is important and use a lotof other ressources like blogs, websites, kaggles to help me along.")


# In[ ]:




