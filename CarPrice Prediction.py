#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[5]:


df = pd.read_csv('Downloads/car_proj/car data.csv')


# In[6]:


df.head()


# In[8]:


df.shape


# In[14]:


print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[15]:


#check missing or null values
df.isnull().sum()


# In[16]:


df.describe()


# In[17]:


df.columns


# In[24]:


final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[28]:


final_dataset.head()


# In[29]:


final_dataset['no_year'] = final_dataset['Current_year'] - final_dataset['Year']


# In[30]:


final_dataset.head()


# In[31]:


final_dataset.drop(['Year'],axis =1, inplace =True)


# In[32]:


final_dataset.head()


# In[33]:


final_dataset.drop(['Current_year'],axis =1, inplace =True)


# In[34]:


final_dataset.head()


# In[35]:


final_dataset = pd.get_dummies(final_dataset,drop_first=True)


# In[36]:


final_dataset.head()


# In[37]:


final_dataset.corr()


# In[38]:


import seaborn as sns
sns.pairplot(final_dataset)


# In[39]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


cormat = final_dataset.corr()
top_corr_features = cormat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[42]:


#independent and dependent features
x = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]
x.head()


# In[43]:


y.head()


# In[47]:


#Feature importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(x,y)


# In[48]:


print(model.feature_importances_)


# In[50]:


#plot graph of feature importances for better visualization 
feat_importances = pd.Series(model.feature_importances_, index=x.columns) 
feat_importances.nlargest(5).plot(kind='barh') 
plt.show()


# In[65]:


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
x_train.shape


# In[66]:


from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()


# In[55]:


###Hyperparameters for tuning
import numpy as np
n_estimator = [int(z) for z in np.linspace(start=100, stop = 1200, num =12)]
print(n_estimator)


# In[67]:


from sklearn.model_selection import RandomizedSearchCV


# In[80]:


#Randomized search CV
#Number of Trees in random forest
n_estimators = [int(z) for z in np.linspace(start=100, stop = 1200, num =12)]
#NUmber of features to select at every split
max_features = ['auto','sqrt']
#Maximum number of levels in trees
max_depth = [int(z) for z in np.linspace(5, 30, num =6)] 
#max.depth.append(NOne)
#Minimum number of sample required to split a node
min_samples_split = [2,5,10,15,100]
#Minimum number of sample required at each leaf node
min_samples_leaf = [1,2,5,10]


# In[81]:


random_grid = {'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,
            'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}


# In[82]:


print(random_grid)


# In[83]:


#using random grid to search for best hyperparameters
#First creating base model to train 
rf = RandomForestRegressor()


# In[84]:


rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring ='neg_mean_squared_error',
                               n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)


# In[85]:


rf_random.fit(x_train,y_train)


# In[86]:


predictions = rf_random.predict(x_test)


# In[87]:


predictions


# In[88]:


sns.distplot(y_test-predictions)


# In[ ]:


#Predictions looks good as difference of y_test-predictions values near 0 are more in number


# In[89]:


plt.scatter(y_test,predictions)


# In[90]:


#Again Predictions looks good as correlation y_test & predictions are almost linear


# In[91]:


import pickle


# In[92]:


file = open('random_forest_regression_model.pkl','wb')
pickle.dump(rf_random, file)


# In[ ]:




