#!/usr/bin/env python
# coding: utf-8

# # Real Estate - Price Predictor

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing.isnull().sum()


# In[6]:


housing['MEDV'].isnull().sum()


# In[7]:


housing['MEDV'].mean()


# In[8]:


housing['MEDV'].replace(np.NaN,housing['MEDV'].mean()).head(506)


# In[9]:


housing.isnull().sum()


# In[10]:


housing['MEDV'].isnull().sum()


# In[11]:


housing['MEDV'].replace(np.NaN,housing['MEDV'].mean()).head(378)


# In[12]:


housing['MEDV'].isnull().sum()


# In[13]:


housing.dropna(inplace = True)
housing.isnull().sum()


# In[14]:


housing.info()


# In[15]:


housing['CHAS'].value_counts()


# In[16]:


housing.describe()


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


import matplotlib.pyplot as plt


# In[19]:


housing.hist(bins=50, figsize=(20,15))


# ## Train-Test Splitting

# In[20]:


# For learning Purpose
import numpy as np
from sklearn.model_selection import train_test_split
def split_train_test(data, test_ratio):
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data) * test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]
    


# In[21]:


# train_set, test_set=split_train_test(housing, 0.2)


# In[22]:


# print(f"Rows in train set: {len(train_set)} \nRows in test set: {len(test_set)}\n")


# In[23]:


from sklearn.model_selection import train_test_split
train_set, test_set=train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)} \nRows in test set: {len(test_set)}\n")


# In[24]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

# split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
# for train_index, test_index in split.split(housing, housing['CHAS']):
#     print(train_index, test_index)
#     strat_train_set = housing.loc[ train_index]
#     strat_test_set = housing.loc[test_index]
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, random_state=42, stratify=housing["CHAS"])


# In[25]:


strat_test_set


# In[26]:


# To store the copy of first dataset(original)
# housing = strat_train.copy()


# ## Looking For Correlations
# 

# In[27]:


corr_matrix = housing.corr()


# In[28]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[29]:


from pandas.plotting import scatter_matrix
attributes=["MEDV", "RM","ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize= (12,8))


# In[30]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


# ## Attribute Combinations

# In[31]:


housing["TAXRM"]=housing["TAX"]/housing["RM"]


# In[32]:


# housing["TAXRM"]
housing.head()


# In[33]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[34]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[35]:


housing=strat_train_set.drop("MEDV", axis=1)
housing_labels=strat_train_set["MEDV"].copy()


# In[36]:


# housing_labels


# ## Missing Attributes

# To take care of missing attributes ,We have three options:
# 
#     1.Get rid of the missing data points
#     2.Get rid of the whole attribute
#     3.Set the value to some attribute(0,mean,median)
#     
# Option 1:
#    a=housing.dropna(subset["RM"])
#    
#    a.shape
# 
# Option 2:
#     housing.drop("RM", axis=1).shape
#     
#     Note that there is no RM column 
#     
# Option 3:Compute median
#      median=housing["RM"].median()
#      
#      housing["RM"].fillna(median)

# ## Imputing for calculating missing data

# In[37]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[38]:


imputer.statistics_


# In[39]:


X = imputer.transform(housing)


# In[40]:


housing_tr=pd.DataFrame(X, columns=housing.columns)


# In[41]:


housing_tr.describe()


# ## Scikit-Learn Design

# Primariliy three typoes of objects:
# 1.Estimators: It estimates some parameter based on dataset.Eg.imputer.
# it has a fit and transform method.
# Fit method- Fits the dataset and calculates internal parameters.
# 
# 2.Transformers:transform method takes input and returns output based on he learnings from fit()
# 
# 3.Predictors:LinearRegression is an example of Predictor.fit() and predict() are two  common functions.It also gives score() function which will evaluate the predictors.

# ## Feature Scaling

# Primariliy,Two types of scaling methods:
# 
# 1.Min-Max Scaling(Normalization):
#                 (value - min)/(max - min)
#     
#     Skleran provides class called as MinMaxScaler for this
# 
# 2.Standardization:
#                 (value - mean)/std
#     
#     Sklearn provides class called as StandardScaler for this
# 

# ## Creating a Pipeline

# In[42]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])


# In[43]:


housing_num_tr=my_pipeline.fit_transform(housing_tr)


# In[44]:


housing_num_tr


# In[45]:


housing_num_tr.shape


# ## Selecting a desired model for Real Estates

# In[46]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model=LinearRegression()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[47]:


some_data = housing.iloc[:5]


# In[48]:


some_labels = housing_labels.iloc[:5]


# In[49]:


prepared_data=my_pipeline.transform(some_data)


# In[50]:


model.predict(prepared_data)


# In[51]:


list(some_labels)


# ## Evaluating the model

# In[52]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse=np.sqrt(mse)


# In[53]:


rmse


# ## Using better evaluation technique: Cross-Validation

# In[54]:


# 1 2 3 4 5 6 7 8  9 10       -10 folds
# Cross validatio needs utility. Utility is greater is better
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[55]:


rmse_scores


# In[56]:


def print_scores(scores):
    print("Scores:",scores)
    print("Mean", scores.mean())
    print("Standard Deviation:", scores.std())
    


# In[57]:


print_scores(rmse_scores)


# ## Saving the model

# In[58]:


from joblib import dump, load
dump(model, 'RealEstates.joblib')


# ## Testing the model on test data

# In[59]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))


# In[60]:


final_rmse


# In[61]:


prepared_data[0]


# ## Using the Model

# In[62]:


from joblib import dump, load
import numpy as np
model = load('RealEstates.joblib')


# In[63]:


features =np.array([[-0.5301006 ,  0.02099025, -0.66033355, -0.28997256, -1.18431286,
       -1.10275785, -1.03091974,  1.20484991, -0.52498507, -0.2399869 ,
        0.29292564,  0.39425368,  0.23912392]])
model.predict(features)

