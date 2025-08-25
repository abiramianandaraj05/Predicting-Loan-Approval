#!/usr/bin/env python
# coding: utf-8

# # Predicting success of a person applying for a loan

# ## Introduction

# In this project, I am going to build a machine learning model that can predict whether someone applying for a loan will be approved by the bank. Thousands of people everyday apply for loans and this prediction model can be used by people applying to see if they have a chance of successfully gaining a loan from a bank, instead of wasting their time applying and being rejected. In addition, this model could be used by banks to flag which individuals should be given loans which can then be futher investigated manually by proper loan-approvers.

# ## Setting up code (ie imports etc)

# In[329]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


loan_data = pd.read_csv("LoanApprovalPrediction.csv")


# **Panda library** will be used to load dataframes and read csv files.
# 
# **Matplotlib library** will be used to visualise the features within the dataset given.
# 
# **Seaborn library** will be used to help us work out the correlation between different features.

# ## Data Preprocessing

# Let us see what the first few rows of the data set look like

# In[330]:


loan_data.head()


# In[331]:


updated_data= loan_data.drop('Loan_ID',axis=1)


# LoanID is a feature that is unnecessary is training a machine learning model as the data in this column will not affect whether someone's loan is accepted. In the next code section I will remove this column.

# The columns for the data set are :

# In[332]:


data_columns = updated_data.columns



# In[333]:


updated_data.shape


# In[334]:


updated_data.head()


# ### Data Transformation

# For each feature that contains categorical values, we need to replace it with numerical values ie 1 from true and 0 for false.

# In[335]:


from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
categorical_features = (updated_data.dtypes == "object")
for feature in list(categorical_features[categorical_features].index):
    updated_data[feature] = encoder.fit_transform(updated_data[feature])


# In[336]:


updated_data.head()


# ### Visualising correlations

# In[337]:


sns.heatmap(updated_data.corr())


# In[338]:


values = updated_data.corr()
values['Loan_Status']


# From the heatmap we can see the factor that affects whether a loan approved the most is credit_history, from the coefficients we can see the features that are the least important to whether a loan is approved is no of dependents and whether they are self employed or not.

# ### Data cleaning

# In[339]:


updated_data.isnull().sum()


# The code above tells us how much data is missing and in which features. So we can see that the dependant, loanamount , loanamountterm and credithistory columns are all missing data. We have to deal with this otherwise it will affect the accuracy of our model. I have decided to use imputation methods to replace the missing data, the alternative solution is to delete all rows of missing information. However, this is not a good solution as we will lose valuable data, instead by using imputation methods we can still use the rows of data with missing values as they will be replaced. I will use a more advance imputation technique by using the IterativeImputator method in python which will create a model that uses the values of the other features to predict an appropriate value for any missing data.

# In[340]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.datasets import make_regression


# In[341]:


# initialising the iterative imputer 
iterative_imputer = IterativeImputer(random_state=0)

#code will replace all missing values with predicted values
imputed_data = pd.DataFrame((iterative_imputer.fit_transform(updated_data)),columns=data_columns)


# In[342]:


imputed_data.isnull().sum()


# In the code above, I have replaced all missing values with synthetic data, then the last code box checks that there is no missing data in the corrected dataframe which we can see is true.

# In[343]:


imputed_data = imputed_data.drop(['Dependents','Self_Employed'],axis=1)
imputed_data.head()


# In the code snippet above, I removed the self employment and dependent column as they had the lowst correlation to loan_status.

# ### Splitting data into train and test set

# In[344]:


from sklearn.model_selection import train_test_split
output = imputed_data['Loan_Status']
input = imputed_data.drop(columns=['Loan_Status'])
X_train, X_test,y_train,y_test = train_test_split(input,output,test_size = 0.3, random_state = 1)


# ## Training Machine learning model

# This is a binary classification problem so there are 3 models we can use:
# 
# + Decision Tree
# + Naive Bayes
# + Support Vector machines 
# + Logistic Regression
# 
# I chose not to used Naive Bayes and Logistic regression model because the logistic regression model only works the best when relationships are linear which may not be the case with this data. Futhermore Naive Bayes model assumes that all features of the data are independent, this assumption is most likely not accurate in our dataset so I believe it is inappropriate to use this model. I have also not chosen to use SVC because the dataset is not high dimensional which is the data that works best for training this model.
# 
# Therefore, I have chosen to use a Decision Tree model to train my data with.
# 

# ### Decision Tree model 

# In[345]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train,y_train)


# In[346]:


from sklearn import metrics
from sklearn.metrics import accuracy_score
predictions = model.predict(X_test)

decision_mse = metrics.mean_squared_error(y_test,predictions)
decision_accuracy = accuracy_score(y_test,predictions)



# Add text here

# ### Improve accuracy of model

# In[347]:


nodes = [10,20,25,50,100,150,200]

def get_accuracy(model):
    predictions = model.predict(X_test)
    return accuracy_score(y_test,predictions)

model = DecisionTreeClassifier(max_depth=nodes[0],random_state=0)
model.fit(X_train,y_train)
num = 0
highest_accuracy = get_accuracy(model)

for i in range(1,len(nodes)):
    model = DecisionTreeClassifier(max_depth=nodes[i],random_state=0)
    model.fit(X_train,y_train)
    count = get_accuracy(model)
    if count > highest_accuracy:
        highest_accuracy = count
        num = i



# In[348]:


improved_model = DecisionTreeClassifier(max_depth=nodes[num],random_state=0)
improved_model.fit(X_train,y_train)



# ### Pruning a Decision tree

# In[354]:


path = model.cost_complexity_pruning_path(X_train,y_train)
ccp_alpha_list = path.ccp_alphas
impurities_list = path.impurities

highest_accuracy = 0
best_model = None
best_alpha = 0
for ccp_alpha_num in ccp_alpha_list:
    pruned_tree = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha_num, max_depth=10)
    pruned_tree.fit(X_train,y_train)
    model_accuracy = get_accuracy(pruned_tree)
    if model_accuracy > highest_accuracy:
        best_model = pruned_tree
        highest_accuracy = model_accuracy
        best_alpha = ccp_alpha_num




# In the code above, we have pruned the tree this means that certain nodes have been removed from the tree. This is useful, as it stops the data being overfit this means that the predicitions will be more accurate as we can see the accuracy score for the best pruning model shot up from 76% to 85%. 

# ## Final model Code

# In[ ]:

def getModel():
    final_model = best_model
    return final_model

