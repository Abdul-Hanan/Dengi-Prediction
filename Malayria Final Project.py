#!/usr/bin/env python
# coding: utf-8

# # Abdul Hanan Ashraf 18101147

# In[161]:


import pandas as pd
import numpy as np


# In[187]:


dftrainfeatures=pd.read_csv(r"C:\Users\my pc\Desktop\Tools Final Malayria Project\dengue_features_train.csv");
dftrainlabels=pd.read_csv(r"C:\Users\my pc\Desktop\Tools Final Malayria Project\dengue_labels_train.csv");
dftest=pd.read_csv(r"C:\Users\my pc\Desktop\Tools Final Malayria Project\dengue_features_test.csv");


# In[188]:


dftrainfeatures.head()


# In[189]:


dftrainfeatures.isnull().any()


# In[190]:


dftrainfeatures.drop('city', axis = 1, inplace = True)
dftrainfeatures.drop('year', axis = 1, inplace = True)
dftrainfeatures.drop('weekofyear', axis = 1, inplace = True)
dftrainfeatures.drop('week_start_date', axis = 1, inplace = True)


# In[191]:


dftrainfeatures.head()


# In[192]:


import seaborn as s
dftrainfeatures.boxplot(['ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw'])


# In[193]:


#Sensor data ofen has noise and ourliners we are going to drop them or replace them with mean
dftrainfeatures.describe()


# In[194]:


# Not that important , this is repeated mantually by putting the quartiles to get IQR ranges
q1=0.144209
q3=0.246982
IQR=1.5*(q3-q1)
q11=q1-IQR
q31=q3+IQR
print(q11)
print(q31)


# In[195]:


dftrainfeatures["ndvi_ne"] = np.where(dftrainfeatures["ndvi_ne"] >0.5537825000000001, 0.142294,dftrainfeatures['ndvi_ne'])
dftrainfeatures["ndvi_ne"] = np.where(dftrainfeatures["ndvi_ne"] <-0.26034950000000007, 0.142294,dftrainfeatures['ndvi_ne'])

dftrainfeatures["ndvi_nw"] = np.where(dftrainfeatures["ndvi_nw"] >0.4676745, 0.130553,dftrainfeatures['ndvi_nw'])
dftrainfeatures["ndvi_nw"] = np.where(dftrainfeatures["ndvi_nw"] <-0.20185749999999997,0.130553,dftrainfeatures['ndvi_nw'])

dftrainfeatures["ndvi_se"] = np.where(dftrainfeatures["ndvi_se"] >0.3894845, 0.203783,dftrainfeatures['ndvi_se'])
dftrainfeatures["ndvi_se"] = np.where(dftrainfeatures["ndvi_se"] <0.014448500000000003,0.203783,dftrainfeatures['ndvi_se'])

dftrainfeatures["ndvi_sw"] = np.where(dftrainfeatures["ndvi_sw"] >0.40114150000000004, 0.202305,dftrainfeatures['ndvi_sw'])
dftrainfeatures["ndvi_sw"] = np.where(dftrainfeatures["ndvi_sw"] <-0.009950500000000001, 0.202305,dftrainfeatures['ndvi_sw'])


# In[196]:


dftrainfeatures.boxplot(['ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw'])


# In[197]:


dftrainlabels.head()


# In[198]:


dftrainfeatures = dftrainfeatures.fillna(dftrainfeatures.mean())
X = dftrainfeatures.iloc[:,0:22]
Y = dftrainlabels[['total_cases']]


# In[199]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
# normalize the data attributes
#values scaled between 0-1
scaler = preprocessing.MinMaxScaler()
scaler.fit(X) #X values ----- Y
normalized_X = scaler.transform(X)


# In[200]:


X_train, X_test, y_train, y_test = train_test_split(normalized_X, Y, test_size=0.3, 
                                                    random_state=0) 


# In[201]:


from sklearn.model_selection import GridSearchCV
model = LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
clf = GridSearchCV(model, parameters, cv=10, verbose=0, n_jobs=-1, refit=True)
clf.fit(X_train,y_train)
clf.score(X_train,y_train)



# In[220]:


from sklearn import ensemble
params = {'n_estimators': 600, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.02, 'loss': 'ls'}
gbr = ensemble.GradientBoostingRegressor(**params)
gbr.fit(X_train, y_train)
gbr.score(X_train,y_train)


# In[221]:


predictionsLR = clf.predict(X_test)
predictionsGB = gbr.predict(X_test)


# In[222]:


from sklearn.metrics import mean_absolute_error


# In[223]:


mean_absolute_error(y_test, predictionsLR)


# In[224]:


mean_absolute_error(y_test, predictionsGB)


# In[225]:


predictionsGB


# In[226]:


dftest.describe()


# In[227]:


dftest.drop('city', axis = 1, inplace = True)
dftest.drop('year', axis = 1, inplace = True)
dftest.drop('weekofyear', axis = 1, inplace = True)
dftest.drop('week_start_date', axis = 1, inplace = True)


# In[228]:


# Not that important , this is repeated mantually by putting the quartiles to get IQR ranges
q1=0.134079
q3=0.253243

IQR=1.5*(q3-q1)
q11=q1-IQR
q31=q3+IQR
print(q11)
print(q31)


# In[229]:


dftest["ndvi_ne"] = np.where(dftest["ndvi_ne"] >0.6605725, 0.126050,dftest['ndvi_ne'])
dftest["ndvi_ne"] = np.where(dftest["ndvi_ne"] <-0.3987435, 0.126050,dftest['ndvi_ne'])

dftest["ndvi_nw"] = np.where(dftest["ndvi_nw"] >0.5820375, 0.126803,dftest['ndvi_nw'])
dftest["ndvi_nw"] = np.where(dftest["ndvi_nw"] <-0.3236625, 0.126803,dftest['ndvi_nw'])

dftest["ndvi_se"] = np.where(dftest["ndvi_se"] >0.41417250000000005, 0.207702,dftest['ndvi_se'])
dftest["ndvi_se"] = np.where(dftest["ndvi_se"] <-0.010631500000000044,0.207702,dftest['ndvi_se'])

dftest["ndvi_sw"] = np.where(dftest["ndvi_sw"] >0.43198899999999996, 0.201721,dftest['ndvi_sw'])
dftest["ndvi_sw"] = np.where(dftest["ndvi_sw"] <-0.044666999999999984,0.201721,dftest['ndvi_sw'])


# In[230]:


dftest.boxplot(['ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw'])


# In[231]:


dftest = dftest.fillna(dftest.mean())


# In[232]:


scaler = preprocessing.MinMaxScaler()
scaler.fit(dftest) #X values ----- Y
normalized_test = scaler.transform(dftest)


# In[233]:


dftest.describe()


# In[234]:


predictions = gbr.predict(normalized_test)
rounded = [np.round(x) for x in predictions]


# In[235]:


pd.DataFrame(rounded, columns=['predictions']).to_csv('prediction.csv')


# In[ ]:





# In[ ]:





# In[ ]:




