#!/usr/bin/env python
# coding: utf-8

# In[132]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[133]:


df=pd.read_csv("car_price.csv")


# In[134]:


df


# In[135]:


df=df.drop('Unnamed: 0',axis=1)


# In[136]:


df


# In[137]:


df.isnull().sum()


# # car_name

# In[138]:


df['car_name'].value_counts()


# In[139]:


x=df['car_name'][0]
x[:x.index(" ")]


# In[140]:


def company_name(x):
    return x[:x.index(" ")]
    


# In[141]:


df['company_name']=df['car_name'].apply(company_name)


# In[142]:


def car_name(x):
    return x[x.index(" ")+1:]


# In[143]:


df['car_name']=df['car_name'].apply(car_name)


# In[144]:


df['car_name']


# In[145]:


df


# # car_price

# In[146]:


df['car_prices_in_rupee'].str.split()


# In[147]:


def currence_change(x):
    try:
        p = x.split(" ")
        if p[1] == "Lakh":
            return str(round(float(p[0]) * 100000, 1))
        elif p[1] == "Crore":
            return str(round(float(p[0]) * 10000000, 1))
    except :
        return x



# In[148]:


df['car_prices_in_rupee']=df['car_prices_in_rupee'].apply(currence_change)


# In[149]:


df['car_prices_in_rupee']=df['car_prices_in_rupee'].astype('float64')


# In[150]:


df['kms_driven']=df['kms_driven'].str.replace("kms",'')
df['kms_driven']=df['kms_driven'].str.replace(",",'')
df['engine']=df['engine'].str.replace("cc",'')
df['Seats']=df['Seats'].str.replace("Seats",'').astype('int')
df['kms_driven']=df['kms_driven'].astype("int64")
df['engine']=df['engine'].astype("int64")


# In[151]:


df.info()


# In[153]:


df


# # 2nd method to preprocesing car_name data

# In[23]:


# df['com_name']=df['car_name'].apply(lambda x:x.split()[0])


# In[24]:


#df['car_name']=df['car_name'].apply(lambda x:" ".join(x.split()[-5:-1]))


# # Encoding part

# In[154]:


from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer


# In[26]:


le=LabelEncoder()
le.fit()


# In[27]:


#df['car_name']=le.fit_transform(df['car_name'])
#df['fuel_type']=le.fit_transform(df['fuel_type'])
#df['transmission']=le.fit_transform(df['transmission'])
#df['ownership']=le.fit_transform(df['ownership'])
#df['company_name']=le.fit_transform(df['company_name'])


# In[28]:


#df


# # finding outliers

# In[155]:


plt.figure(figsize=(4,4))
sns.boxplot(x=df["engine"],data=df)
plt.show()


# In[156]:


q1=np.quantile(df['engine'],0.25)
q3=np.quantile(df['engine'],0.75)


# In[157]:


q1,q3


# In[158]:


iqr=q3-q1


# In[159]:


min_r=q1-1.5*iqr
max_r=q1+1.5*iqr


# In[160]:


min_r,max_r


# In[161]:


df=df[df['engine']<=max_r]


# In[162]:


df.shape


# In[238]:


df.to_csv("cleaned car data.csv")


# In[219]:


x=df.drop('car_prices_in_rupee',axis=1)
y=df['car_prices_in_rupee']


# In[220]:


x


# In[221]:


y


# In[222]:



ohe=OneHotEncoder()
ohe.fit(x[['car_name','fuel_type','transmission','ownership','company_name']])


# In[223]:


column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['car_name','fuel_type','transmission','ownership','company_name']),remainder='passthrough')


# In[224]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[225]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)


# In[226]:


x_train.fillna(0, inplace=True)
x_test.fillna(0, inplace=True)
y_train.fillna(0, inplace=True)
y_test.fillna(0, inplace=True)


# In[227]:


lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)


# In[228]:


pipe.fit(x_train,y_train)


# In[232]:


y_pred=pipe.predict(x_test)


# In[233]:


r2_score(y_test,y_pred)


# In[235]:


import pickle 


# In[236]:


pickle.dump(pipe,open('carfile.pkl','wb'))


# In[ ]:




