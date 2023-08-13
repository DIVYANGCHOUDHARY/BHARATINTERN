#!/usr/bin/env python
# coding: utf-8

# In[5]:


#import required libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# % matlpotlib in line


# In[7]:


df= pd.read_csv('train.csv')
df.head()


# In[8]:


df.shape


# In[9]:


#Analyzing the Titanic Data

sns.countplot(x='Survived', data=df)

# NOTE: Here in the below graph number males who had survived is nearly 110 and number of females is nearly 220


# In[10]:


sns.countplot(x='Survived', hue='Sex', data=df, palette='winter')

# NOTE: Here in the below graph 0 -> represents 'NOT SURVIVED' and 1 -> reprents 'SURVIVED'
# NOTE: Womens are thrice more likely to survive than males


# In[12]:


sns.countplot(x='Survived', hue='Pclass', data=df, palette='PuBu')

# NOTE: Passengers of 1-class are more likely to be survived than 2-class and 3-class
#       3-class passengers are more likely to be not survived 


# In[13]:


df['Age'].plot.hist()

#NOTE: Here in the below we can easily say that the highest age groups of passengers is youth and its between age of 20-40
#      Least age group of passengers is between age of 70-80


# In[15]:


df['Fare'].plot.hist(bins=20, figsize=(10,5))

# NOTE: Here in the below graph we can analyze that most of and very are on the higher side of the fare price i.e. 200-500


# In[17]:


sns.countplot(x='SibSp', data=df, palette='rocket')

# NOTE: Here we analyzed that majority of the passengers are travelling alone 


# In[18]:


df['Parch'].plot.hist()


# In[20]:


sns.countplot(x='Parch', data=df, palette='summer')

# NOTE: Here we are analyzing that the numbers of parents and siblings who aboard the ship are less


# In[22]:


#Data-Cleaning

df.isnull().sum()

# NOTE: By analyzing the below output we can say 'age' and 'cabin' has most null values


# In[23]:


sns.heatmap(df.isnull(), cmap='spring')

# NOTE: Plotting the null values on heatmap
#      Here in the below box yellow color shows nullvalues


# In[ ]:


sns.boxplot(x='Pclass', y='Age', data=df)

# NOTE: We can observe that older age group passengers are travelling more in Class-1 and Class-2 as compared to Class-3
#       The hue parameter determines which column in the data frame should be used for colour encoding


# In[24]:


#Dropping irrelevent columns as they are not required

df.head()


# In[25]:


df.drop('Cabin', axis=1, inplace=True)


# In[28]:


# Dropped the cabin column

df.head(3)


# In[29]:


df.dropna(inplace=True)


# In[31]:


sns.heatmap(df.isnull(), cbar=False)

# NOTE: After dropping all the null values now its not showing any nullvalues on heatmap


# In[32]:


df.isnull().sum()


# In[33]:


df.head(2)


# In[42]:


#Training Data

x= df.drop('Survived', axis=1)
y= df['Survived']


# In[43]:


from sklearn.model_selection import train_test_split


# In[46]:


X_train, X_test, Y_train, Y_test= train_test_split(x, y, test_size= 0.33, random_state= 4)


# In[47]:


#Using Logistic Regression

from sklearn.linear_model import LogisticRegression
lm= LogisticRegression()


# In[48]:


lm.fit(X_train, Y_train)


# In[49]:


prediction= lm.predict(X_test)


# In[51]:


from sklearn.metrics import classification_report


# In[54]:


from sklearn.metrics import classification_report


# In[55]:


from sklearn.metrics import confusion_matrix


# In[56]:


confusion_matrix(Y_test, prediction)


# In[57]:


from sklearn.metrics import accuracy_score


# In[58]:


accuracy_score(Y_test, prediction)

