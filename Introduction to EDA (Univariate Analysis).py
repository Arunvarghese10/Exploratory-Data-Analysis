#!/usr/bin/env python
# coding: utf-8

# # EDA  (Exploratory Data Analysis)
# **univariate analysis**
# 
# **bivariate analysis**
# 
#            **num vs num
#            **cat vs num - Boxplot
#            **cat vs cat
# 
# **Missing values**
# 
# **Outlier analysis and removal-Boxplot & Normal Distribution.**
# 
# **Feature Engineering.**
# 
# **Statistical Analysis to verify the relation between Predictor & Target**
# 
# **Data Transformation, Scaling & Encoding ~ Data Preprocessing Stage**
# 
# **Model Building**

# In[1]:


## Here we will focus on all the major parameters in Exploratory Data Analysis as mentioned above, 
## and will explain each topic while working with a dataset, draw insights and make inferences from the data. 


# In[7]:


## Importing the important libraries in python in order to work effortlessly in python.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


## Importing the data set directly from the url.


sales=pd.read_csv('https://datahack-prod.s3.amazonaws.com/train_file/train_v9rqX0R.csv')


# In[14]:


## Using the head function to inspect the top 5 rows of the dataset,
## Gives us an upperhand in learning more/comprehending about the Data 
##and later working on it in the best possible way.

sales.head()


# In[16]:


sales.shape 

## To get an idea about the total no. of rows and columns in the provided data set


# In[18]:


sales.describe()


## To have a basic understanding of the range of numerical columns in the data
## and how they are spread out from a descriptive perspective.


# In[20]:


nums=sales.select_dtypes(include=np.number).columns

# to find the numerical variables in a dataset and undergo the
## process of univariate analysis, we must store the numeical columns of the
## Data in a variable and then begin the process of univariate analysis.

nums


# # UNIVARIATE Analysis:

# In[21]:


len(nums)


# In[23]:


# for warnings
import warnings
warnings.filterwarnings('ignore')
#Plot dimension
plt.rcParams['figure.figsize']=[20,10]

n_rows=3
n_cols=2
counter=1
for i in nums:
    plt.subplot(n_rows,n_cols,counter) # Creating the space for 6 plots so the plots can be shown 
    sns.distplot(sales.loc[:,i].dropna()) # command for creating distplot using loop, i is iterating through the columns
                                          #.dropna()is used to drop the missing values in the columns
    counter+=1
plt.tight_layout()
plt.show()


##Plotted the graph using loops for all the numerical columns in the dataset
##Now we would be furher plotting the distplot of these individual columns 
## And try to draw meaningful insights from the same,


# In[33]:


sns.distplot(sales.loc[:,'Item_Weight'])


## Inference. 

## Item Weight is kind of uniform in nature, as not much changes were observed in 
## The given plot.


# In[25]:


sns.distplot(sales.loc[:,'Item_Visibility'])


# In[30]:


sns.distplot(sales.loc[:,'Item_MRP'])


# Inference
# This is a mutimodal data Because there are multiple modes of the product MRP


# In[32]:


sns.distplot(sales.loc[:,'Outlet_Establishment_Year'])


##Inference

## No such meaningfull insights drawn from the outlet estblishment year column


# In[34]:


sns.distplot(sales.loc[:,'Item_Outlet_Sales'])


#Inference
#Item outlet sales is positively skewed in nature


# In[36]:


# Plotting the categorical column


# In[39]:


## Checking for all the categorical variables in the data,
## trying to draw meaningful insights from each of the columns 

cat=sales.select_dtypes(include=np.object_).columns
cat


# In[44]:


## Checking for "Item_Fat_Content"

sales.Item_Fat_Content.value_counts().plot(kind="bar")
plt.show()


##Here we can make out that basically there are only 2 types of fat categories
## But the data seems to have multiple categories, representing the same thing.
## So we will replace these irragularities, by replacing them,
## And bringing them in the same category and draw a better meaningful insight.


## Thus replacing the LF and REg in their respective categories.


# In[47]:


sales.Item_Fat_Content.replace(to_replace=['LF','low fat','reg'],value=['Low Fat','Low Fat','Regular'],inplace=True)


# In[52]:


sales.Item_Fat_Content.value_counts().plot(kind='bar')
                                           
                    
##Inference.
## THe data here tends to contain more items of low fat as compared to regular 


# In[55]:


# Item Type 

sales.Item_Type.value_counts().plot(kind='bar')

## With this we can say that the top 5 selling products are 

#'Fruits and Vegetables' 
#'Snack Foods'
#'Household' 
#'Frozen Foods',
#'Dairy'.


# In[57]:


# For Outlet_id
sns.countplot(sales.Outlet_Identifier)
plt.show()


# In[59]:


## Inference

##  the outlet 10 and outlet 19 has the lowest sales as compare to the other outlets


# In[62]:


##CHecking for outlet size

sns.countplot(sales.Outlet_Size)

## Inference
## Medium and small sized outlets are performing better 


# In[66]:


# Outlet_Type


sns.countplot(sales.Outlet_Type)


## Inference 
## With this we can say that the supermarket Type 1 outlet types 
## Are performing extremly well.


# **SUMMARY of UNIVARIATE ANALYSIS**
# 
# 
#  **Out 27 is the maximum revenue genrator for the buisness
#  
#  **We relaize this stores is the mostly opened in Tier 3 cities and the type of the outlet is medium size outlet
#  
#  **The items that are sold in the outlet are fruits and veggies,snacks,frozen, household followed by diary
#  
#  **Most of the items are genrally low fat and regular types 
#  
#  **The lowest performing outelets are out10 and out19 
#  
#  **super market type 1 appear the most commonly seen outlet across the the locations 
# 

# # The End.
