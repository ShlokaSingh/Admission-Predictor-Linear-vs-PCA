#!/usr/bin/env python
# coding: utf-8

# # CS 556 Group Project - College Application

# Author:
# Shloka Brijesh Singh | CWID:20015697

# ## Problem Statement

# ![image-5.png](attachment:image-5.png)

# The dataset documents the test scores, school rankings, and application strengths of a series of college applicants, and correlates each with the probability that application is accepted to the school of their choice. Each of these features is provided as a continuous, qualitative numerical value, which means that we have the ability to process this data to review applications and learn what makes an application strong or not.

# Every application in the dataset contains the following features:
# 
# • Graduate Record Examinations (‘GRE Score‘): A score (out of 340) on the GREs.
# 
# • Test of English as a Foreign Language (‘TOEFL Score‘): A score (out of 120) on the TOEFL.
# 
# • University Rating (‘University Rating‘): A rank from 1 to 5 (with 5 being the best) of the university this entry describes application to.
# 
# • Statement of Purpose (‘SOP‘): A rank from 1 to 5 (in increments of 0.5) of the Statement of Purpose provided as part of the application.
# 
# • Letter of Recommendation (‘LOR‘): A rank from 1 to 5 (in increments of 0.5) of the Letter of Recommendation provided as part of the application.
# 
# • Undergraduate GPA (‘CGPA‘): Undergraduate college GPA, scaled here to be between 0 and 10, with 10 being the highest.
# 
# • Research Experience (‘Research‘): A boolean ‘0‘ or ‘1‘ value indicating whether the applicant has research experience.
# 
# • Chance of Admission (‘Chance of Admit‘): The chance (as a decimal probability between 0 and 1) that the application described in the previous data points will be accepted by the target university

# ## Task

# This project focuses on predicting the likelihood of student admission using machine learning models trained on real-world application data. The goal was to explore how well various features — such as test scores, GPA, and research experience — correlate with admission outcomes, and to evaluate how different modeling approaches perform on this task.
# 
# To start, I loaded and explored the dataset using Python in a Jupyter Notebook. I analyzed at least three key features for distribution patterns, presence of outliers, and potential correlation with the target variable (Chance of Admit). After preprocessing and understanding the data, I trained two types of models using an 80/20 train-test split:
# 
# A Linear Regression model
# A Support Vector Regression (SVR) model
# I then evaluated both using Mean Squared Error (MSE) as the primary metric.
# 
# As a second phase, I applied Principal Component Analysis (PCA) to reduce the dataset to two dimensions. A new regression model was trained on these principal components to assess whether dimensionality reduction could improve model performance. I also visualized the results using a 2D scatter plot, with data points in blue and the model’s decision boundary in black.
# 
# In the end, I compared both models — one trained on original features and the other on PCA-reduced features — to determine which was more effective and explain why based on performance metrics and visualization.

# # Our Approach

# ## Importing the required modules

# In[1]:


#Library pandas will be required to work with data in tabular representation.
#Library numpy will be required to round the data in the correlation matrix.
#Library missingnowill be required to visualize missing values in the data.
#Library matplotlib, seaborn, plotly required for data visualization.
#Library scipy will be required to test hypotheses.

import numpy as np # linear algebra....
import pandas as pd # data processing
from matplotlib import pyplot as plt #Visualization of the data

import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
import plotly.graph_objs as go

#Libraries to visualize the dataset
import matplotlib as mpl
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
from plotly import tools
from plotly.subplots import make_subplots
from plotly.offline import iplot

#Libraries for Hypothesis testing...
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import chi2_contingency
from scipy.stats import chi2

#Libraries for model creation....
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import metrics


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Loading the Dataset

# In[3]:


df = pd.read_csv("College_Admissions.csv" )


# ## Reading the Dataset

# In[4]:


df.head()


# In[ ]:





# ## Meta Information of the Dataset

# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.info()


# ## Deleating the unwanted Column

# In[8]:


del df['Serial No.']


# ## Distribution of each feature

# In[9]:


# quickly get distributions of each column (feature)
df.hist(bins=50, figsize=(20,15), color='#b8bd5c')


# In[10]:


#I have observed some spaces at column names...
df.columns = df.columns.str.replace(' ', '')
df.columns


# ## Checking for Outliers

# ## Basic Statistics

# In[11]:


round(df.describe())


# ## Dividing the features into Categorical and Numerical Features

# ## Finding the Unique Values

# In[12]:


''' Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell '''
from IPython.core.display import HTML

def multi_table(table_list):
    return HTML(
        '<table><tr style="background-color:white;">' + 
        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>')


# In[13]:


df_nunique = {var: pd.DataFrame(df[var].value_counts()) 
              for var in {'GREScore', 'TOEFLScore', 'UniversityRating', 'SOP', 'LOR','CGPA',
       'Research', 'ChanceofAdmit'}}
multi_table([df_nunique['GREScore'], df_nunique['TOEFLScore'],df_nunique['UniversityRating'] ,df_nunique['SOP'],df_nunique['LOR'],df_nunique['CGPA'],df_nunique['Research'],df_nunique['ChanceofAdmit']])


# ## Finding the Missing Data

# In[14]:


#PERCENTAGE OF THE MISSING VALUES - DATAFRAME..... 
def missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    Percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, Percentage], axis=1, keys=['Total', 'Percentage'])
missing_data(df)


# ## Checking if any Duplicate Data

# In[15]:


df[df.duplicated()]


# ## Univariate Analysis

# In[16]:


"""This function takes in a dataframe and a column and finds the percentage of the value_counts"""
def percent_value_counts(df, feature):
    percent = pd.DataFrame(round(df.loc[:,feature].value_counts(dropna=False, normalize=True)*100,2))
    ## creating a df with th
    total = pd.DataFrame(df.loc[:,feature].value_counts(dropna=False))
    ## concating percent and total dataframe

    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis = 1)


# In[17]:


percent_value_counts(df, 'GREScore')


# In[18]:


fig = make_subplots(rows=1,cols=1)
fig.add_trace(go.Bar(y = df['GREScore'].value_counts().values.tolist(), 
                      x = df['GREScore'].value_counts().index, 
                      text=df['GREScore'].value_counts().values.tolist(),
              textfont=dict(size=15),
                      textposition = 'outside',
                      showlegend=False,
              marker = dict(color = '#b8bd5c',
                            line_width=3)),row = 1,col = 1)

fig.update_layout(title='Density distribution of GRE Score',
                  template='plotly_white')
fig.update_yaxes(range=[0,25])
iplot(fig)


# By analysisng the above distribution we can say that the highest number of students were scoring GRE score of 312 and 324. And majority have scored between 305 and 330

# In[19]:


percent_value_counts(df, 'TOEFLScore')


# In[20]:


fig = make_subplots(rows=1,cols=1)
fig.add_trace(go.Bar(y = df['TOEFLScore'].value_counts().values.tolist(), 
                      x = df['TOEFLScore'].value_counts().index, 
                      text=df['TOEFLScore'].value_counts().values.tolist(),
              textfont=dict(size=15),
                      textposition = 'outside',
                      showlegend=False,
              marker = dict(color = '#b8bd5c',
                            line_width=3)),row = 1,col = 1)

fig.update_layout(title='Density distribution of TOEFL Score',
                  template='plotly_white')
fig.update_yaxes(range=[0,48])
iplot(fig)


# By analysisng the above distribution we can say that the highest number of students were scoring TOEFL score of 105 and 110. And majority have scored between 99 and 115

# In[21]:


percent_value_counts(df, 'UniversityRating')


# In[22]:


fig = make_subplots(rows=1,cols=1)
fig.add_trace(go.Bar(y = df['UniversityRating'].value_counts().values.tolist(), 
                      x = df['UniversityRating'].value_counts().index, 
                      text=df['UniversityRating'].value_counts().values.tolist(),
              textfont=dict(size=15),
                      textposition = 'outside',
                      showlegend=False,
              marker = dict(color = '#b8bd5c',
                            line_width=3)),row = 1,col = 1)

fig.update_layout(title='Density distribution of University Rating',
                  template='plotly_white')
fig.update_yaxes(range=[0,200])
iplot(fig)


# We can see from the distribution that highest number of the University rating is 2 and 3

# In[23]:


percent_value_counts(df, 'SOP')


# In[24]:


fig = make_subplots(rows=1,cols=1)
fig.add_trace(go.Bar(y = df['SOP'].value_counts().values.tolist(), 
                      x = df['SOP'].value_counts().index, 
                      text=df['SOP'].value_counts().values.tolist(),
              textfont=dict(size=15),
                      textposition = 'outside',
                      showlegend=False,
              marker = dict(color = '#b8bd5c',
                            line_width=3)),row = 1,col = 1)

fig.update_layout(title='Density distribution of SOP',
                  template='plotly_white')
fig.update_yaxes(range=[0,100])
iplot(fig)


# By analyzing the distribution of SOP, we can say that highest number of the Statement of purpose are rated between 3 and 4.5

# In[25]:


percent_value_counts(df, 'LOR')


# In[26]:


fig = make_subplots(rows=1,cols=1)
fig.add_trace(go.Bar(y = df['LOR'].value_counts().values.tolist(), 
                      x = df['LOR'].value_counts().index, 
                      text=df['LOR'].value_counts().values.tolist(),
              textfont=dict(size=15),
                      textposition = 'outside',
                      showlegend=False,
              marker = dict(color = '#b8bd5c',
                            line_width=3)),row = 1,col = 1)

fig.update_layout(title='Density distribution of LOR',
                  template='plotly_white')
fig.update_yaxes(range=[0,100])
iplot(fig)


# By analyzing the distribution of LOR, we can say that highest number of the LOR rating is between 2.5 and 4.5

# In[27]:


percent_value_counts(df, 'Research')


# In[28]:


fig = make_subplots(rows=1,cols=1)
fig.add_trace(go.Bar(y = df['Research'].value_counts().values.tolist(), 
                      x = df['Research'].value_counts().index, 
                      text=df['Research'].value_counts().values.tolist(),
              textfont=dict(size=15),
                      textposition = 'outside',
                      showlegend=False,
              marker = dict(color = '#b8bd5c',
                            line_width=3)),row = 1,col = 1)

fig.update_layout(title='Density distribution of "Research"',
                  template='plotly_white')
fig.update_yaxes(range=[0,300])
iplot(fig)


# By analyzing the above distribution we can say that maximum students have 1 Research

# In[29]:


percent_value_counts(df, 'ChanceofAdmit')


# In[30]:


fig = make_subplots(rows=1,cols=1)
fig.add_trace(go.Bar(y = df['ChanceofAdmit'].value_counts().values.tolist(), 
                      x = df['ChanceofAdmit'].value_counts().index, 
                      text=df['ChanceofAdmit'].value_counts().values.tolist(),
              textfont=dict(size=15),
                      textposition = 'outside',
                      showlegend=False,
              marker = dict(color = '#b8bd5c',
                            line_width=3)),row = 1,col = 1)

fig.update_layout(title='Density distribution of "Chance of Admit"',
                  template='plotly_white')
fig.update_yaxes(range=[0,25])
iplot(fig)


# By analyzing the above distribution we can say that the highest percentage of getting admission is between 0.61 and 0.96

# ## Bivariate Analysis

# ### GRE Score vs Chance of Admit

# In[31]:


fig = px.scatter(df, x="GREScore", y="ChanceofAdmit",trendline="ols",trendline_color_override = '#000000',width=1000, height=400)
fig.update_traces(marker_size=12,marker_color='#b8bd5c')
fig.update_layout(title='GREScore Vs ChanceofAdmit',
                  template='plotly_white')
fig.show()


# From the above graph we can see that Chance of Admit is highly related to GRE Score

# ### TOEFL Score vs Chance of Admit

# In[32]:


fig = px.scatter(df, x="TOEFLScore", y="ChanceofAdmit", trendline="ols",trendline_color_override = '#000000',width=1000, height=400)
fig.update_layout(title='TOEFLScore Vs ChanceofAdmit',
                  template='plotly_white')
fig.update_traces(marker_size=12,marker_color='#b8bd5c')
fig.show()


# From the above graph we can see that Chance of Admit is highly related to TOEFL Score

# ### University Ranking vs Chance of Admit

# In[33]:


fig = px.scatter(df, x="UniversityRating", y="ChanceofAdmit", trendline="ols",trendline_color_override = '#000000',width=1000, height=400)
fig.update_traces(marker_size=12,marker_color='#b8bd5c')
fig.update_layout(title='UniversityRating Vs ChanceofAdmit',
                  template='plotly_white')
fig.show()


# From the above graph we can see that Chance of Admit is not related to University Rancking

# ### SOP vs Chance of Admit

# In[34]:


fig = px.scatter(df, x="SOP", y="ChanceofAdmit", trendline="ols",trendline_color_override = '#000000',width=1000, height=400)
fig.update_layout(title='SOP Vs ChanceofAdmit',
                  template='plotly_white')
fig.update_traces(marker_size=12,marker_color='#b8bd5c')
fig.show()


# From the above graph we can see that Chance of Admit is not related to SOP

# ### LOR vs Chance of Admit

# In[35]:


fig = px.scatter(df, x="LOR", y="ChanceofAdmit", trendline="ols",trendline_color_override = '#000000',width=1000, height=400)
fig.update_traces(marker_size=12,marker_color='#b8bd5c')
fig.update_layout(title='LOR Vs ChanceofAdmit',
                  template='plotly_white')
fig.show()


# From the above graph we can see that Chance of Admit is not related to LOR

# ### CGPA vs Chance of Admit

# In[36]:


fig = px.scatter(df, x="CGPA", y="ChanceofAdmit", trendline="ols",trendline_color_override = '#000000',width=1000, height=400)
fig.update_traces(marker_size=12,marker_color='#b8bd5c')
fig.update_layout(title='CGPA Vs ChanceofAdmit',
                  template='plotly_white')
fig.show()


# From the above graph we can see that Chance of Admit is highly related to CGPA

# ### Research vs Chance of Admit

# In[37]:


fig = px.scatter(df, x="Research", y="ChanceofAdmit", trendline="ols",trendline_color_override = '#000000',width=1000, height=400)
fig.update_traces(marker_size=12,marker_color='#b8bd5c')
fig.update_layout(title='Research Vs ChanceofAdmit',
                  template='plotly_white')
fig.show()


# In[38]:


sns.pairplot(df)


# From the above graph we can see that Chance of Admit is not related to Research

# In[39]:


fig = plt.figure(figsize=(8, 8))
sns.heatmap(df.corr(), annot=True)
fig.show()
print("Correlation in a nutshell: ")


# In[40]:


df.head()


# In[ ]:





# In[41]:


features =['CGPA','GREScore','TOEFLScore']


# In[42]:


X= df[features]
y=df['ChanceofAdmit']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# ## Linear Regression

# In[43]:


model = LinearRegression() 
model.fit(X_train, y_train)


# In[44]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Square Error:", mse)


# In[45]:


prediction = pd.DataFrame(y_pred, columns=['Predicted_Admit']).to_csv('LR_Prediction.csv')


# In[46]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)


# In[47]:


X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model_pca = LinearRegression() # or SVM()
model_pca.fit(X_train_pca, y_train_pca)


# In[48]:


y_pred_pca = model_pca.predict(X_test_pca)
mse_pca = mean_squared_error(y_test_pca, y_pred_pca)
print("Mean Square Error:", mse_pca)


# In[49]:


prediction = pd.DataFrame(y_pred, columns=['Predicted_Admit']).to_csv('LR_Prediction_PCA.csv')


# ## Support Vector Machine

# In[50]:


model = SVR()
model.fit(X_train, y_train)


# In[51]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Square Error:", mse)


# In[52]:


prediction = pd.DataFrame(y_pred, columns=['Predicted_Admit']).to_csv('SVM_Prediction.csv')


# In[53]:


pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)


# In[54]:


X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model_pca = SVR()
model_pca.fit(X_train_pca, y_train_pca)


# In[55]:


y_pred_pca = model_pca.predict(X_test_pca)
mse_pca = mean_squared_error(y_test_pca, y_pred_pca)
print("Mean Square Error:",mse_pca)


# In[56]:


prediction = pd.DataFrame(y_pred, columns=['Predicted_Admit']).to_csv('SVM_Prediction_PCA.csv')


# ## Decision Tree

# In[57]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[58]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Square Error:", mse)


# In[59]:


prediction = pd.DataFrame(y_pred, columns=['Predicted_Admit']).to_csv('DT_Prediction.csv')


# In[60]:


pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)


# In[61]:


X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model_pca = DecisionTreeRegressor()
model_pca.fit(X_train_pca, y_train_pca)


# In[62]:


y_pred_pca = model_pca.predict(X_test_pca)
mse_pca = mean_squared_error(y_test_pca, y_pred_pca)
print("Mean Square Error :",mse_pca)


# In[63]:


prediction = pd.DataFrame(y_pred, columns=['Predicted_Admit']).to_csv('DT_Prediction_PCA.csv')

