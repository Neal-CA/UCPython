#!/usr/bin/env python
# coding: utf-8

# # **PCA and t-SNE Project: Auto MPG**
# 
# # **Marks: 30**
# 
# Welcome to the project on PCA and t-SNE. In this project, we will be using the **auto-mpg dataset**.
# 
# 
# -----------------------------
# ## **Context**
# -----------------------------
# 
# The shifting market conditions, globalization, cost pressure, and volatility are leading to a change in the automobile market landscape. The emergence of data, in conjunction with machine learning in automobile companies, has paved a way that is helping bring operational and business transformations.
# 
# The automobile market is vast and diverse, with numerous vehicle categories being manufactured and sold with varying configurations of attributes such as displacement, horsepower, and acceleration. We aim to find combinations of these features that can clearly distinguish certain groups of automobiles from others through this analysis, as this will inform other downstream processes for any organization aiming to sell each group of vehicles to a slightly different target audience.
# 
# You are a Data Scientist at SecondLife which is a leading used car dealership with numerous outlets across the US. Recently, they have started shifting their focus to vintage cars and have been diligently collecting data about all the vintage cars they have sold over the years. The Director of Operations at SecondLife wants to leverage the data to extract insights about the cars and find different groups of vintage cars to target the audience more efficiently.
# 
# -----------------------------
# ## **Objective**
# -----------------------------
# The objective of this problem is to **explore the data, reduce the number of features by using dimensionality reduction techniques like PCA and t-SNE, and extract meaningful insights**.
# 
# -----------------------------
# ## **Dataset** 
# -----------------------------
# There are 8 variables in the data: 
# 
# - mpg: miles per gallon
# - cyl: number of cylinders
# - disp: engine displacement (cu. inches) or engine size
# - hp: horsepower
# - wt: vehicle weight (lbs.)
# - acc: time taken to accelerate from 0 to 60 mph (sec.)
# - yr: model year
# - car name: car model name

# ## **Importing the necessary libraries and overview of the dataset**

# In[ ]:
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# ### **Loading the data**

# In[ ]:
df = pd.read_csv('auto-mpg.csv')

# ### **Data Overview**
# In[ ]:
#Displaying first few rows of the dataset
print(df.head())
# Checking the shape of the dataset
print(df.shape)

# - Observations
# - Sanity checks
# In[ ]:

# ## **Data Preprocessing and Exploratory Data Analysis**
# 
# 
# - EDA is an important part of any project involving data.
# - It is important to investigate and understand the data better before building a model with it.
# - A few questions have been mentioned below which will help you approach the analysis in the right manner and generate insights from the data.
# - Missing value treatment
# - Feature engineering (if needed)
# - Check the correlation among the variables
# - Outlier detection and treatment (if needed)
# - Preparing data for modeling
# - Any other preprocessing steps (if needed)


# In[ ]:
# Checking for missing values
print(df.isnull().sum())
# Checking the data types of the columns
print(df.dtypes)
# 
# ### **Summary Statistics**

# In[ ]:
#Summary Statistics
print(df.describe())

# CN: Added by Connor
# CN: Here's a function that will drop rows of a df that contain NaNs when coerced to a number, i.e. they're strings
# CN: Inputs are your dataframe, and the indices of the columns you want to check for non-numbers as a list
def drop_non_numeric_rows_by_index(df, column_indices):
    # CN:  Convert specified columns to numeric, coercing any errors to NaN
    for col_index in column_indices:
        col_name = df.columns[col_index]
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    # CN:  Identify those rows containing NaN values in the specified columns
    dropped_rows = df[df.isnull().any(axis=1)]

    # CN:  Drop those rows containing NaNs
    df_numeric = df.dropna(subset=df.columns[column_indices])
    
    # CN:  Return the cleaned dataset and a list of those rows we dropped so we can see what was lost
    return df_numeric, dropped_rows


# CN: Specify column numbers to check for non-numeric data
column_indices_to_check = [0, 1, 2, 3, 4, 5, 6]  # Replace with your column indices

# CN:  Drop rows with non-numeric data in specified columns by index
df_numeric, dropped_rows = drop_non_numeric_rows_by_index(df, column_indices_to_check)

# CN:  Print the rows that were dropped
print("Rows with non-numeric data:")
print(dropped_rows)

# **Observations:________**


# ### **Scaling the data**

# In[ ]:
# Scaling the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_numeric.drop(columns=['car name']))


# ## **Principal Component Analysis**

# In[ ]:
# Principal Component Analysis
pca = PCA()
pca_data = pca.fit_transform(scaled_features)

# Explained variance ratio
print(pca.explained_variance_ratio_)

# Plotting the explained variance
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, alpha=0.6, color='g')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.title('Explained Variance Ratio by PCA Components')
plt.show()



# **Observations:___________________**

# #### **Interpret the coefficients of the first three principal components from the below DataFrame**

# In[ ]:
# Interpreting the coefficients of the first three principal components
pca_components = pd.DataFrame(pca.components_, columns=df.drop(columns=['car name']).columns)
print(pca_components.head(3))

# **Observations:__________________**

# #### **Visualize the data in 2 dimensions using the first two principal components**
 
# In[ ]:
# Visualizing the data in 2 dimensions using the first two principal components
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA - First two principal components')
plt.show()

# **Observations:___________**

# ## **t-SNE**
# In[ ]:
# t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(scaled_features)

# **Observations:______________**

# #### **Visualize the clusters w.r.t different variables using scatter plot and box plot**

# In[ ]:

# Visualizing t-SNE in 2D
plt.figure(figsize=(8, 6))
sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1])
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('t-SNE Data Representation in 2D')
plt.show()


# **Observations:___________**

# ## **Actionable Insights and Recommendations**

# **write your insights and recommendations here:**  ______
