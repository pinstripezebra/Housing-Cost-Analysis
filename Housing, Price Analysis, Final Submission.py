# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 15:05:15 2020

@author: seelc
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

'''
Key goal is to perform analysis on categorical variables and determine which
have significant impact on price
'''

final_df = pd.read_csv("C:\\Users\\seelc\\OneDrive\\Desktop\\Projects\\Housing Prices Kaggle\\test.csv")


df = pd.read_csv("C:\\Users\\seelc\\OneDrive\\Desktop\\Projects\\Housing Prices Kaggle\\train.csv")

#Cleaning data:
#nan lot frontage should really be 0
df["LotFrontage"].fillna(0, inplace = True)


#Dropping all columns except numeric:
#In V2 changing this to encoding just one of the categorical variables
smallest_df = df.copy(deep = True)
smallest_df = smallest_df.fillna(0)
df.select_dtypes(exclude = ['object'])
smallest_df = smallest_df.select_dtypes(exclude = ['object'])
df_with_cat = smallest_df.copy()

#doing same thing with final dataset
smallest_df_final = final_df.copy(deep = True)
smallest_df_final = smallest_df_final.fillna(0)
smallest_df_final = smallest_df_final.select_dtypes(exclude = ['object'])



#Dropping columns from the model with a value less than .2        
correlation_matrix = smallest_df.corr()
int_upper_filter = 0.4
int_lower_filter = -0.4
count = 0
for i in correlation_matrix["SalePrice"]:
    if  int_upper_filter > i > int_lower_filter:
        smallest_df = smallest_df.drop(correlation_matrix["SalePrice"]
                                       [correlation_matrix["SalePrice"]==i].index, axis = 1)
        #Doing same thing to final dataset
        smallest_df_final = smallest_df_final.drop(correlation_matrix["SalePrice"]
                                       [correlation_matrix["SalePrice"]==i].index, axis = 1)


categorical_to_add = ['ExterQual', 'KitchenQual', 'Foundation', 'BsmtQual']
correlation_test = smallest_df
for i in categorical_to_add:
    encoded = pd.get_dummies(df[i])
    correlation_test = pd.concat([smallest_df, encoded], axis = 1, sort = False)
    
    #Doing same thing with final dataset
    final_encoded = pd.get_dummies(final_df[i])
    final_correlation = pd.concat([smallest_df_final, final_encoded], axis = 1, sort = False)
    

#Now applying machine learning
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('std_scaler', StandardScaler())])


y = correlation_test['SalePrice']
x = correlation_test.drop(['SalePrice'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Now transforming x_train data and fitting to model
x_train = num_pipeline.fit_transform(x_train)
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)
    
#Now checking the accuracy of the model
price_prediction = linear_reg.predict(x_train)
lin_mse = mean_squared_error(y_train, price_prediction)
lin_rmse = np.sqrt(lin_mse)
print("Train :", lin_rmse)


x_test = num_pipeline.fit_transform(x_test)

price_prediction_test = linear_reg.predict(x_test)
lin_mse = mean_squared_error(y_test, price_prediction_test)
lin_rmse_test = np.sqrt(lin_mse)
print("Test :", lin_rmse_test)


#Now running on actual test dataset to generate outputs for submission
x_test_final = num_pipeline.fit_transform(final_correlation)
price_prediction_final = linear_reg.predict(x_test_final)

#Now writing results to excel
pd.DataFrame([final_df['Id'],price_prediction_final]).to_excel("For Submission.xlsx")







