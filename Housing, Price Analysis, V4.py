# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:46:58 2020

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


#Dropping columns from the model with a value less than .2        
correlation_matrix = smallest_df.corr()
int_upper_filter = 0.4
int_lower_filter = -0.4
count = 0
for i in correlation_matrix["SalePrice"]:
    if  int_upper_filter > i > int_lower_filter:
        smallest_df = smallest_df.drop(correlation_matrix["SalePrice"]
                                       [correlation_matrix["SalePrice"]==i].index, axis = 1)


#Looping through all the categorical vaiables, determining the correlation value,
#and ranking the impact relative to each other
cat_columns = df.select_dtypes(include = ['object']).columns
int_columns = smallest_df.columns

#Using correlation bounds to filter non-important variables
upper_correlation_bound = 0.4
lower_correlation_bound = -0.4
categorical_ranking = {}
for i in cat_columns:
    encoded = pd.get_dummies(df[i])
    correlation_test = pd.concat([smallest_df, encoded], axis = 1, sort = False)
    correlation_test_encoded = correlation_test.corr()
    test = correlation_test_encoded["SalePrice"].sort_values(ascending=False)

        
    #Looping through encoded columns to determine if one of them exceeds correlation
    #bounds
    for j in encoded.columns:
        print("Column ", j)
        print("Value", test[j])
        if test[j] > upper_correlation_bound or test[j] < lower_correlation_bound:
            categorical_ranking[i] = test[j]




    
#Now applying machine learning model

'''
Approach will be to add the categorical variables one at a time and calculate the
RMSE for the training and the test set. The variable combination that yields the smallest
RMSE will be the winning combination
'''
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('std_scaler', StandardScaler())])


keys_by_importance = sorted(categorical_ranking.items(), key=lambda x: abs(x[1]), reverse=True)
test_df = smallest_df.copy()
train_error = list()
test_error = list()

for i in range(len(list(categorical_ranking.keys()))):

    test_df = pd.concat([test_df, pd.get_dummies(df[keys_by_importance[i][0]])], axis = 1)
    

    #Splitting data into test set and train set, and normalizing x variables
    y = test_df['SalePrice']
    x = test_df.drop(['SalePrice'], axis = 1)
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

    #Now transforming test data and evaluating accuracy of fitted model
    x_test = num_pipeline.fit_transform(x_test)

    price_prediction_test = linear_reg.predict(x_test)
    lin_mse = mean_squared_error(y_test, price_prediction_test)
    lin_rmse_test = np.sqrt(lin_mse)
    print("Test :", lin_rmse_test)
    train_error.append(lin_rmse)
    test_error.append(lin_rmse_test)
    

    
    
#Graphing results, need to insert axes break due to extreme outliers on test accuracy

figure, (ax, ax2) = plt.subplots(2, 1, sharex=True)
ax.plot(train_error)
ax.plot(test_error)
ax2.plot(train_error)
ax2.plot(test_error)

ax.set_ylim(40000, max(test_error))  # outliers only
ax2.set_ylim(30000, 40000)

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
#ax.yaxis.set_label_coords(1.05, -0.025)
ax2.set_xlabel("Number of Categorical variables")
ax.set_ylabel("RMSE")
ax2.legend( ('Test Error', 'Train Error'), loc='upper right', shadow=True)
ax.title.set_text("RMSE Versus Variables Added")
ax.yaxis.set_label_coords(-0.15, 0)
plt.show()





