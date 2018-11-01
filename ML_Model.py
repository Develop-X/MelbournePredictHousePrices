# -*- coding: utf-8 -*-
"""
@author: ady.kalra

Pandas [Python Data Analysis Library] - head(), info(), crosstab(), mean(), corr(), 
Matplot - subplots(), show(), 
Seaborn - regplot(), heatmap(), 

"""

import pandas as varpanda
import numpy as varnumpy
import seaborn as varseaborn
import matplotlib.pyplot as varmatplt

# read Dataset from csv file
melbourne_file_path = 'Raw_Data/melHousingData.csv'
melbourne_data = varpanda.read_csv(melbourne_file_path)
melbourne_data.head()


melbourne_data.info()

# data washing / cleaning
clean_data = melbourne_data[melbourne_data.Price.notnull()]
clean_data = clean_data[clean_data.BuildingArea.notnull()]
clean_data = clean_data[clean_data.Car.notnull()]
clean_data = clean_data[clean_data.Landsize.notnull()]
clean_data = clean_data[clean_data.Lattitude.notnull()]
clean_data = clean_data[clean_data.Longtitude.notnull()]
clean_data.info()

# do the number of bedrooms have a relationship to price?
print(varpanda.crosstab(melbourne_data['Price'].mean(), melbourne_data['Bedroom2']))
print('\n')
f, ax = varmatplt.subplots(figsize=(5, 5))
varseaborn.regplot(data=clean_data, x='Rooms', y='Price')
varmatplt.show()

# Although weak, it appears that there seems to be a positive relationship. Let's see what is the actual correlation between price and the other data points. We will look at this in 2 ways heatman for visualization and the correlation coefficient score.

f, ax = varmatplt.subplots(figsize=(5, 5))
corrmat = clean_data.corr()
varseaborn.heatmap(corrmat, vmax=.8, square=True)
varmatplt.show()
corrmat

# Distribution of the data
varmatplt.rcParams['figure.figsize'] = 16, 16
clean_data.loc[:,:].hist(bins=100)
varmatplt.show()

# Create new features: Let's create new features to see if these new features will have a stronger correlation coefficient score than the original. We will do so by mixing the data and altering the data.
clean_data['Roomssq'] = clean_data.Rooms ** 2
clean_data['Roomssqrt'] = clean_data.Rooms ** (1/2)
clean_data['Plus'] = clean_data.Rooms + clean_data.Bedroom2 + clean_data.Bathroom
clean_data['Prod'] = clean_data.Rooms * clean_data.Bedroom2 * clean_data.Bathroom
clean_data['year'] = (2017 - clean_data.YearBuilt)
clean_data['yearsq'] = (2017 - clean_data.YearBuilt) ** 2
clean_data['yearsqrt'] = (2017 - clean_data.YearBuilt) ** (1/2)

f, ax = varmatplt.subplots(figsize=(5, 5))
corrmat = clean_data.corr()
varseaborn.heatmap(corrmat, vmax=.8, square=True)
varmatplt.show()
corrmat
# It appears that the column "Plus" has a correlation score of .529 which is the highest correlation score out of all the features new and old. This new feature was created by rooms, bedrooms and bathrooms.


# Modelling
# Decision Tree Regression

# Let's drop the weakest scores and run our first model "Decision Tree Regressor"
X = clean_data.drop(['YearBuilt', 'year', 'yearsq', 'yearsqrt', 'Price',], axis=1)
Y = clean_data.Price
X = varpanda.get_dummies(data=X)


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

dtr = DecisionTreeRegressor(max_features=10, max_depth=10)
dtr.fit(X, Y)
print(cross_val_score(dtr, X, Y, cv=5))

# [-0.04384913  0.00482545  0.00621629 -0.04248696 -0.00801812]
# The scores above is bad...Let's take a look at why it's bad using Root Mean Square Error/ Deviation

predicted = dtr.predict(X)
residual = Y - predicted

fig = varmatplt.figure(figsize=(30,30))
ax1 = varmatplt.subplot(211)
varseaborn.distplot(residual, color ='orange')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.title('Residual counts',fontsize=35)
varmatplt.xlabel('Residual',fontsize=25)
varmatplt.ylabel('Count',fontsize=25)

ax2 = varmatplt.subplot(212)
varmatplt.scatter(predicted, residual, color ='orange')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.xlabel('Predicted',fontsize=25)
varmatplt.ylabel('Residual',fontsize=25)
varmatplt.axhline(y=0)
varmatplt.title('Residual vs. Predicted',fontsize=35)

varmatplt.show()

from sklearn.metrics import mean_squared_error
rmse = varnumpy.sqrt(mean_squared_error(Y, predicted))
print('RMSE:')
print(rmse)


# RMSE: 677273.379164

# The root-mean-square deviation (RMSD) or root-mean-square error (RMSE) (or sometimes root-mean-squared error) is a frequently used measure of the differences between values (sample and population values) predicted by a model or an estimator and the values actually observed. The RMSD represents the sample standard deviation of the differences between predicted values and observed values. These individual differences are called residuals when the calculations are performed over the data sample that was used for estimation, and are called prediction errors when computed out-of-sample. The RMSD serves to aggregate the magnitudes of the errors in predictions for various times into a single measure of predictive power. RMSD is a measure of accuracy, to compare forecasting errors of different models for a particular data and not between datasets, as it is scale-dependent.

# The higher the RMSE the worst our predicting model is and the reason why the RMSE score is high and bad is because it appears this model picked a single price point for the categories that the model selected this is shown in the graphs above.

# Let's take a different approach...
# and think like a real estate agents. As real estate agents we should look the comparables see what a house like what we're trying to predict is priced based on sales of similar homes. Let's use the next model "Nearest Neighbors" and use the 2 nearnest neighbors.

# Modelling
# Nearest Neighbors

from sklearn import neighbors

knn = neighbors.KNeighborsRegressor(n_neighbors=2)
knn.fit(X, Y)

predicted = knn.predict(X)
residual = Y - predicted

fig = varmatplt.figure(figsize=(30,30))
ax1 = varmatplt.subplot(211)
varseaborn.distplot(residual, color ='orange')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.title('Residual counts',fontsize=35)
varmatplt.xlabel('Residual',fontsize=25)
varmatplt.ylabel('Count',fontsize=25)

ax2 = varmatplt.subplot(212)
varmatplt.scatter(predicted, residual, color ='blue')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.xlabel('Predicted',fontsize=25)
varmatplt.ylabel('Residual',fontsize=25)
varmatplt.axhline(y=0)
varmatplt.title('Residual vs. Predicted',fontsize=35)

varmatplt.show()

from sklearn.metrics import mean_squared_error
rmse = varnumpy.sqrt(mean_squared_error(Y, predicted))
print('RMSE:')
print(rmse)

# RMSE:
# 246010.726124
# By looking at the RMSE score this we've vast improvements, as you can see from the "Residual vs. Predicted" that the predicted score is closer to zero and is tighter around the lines which means that we are guessing alot closer to the price.

# What if...
# the 2 nearest neighbors is not the optimal number. Lets see if we can find the optimal number of neighbors.

rmse_l = []
num = []
for n in range(2, 16):
    knn = neighbors.KNeighborsRegressor(n_neighbors=n)
    knn.fit(X, Y)
    predicted = knn.predict(X)
    rmse_l.append(varnumpy.sqrt(mean_squared_error(Y, predicted)))
    num.append(n)
	
df_plt = varpanda.DataFrame()
df_plt['rmse'] = rmse_l
df_plt['n_neighbors'] = num
ax = varmatplt.figure(figsize=(15,7))
varseaborn.barplot(data = df_plt, x = 'n_neighbors', y = 'rmse')
varmatplt.show()

print(rmse_l)

# [246010.72612387789, 299140.89928988385, 331499.77892903792, 347367.56599903584, 363610.28713245952, 373641.28252136859, 382776.00220798928, 390369.98595582315, 399132.94385391136, 404725.34418166213, 410064.63836764981, 414764.42545172054, 419022.88172154903, 423461.00056615419]

# It appears that 2 nearest neighbors is the optimal number of neighbors. This is evidenced by the increasing RMSE as we increase the number of neighbors.

# Modelling - Lasso & Ridge
# Let's look at a model "Lasso" & "Ridge", these models will penalize the model for larger coefficients. So these models will regularizes the way it predicts the price. Lets try a low alpha first and see how it performs.

from sklearn import linear_model
lass = linear_model.Lasso(alpha = .025)
lass.fit (X, Y)

predicted = lass.predict(X)
residual = Y - predicted

fig = varmatplt.figure(figsize=(30,30))
ax1 = varmatplt.subplot(211)
varseaborn.distplot(residual, color ='orange')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.title('Residual counts',fontsize=35)
varmatplt.xlabel('Residual',fontsize=25)
varmatplt.ylabel('Count',fontsize=25)

ax2 = varmatplt.subplot(212)
varmatplt.scatter(predicted, residual, color ='orange')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.xlabel('Predicted',fontsize=25)
varmatplt.ylabel('Residual',fontsize=25)
varmatplt.axhline(y=0)
varmatplt.title('Residual vs. Predicted',fontsize=35)

varmatplt.show()

from sklearn.metrics import mean_squared_error
rmse = varnumpy.sqrt(mean_squared_error(Y, predicted))
print('RMSE:')
print(rmse)

# RMSE: 3127.28332239

alp = .1
for n in range (0,3):
    lass = linear_model.Lasso(alpha = alp)
    lass.fit (X, Y)
    predicted = lass.predict(X)
    rmse = varnumpy.sqrt(mean_squared_error(Y, predicted))
    alp = alp + .1
    print('RMSE:')
    print(rmse)
	

# RMSE: 5051.22727146
# RMSE: 6992.45802529
# RMSE: 8405.56195252


rid = linear_model.Ridge(alpha = .025)
rid.fit (X, Y)

predicted = rid.predict(X)
residual = Y - predicted

fig = varmatplt.figure(figsize=(30,30))
ax1 = varmatplt.subplot(211)
varseaborn.distplot(residual, color ='orange')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.title('Residual counts',fontsize=35)
varmatplt.xlabel('Residual',fontsize=25)
varmatplt.ylabel('Count',fontsize=25)

ax2 = varmatplt.subplot(212)
varmatplt.scatter(predicted, residual, color ='orange')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.xlabel('Predicted',fontsize=25)
varmatplt.ylabel('Residual',fontsize=25)
varmatplt.axhline(y=0)
varmatplt.title('Residual vs. Predicted',fontsize=35)

varmatplt.show()

from sklearn.metrics import mean_squared_error
rmse = varnumpy.sqrt(mean_squared_error(Y, predicted))
print('RMSE:')
print(rmse)

# RMSE: 14735.7430427

lass = linear_model.Lasso(alpha = .025)
lass.fit (X, Y)

print('\nR² for the model with many features:')
print(lass.score(X, Y))
origparams = varnumpy.append(lass.coef_, lass.intercept_)
print('\nParameter features:')
print(origparams)

print('\nCross Validation Score:')
print(cross_val_score(lass, X, Y, cv=3))

# R² for the model with many features:
# 0.999978896199
# Parameter features:
# [  2.78474750e+05  -3.18044880e+04   1.21400677e+03 ...,  -5.06936783e+03
#    8.31002277e+03  -1.13875073e+08]
# Cross Validation Score:
# [ 0.55646913  0.50086776  0.316526  ]

print('\nR² for the model with many features:')
print(rid.score(X, Y))
origparams_rid = varnumpy.append(rid.coef_, rid.intercept_)
print('\nParameter features:')
print(origparams_rid)

print('\nCross Validation Score:')
print(cross_val_score(rid, X, Y, cv=3))

knn = neighbors.KNeighborsRegressor(n_neighbors=2)
knn.fit(X, Y)

print(cross_val_score(knn, X, Y, cv=3))

# Lasso & Ridge Summary:
 # It appears that a low alpha created a low RMSE score however when we crossvalidated the scores it appears that the model is overfitting, and this is evidenced by the graph with alot of the prices really tight along the line of zero.

# Working with more data:
# Since working with a reduced amount of data may skew our models predicting ability let's attempt to keep as much of the data as possible.

addNulls_clean_data = melbourne_data[melbourne_data.Price.notnull()]
#addNulls_clean_data = addNulls_clean_data[addNulls_clean_data.BuildingArea.notnull()]
addNulls_clean_data = addNulls_clean_data[addNulls_clean_data.Car.notnull()]
#addNulls_clean_data = addNulls_clean_data[addNulls_clean_data.Landsize.notnull()]
#addNulls_clean_data = addNulls_clean_data[addNulls_clean_data.Lattitude.notnull()]
#addNulls_clean_data = addNulls_clean_data[addNulls_clean_data.Longtitude.notnull()]
addNulls_clean_data.info()

addNulls_clean_data = addNulls_clean_data.drop(['BuildingArea', 'YearBuilt', 'Landsize', 'Lattitude', 'Longtitude'], axis=1)
addNulls_clean_data.info()

# Alright we kept about 60% of the original data. Please note that features above that were dropped were due to low correlation coefficient scores.
# Let's take feature engineering to a new level:
# Lets create markers of min, mean and max price of:

# Suburb
# Type (of property)
# Method
# Seller (Sales Agent)
# Council Area
# Region Name
# By creating these markers, we are hoping this will give the model a better idea that certain area, types, sales agent, etc.. will be a better indicator of price points.

temp = addNulls_clean_data.groupby('Suburb').agg({'min', 'mean', 'max'})
temp2 = temp['Price']
temp2 = temp2.reset_index()
temp2.columns = ['Suburb', 'max_sub_id', 'min_sub_id', 'mean_sub_id']
print(temp2.info())
temp2.head()

f, ax = varmatplt.subplots(figsize=(15, 60))
varseaborn.boxplot(data = addNulls_clean_data, x='Price', y='Suburb')
varmatplt.show()

df_copy = addNulls_clean_data
df_copy = varpanda.merge(df_copy, temp2, on='Suburb', how='left')
df_copy.head()

type_g = addNulls_clean_data.groupby('Type').agg({'min', 'mean', 'max'})
temp3 = type_g['Price']
temp3 = temp3.reset_index()
temp3.columns = ['Type', 'max_t_id', 'min_t_id', 'mean_t_id']
temp3

f, ax = varmatplt.subplots(figsize=(15, 10))
varseaborn.stripplot(data = addNulls_clean_data, x='Type', y='Price', jitter=.5)
varmatplt.show()
 
df_copy = varpanda.merge(df_copy, temp3, on='Type', how='left')
df_copy.head()

method = addNulls_clean_data.groupby('Method').agg({'min', 'mean', 'max'})
temp4 = method['Price']
temp4 = temp4.reset_index()
temp4.columns = ['Method', 'max_m_id', 'min_m_id', 'mean_m_id']
temp4


f, ax = varmatplt.subplots(figsize=(15, 5))
varseaborn.violinplot(data = addNulls_clean_data, x='Price', y='Method', jitter=.5)
varmatplt.show()

df_copy = varpanda.merge(df_copy, temp4, on='Method', how='left')
df_copy.head()

sellerg = addNulls_clean_data.groupby('SellerG').agg({'min', 'mean', 'max'})
temp5 = sellerg['Price']
temp5 = temp5.reset_index()
temp5.columns = ['SellerG', 'max_s_id', 'min_s_id', 'mean_s_id']
print(temp5.info())
temp5.head()

f, ax = varmatplt.subplots(figsize=(15, 60))
varseaborn.stripplot(data = addNulls_clean_data, x='Price', y='SellerG', jitter=.1)
varmatplt.show()

df_copy = varpanda.merge(df_copy, temp5, on='SellerG', how='left')
df_copy.head()

council = addNulls_clean_data.groupby('CouncilArea').agg({'min', 'mean', 'max'})
temp6 = council['Price']
temp6 = temp6.reset_index()
temp6.columns = ['CouncilArea', 'max_c_id', 'min_c_id', 'mean_c_id']
print(temp6.info())
temp6.head()

f, ax = varmatplt.subplots(figsize=(15, 5))
varseaborn.boxplot(data = addNulls_clean_data, x='CouncilArea', y='Price')
varmatplt.xticks(rotation='vertical')
varmatplt.show()

#df_copy = df_copy.drop(['max_c_id_y', 'min_c_id_y', 'mean_c_id_y'], axis=1)

df_copy = varpanda.merge(df_copy, temp6, on='CouncilArea', how='left')
df_copy.head()

region = addNulls_clean_data.groupby('Regionname').agg({'min', 'mean', 'max'})
temp7 = region['Price']
temp7 = temp7.reset_index()
temp7.columns = ['Regionname', 'max_r_id', 'min_r_id', 'mean_r_id']
print(temp7.info())
temp7.head()

f, ax = varmatplt.subplots(figsize=(15, 10))
varseaborn.boxplot(data = addNulls_clean_data, x='Price', y='Regionname')
#varmatplt.xticks(rotation='vertical')
varmatplt.show()

df_copy = varpanda.merge(df_copy, temp7, on='Regionname', how='left')
df_copy.head()

df_copy['date_m'], df_copy['date_d'], df_copy['date_y'] = df_copy['Date'].str.split('/', 2).str
df_copy.head()

# Does the "when you sell your home" matter?
df_copy['date_m'] = df_copy['date_m'].astype(int)

f, ax = varmatplt.subplots(figsize=(15, 5))
varseaborn.stripplot(data = df_copy, x='date_m', y='Price', jitter=.25)
varmatplt.show()

df_copy['date_d'] = df_copy['date_d'].astype(int)
f, ax = varmatplt.subplots(figsize=(15, 10))
varseaborn.boxplot(data = df_copy, x='date_d', y='Price')
varmatplt.show()

df_copy['date_y'] = df_copy['date_y'].astype(int)
f, ax = varmatplt.subplots(figsize=(7, 10))
varseaborn.violinplot(data = df_copy, x='date_y', y='Price')
varmatplt.show()

# Lets see how many features we now have:
df_copy.info()

df_copy = df_copy.drop(['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Date', 'CouncilArea', 'Regionname'], axis=1)
df_copy.info()

# Let's see how our new features correlate to the price:
f, ax = varmatplt.subplots(figsize=(10, 10))
corrmat = df_copy.corr()
varseaborn.heatmap(corrmat, vmax=.8, square=True)
varmatplt.show()
corrmat

# WOW!!! If you look at max sub id, this has a .575 score to price. This is really good, lets see how our new features fare against the same models we used. Let's revisit Decision Tree Regressor.
X = df_copy.drop('Price', axis=1)
Y = df_copy.Price

scores = []
depth = []
for n in range(2, 15):
    dtr = DecisionTreeRegressor(max_features=15, max_depth=n)
    dtr.fit(X, Y)
    scores.append(cross_val_score(dtr, X, Y, cv=12).mean())
    depth.append(n)
	

plt_dtr = varpanda.DataFrame()

plt_dtr['mean_scores'] = scores
plt_dtr['depth'] = depth

f, ax = varmatplt.subplots(figsize=(15, 5))
varseaborn.barplot(data = plt_dtr, x='depth', y='mean_scores')
varmatplt.show()

# The above bar graph represent number of max depth on the x-axis that the model will limit itself to and on the y-axis is the mean scores of the cross validation of 12 folds. As you can see even the worst mean score of approximately 4, this model guesses way better due to having more information.

# Let's try "Nearest Neighbors" once more:

knn = neighbors.KNeighborsRegressor(n_neighbors=2)
knn.fit(X, Y)

print(cross_val_score(knn, X, Y, cv=12))

knn = neighbors.KNeighborsRegressor(n_neighbors=2)
knn.fit(X, Y)

predicted = knn.predict(X)
residual = Y - predicted

fig = varmatplt.figure(figsize=(30,30))
ax1 = varmatplt.subplot(211)
varseaborn.distplot(residual, color ='purple')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.title('Residual counts',fontsize=35)
varmatplt.xlabel('Residual',fontsize=25)
varmatplt.ylabel('Count',fontsize=25)

ax2 = varmatplt.subplot(212)
varmatplt.scatter(predicted, residual, color ='purple')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.xlabel('Predicted',fontsize=25)
varmatplt.ylabel('Residual',fontsize=25)
varmatplt.axhline(y=0)
varmatplt.title('Residual vs. Predicted',fontsize=35)

varmatplt.show()

from sklearn.metrics import mean_squared_error
rmse = varnumpy.sqrt(mean_squared_error(Y, predicted))
print('RMSE:')
print(rmse)

# RMSE: 264179.577229
# It appears that the RMSE is higher this time around, but I believe this is due to higher priced homes does not follow the normal pricing pattern. It seems that $2,000,000 would be the threshold for home prices.
# Lets give "Lasso & Ridge" another shot:

lass = linear_model.Lasso(alpha = 1.15e8)
lass.fit (X, Y)

predicted = lass.predict(X)
residual = Y - predicted

fig = varmatplt.figure(figsize=(30,30))
ax1 = varmatplt.subplot(211)
varseaborn.distplot(residual, color ='pink')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.title('Residual counts',fontsize=35)
varmatplt.xlabel('Residual',fontsize=25)
varmatplt.ylabel('Count',fontsize=25)

ax2 = varmatplt.subplot(212)
varmatplt.scatter(predicted, residual, color ='pink')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.xlabel('Predicted',fontsize=25)
varmatplt.ylabel('Residual',fontsize=25)
varmatplt.axhline(y=0)
varmatplt.title('Residual vs. Predicted',fontsize=35)

varmatplt.show()

from sklearn.metrics import mean_squared_error
rmse = varnumpy.sqrt(mean_squared_error(Y, predicted))
print('RMSE:')
print(rmse)

lass = linear_model.Lasso(alpha = 1.15e8)
lass.fit (X, Y)

print('\nR²:')
print(lass.score(X, Y))
origparams = varnumpy.append(lass.coef_, lass.intercept_)
print('\nParameter features:')
print(origparams)

print('\nCross Validation Score:')
print(cross_val_score(lass, X, Y, cv=12))

rid = linear_model.Ridge(alpha = 90)
rid.fit (X, Y)

predicted = rid.predict(X)
residual = Y - predicted

fig = varmatplt.figure(figsize=(30,30))
ax1 = varmatplt.subplot(211)
varseaborn.distplot(residual, color ='teal')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.title('Residual counts',fontsize=35)
varmatplt.xlabel('Residual',fontsize=25)
varmatplt.ylabel('Count',fontsize=25)

ax2 = varmatplt.subplot(212)
varmatplt.scatter(predicted, residual, color ='teal')
varmatplt.tick_params(axis='both', which='major', labelsize=20)
varmatplt.xlabel('Predicted',fontsize=25)
varmatplt.ylabel('Residual',fontsize=25)
varmatplt.axhline(y=0)
varmatplt.title('Residual vs. Predicted',fontsize=35)

varmatplt.show()

from sklearn.metrics import mean_squared_error
rmse = varnumpy.sqrt(mean_squared_error(Y, predicted))
print('RMSE:')
print(rmse)

rid = linear_model.Ridge(alpha = 90)
rid.fit (X, Y)

print('\nR²:')
print(rid.score(X, Y))
origparams_rid = varnumpy.append(rid.coef_, rid.intercept_)
print('\nParameter features:')
print(origparams_rid)

print('\nCross Validation Score:')
print(cross_val_score(rid, X, Y, cv=12))

# Ridge is the winner:
# Although the RMSE score is high look at the cross validation score, it ranges between 60-70%. This is really good. Lets take a look at the coefficients and see dollar to dollar what each feature is valued at.

cdf = varpanda.DataFrame(data = rid.coef_, index = X.columns, columns = ['Coefficients'])
cdf

# Boosting:
# Boosting is a machine learning ensemble meta-algorithm for primarily reducing bias, and also variance in supervised learning, and a family of machine learning algorithms which convert weak learners to strong ones. Boosting is based on the question posed by Kearns and Valiant (1988, 1989): Can a set of weak learners create a single strong learner? A weak learner is defined to be a classifier which is only slightly correlated with the true classification (it can label examples better than random guessing). In contrast, a strong learner is a classifier that is arbitrarily well-correlated with the true classification. ~ WikiPedia)

# Let's see if boosting can improve our scores.

from sklearn.ensemble import GradientBoostingRegressor

r_sq = []
deep = []
mean_scores = []

#loss : {‘ls’, ‘lad’, ‘huber’, ‘quantile’}
for n in range(3, 11):
    gbr = GradientBoostingRegressor(loss ='ls', max_depth=n)
    gbr.fit (X, Y)
    deep.append(n)
    r_sq.append(gbr.score(X, Y))
    mean_scores.append(cross_val_score(gbr, X, Y, cv=12).mean())
	
plt_gbr = varpanda.DataFrame()

plt_gbr['mean_scores'] = mean_scores
plt_gbr['depth'] = deep
plt_gbr['R²'] = r_sq

f, ax = varmatplt.subplots(figsize=(15, 5))
varseaborn.barplot(data = plt_gbr, x='depth', y='R²')
varmatplt.show()

f, ax = varmatplt.subplots(figsize=(15, 5))
varseaborn.barplot(data = plt_gbr, x='depth', y='mean_scores')
varmatplt.show()

gbr = GradientBoostingRegressor(loss ='ls', max_depth=6)
gbr.fit (X, Y)
predicted = gbr.predict(X)
rmse = varnumpy.sqrt(mean_squared_error(Y, predicted))
scores = cross_val_score(gbr, X, Y, cv=12)

print('\nCross Validation Scores:')
print(scores)
print('\nMean Score:')
print(scores.mean())
print('\nRMSE:')
print(rmse)

# Boosting is the clear winner
# It reduced the bias and variances. This is evidenced by a higher mean score and a much lower RMSE than the ridge model.

cdf = varpanda.DataFrame(data = gbr.feature_importances_, index = X.columns, columns = ['Importance'])
cdf