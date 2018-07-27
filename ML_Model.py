# -*- coding: utf-8 -*-
"""
@author: ady.kalra
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read Dataset from csv file
melbourne_file_path = 'Raw_Data/melHousingData.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data.head()


melbourne_data.info()

# remove all missing data
df2 = melbourne_data[melbourne_data.Price.notnull()]
df2 = df2[df2.BuildingArea.notnull()]
df2 = df2[df2.Car.notnull()]
df2 = df2[df2.Landsize.notnull()]
df2 = df2[df2.Lattitude.notnull()]
df2 = df2[df2.Longtitude.notnull()]
df2.info()

# if number of bedrooms has a relationship to price
print(pd.crosstab(melbourne_data['Price'].mean(), melbourne_data['Bedroom2']))
print('\n')


f, ax = plt.subplots(figsize=(5, 5))
sns.regplot(data=df2, x='Rooms', y='Price')
plt.show()

# Although weak, it appears that there seems to be a positive relationship. Let's see what is the actual correlation between price and the other data points. We will look at this in 2 ways heatman for visualization and the correlation coefficient score.

f, ax = plt.subplots(figsize=(5, 5))
corrmat = df2.corr()
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
corrmat

# Distribution of the data
plt.rcParams['figure.figsize'] = 16, 16
df2.loc[:,:].hist(bins=100)
plt.show()

# Create new features: Let's create new features to see if these new features will have a stronger correlation coefficient score than the original. We will do so by mixing the data and altering the data.
df2['Roomssq'] = df2.Rooms ** 2
df2['Roomssqrt'] = df2.Rooms ** (1/2)
df2['Plus'] = df2.Rooms + df2.Bedroom2 + df2.Bathroom
df2['Prod'] = df2.Rooms * df2.Bedroom2 * df2.Bathroom
df2['year'] = (2017 - df2.YearBuilt)
df2['yearsq'] = (2017 - df2.YearBuilt) ** 2
df2['yearsqrt'] = (2017 - df2.YearBuilt) ** (1/2)

f, ax = plt.subplots(figsize=(5, 5))
corrmat = df2.corr()
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
corrmat

# It appears that the column "Plus" has a correlation score of .529 which is the highest correlation score out of all the features new and old. This new feature was created by rooms, bedrooms and bathrooms.
# Let's drop the weakest scores and run our first model "Decision Tree Regressor"
X = df2.drop(['YearBuilt', 'year', 'yearsq', 'yearsqrt', 'Price',], axis=1)
Y = df2.Price
X = pd.get_dummies(data=X)


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

dtr = DecisionTreeRegressor(max_features=10, max_depth=10)
dtr.fit(X, Y)
print(cross_val_score(dtr, X, Y, cv=5))

# [-0.04384913  0.00482545  0.00621629 -0.04248696 -0.00801812]
# The scores above is bad...Let's take a look at why it's bad

predicted = dtr.predict(X)
residual = Y - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='orange')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='orange')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y, predicted))
print('RMSE:')
print(rmse)


# RMSE: 677273.379164
#The root-mean-square deviation (RMSD) or root-mean-square error (RMSE) (or sometimes root-mean-squared error) is a frequently used measure of the differences between values (sample and population values) predicted by a model or an estimator and the values actually observed. The RMSD represents the sample standard deviation of the differences between predicted values and observed values. These individual differences are called residuals when the calculations are performed over the data sample that was used for estimation, and are called prediction errors when computed out-of-sample. The RMSD serves to aggregate the magnitudes of the errors in predictions for various times into a single measure of predictive power. RMSD is a measure of accuracy, to compare forecasting errors of different models for a particular data and not between datasets, as it is scale-dependent. ~ WikiPedia

# The higher the RMSE the worst our predicting model is and the reason why the RMSE score is high and bad is because it appears this model picked a single price point for the categories that the model selected this is shown in the graphs above.

# Let's take a different approach...
# and think like a real estate agents. As real estate agents we should look the comparables see what a house like what we're trying to predict is priced based on sales of similar homes. Let's use the next model "Nearest Neighbors" and use the 2 nearnest neighbors.

from sklearn import neighbors

knn = neighbors.KNeighborsRegressor(n_neighbors=2)
knn.fit(X, Y)

predicted = knn.predict(X)
residual = Y - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='orange')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='blue')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y, predicted))
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
    rmse_l.append(np.sqrt(mean_squared_error(Y, predicted)))
    num.append(n)
	
df_plt = pd.DataFrame()
df_plt['rmse'] = rmse_l
df_plt['n_neighbors'] = num
ax = plt.figure(figsize=(15,7))
sns.barplot(data = df_plt, x = 'n_neighbors', y = 'rmse')
plt.show()

print(rmse_l)

# [246010.72612387789, 299140.89928988385, 331499.77892903792, 347367.56599903584, 363610.28713245952, 373641.28252136859, 382776.00220798928, 390369.98595582315, 399132.94385391136, 404725.34418166213, 410064.63836764981, 414764.42545172054, 419022.88172154903, 423461.00056615419]
# It appears that 2 nearest neighbors is the optimal number of neighbors. This is evidenced by the increasing RMSE as we increase the number of neighbors.

# Lasso & Ridge:
# Let's look at a model "Lasso" & "Ridge", these models will penalize the model for larger coefficients. So these models will regularizes the way it predicts the price. Lets try a low alpha first and see how it performs.

from sklearn import linear_model
lass = linear_model.Lasso(alpha = .025)
lass.fit (X, Y)

predicted = lass.predict(X)
residual = Y - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='orange')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='orange')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y, predicted))
print('RMSE:')
print(rmse)

# RMSE: 3127.28332239

alp = .1
for n in range (0,3):
    lass = linear_model.Lasso(alpha = alp)
    lass.fit (X, Y)
    predicted = lass.predict(X)
    rmse = np.sqrt(mean_squared_error(Y, predicted))
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

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='orange')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='orange')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y, predicted))
print('RMSE:')
print(rmse)

# RMSE: 14735.7430427

lass = linear_model.Lasso(alpha = .025)
lass.fit (X, Y)

print('\nR² for the model with many features:')
print(lass.score(X, Y))
origparams = np.append(lass.coef_, lass.intercept_)
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
origparams_rid = np.append(rid.coef_, rid.intercept_)
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

df3 = melbourne_data[melbourne_data.Price.notnull()]
#df3 = df3[df3.BuildingArea.notnull()]
df3 = df3[df3.Car.notnull()]
#df3 = df3[df3.Landsize.notnull()]
#df3 = df3[df3.Lattitude.notnull()]
#df3 = df3[df3.Longtitude.notnull()]
df3.info()

df3 = df3.drop(['BuildingArea', 'YearBuilt', 'Landsize', 'Lattitude', 'Longtitude'], axis=1)
df3.info()

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

temp = df3.groupby('Suburb').agg({'min', 'mean', 'max'})
temp2 = temp['Price']
temp2 = temp2.reset_index()
temp2.columns = ['Suburb', 'max_sub_id', 'min_sub_id', 'mean_sub_id']
print(temp2.info())
temp2.head()

f, ax = plt.subplots(figsize=(15, 60))
sns.boxplot(data = df3, x='Price', y='Suburb')
plt.show()

df_copy = df3
df_copy = pd.merge(df_copy, temp2, on='Suburb', how='left')
df_copy.head()

type_g = df3.groupby('Type').agg({'min', 'mean', 'max'})
temp3 = type_g['Price']
temp3 = temp3.reset_index()
temp3.columns = ['Type', 'max_t_id', 'min_t_id', 'mean_t_id']
temp3

f, ax = plt.subplots(figsize=(15, 10))
sns.stripplot(data = df3, x='Type', y='Price', jitter=.5)
plt.show()
 
df_copy = pd.merge(df_copy, temp3, on='Type', how='left')
df_copy.head()

method = df3.groupby('Method').agg({'min', 'mean', 'max'})
temp4 = method['Price']
temp4 = temp4.reset_index()
temp4.columns = ['Method', 'max_m_id', 'min_m_id', 'mean_m_id']
temp4


f, ax = plt.subplots(figsize=(15, 5))
sns.violinplot(data = df3, x='Price', y='Method', jitter=.5)
plt.show()

df_copy = pd.merge(df_copy, temp4, on='Method', how='left')
df_copy.head()

sellerg = df3.groupby('SellerG').agg({'min', 'mean', 'max'})
temp5 = sellerg['Price']
temp5 = temp5.reset_index()
temp5.columns = ['SellerG', 'max_s_id', 'min_s_id', 'mean_s_id']
print(temp5.info())
temp5.head()

f, ax = plt.subplots(figsize=(15, 60))
sns.stripplot(data = df3, x='Price', y='SellerG', jitter=.1)
plt.show()

df_copy = pd.merge(df_copy, temp5, on='SellerG', how='left')
df_copy.head()

council = df3.groupby('CouncilArea').agg({'min', 'mean', 'max'})
temp6 = council['Price']
temp6 = temp6.reset_index()
temp6.columns = ['CouncilArea', 'max_c_id', 'min_c_id', 'mean_c_id']
print(temp6.info())
temp6.head()

f, ax = plt.subplots(figsize=(15, 5))
sns.boxplot(data = df3, x='CouncilArea', y='Price')
plt.xticks(rotation='vertical')
plt.show()

#df_copy = df_copy.drop(['max_c_id_y', 'min_c_id_y', 'mean_c_id_y'], axis=1)

df_copy = pd.merge(df_copy, temp6, on='CouncilArea', how='left')
df_copy.head()

region = df3.groupby('Regionname').agg({'min', 'mean', 'max'})
temp7 = region['Price']
temp7 = temp7.reset_index()
temp7.columns = ['Regionname', 'max_r_id', 'min_r_id', 'mean_r_id']
print(temp7.info())
temp7.head()

f, ax = plt.subplots(figsize=(15, 10))
sns.boxplot(data = df3, x='Price', y='Regionname')
#plt.xticks(rotation='vertical')
plt.show()

df_copy = pd.merge(df_copy, temp7, on='Regionname', how='left')
df_copy.head()

df_copy['date_m'], df_copy['date_d'], df_copy['date_y'] = df_copy['Date'].str.split('/', 2).str
df_copy.head()

# Does the "when you sell your home" matter?
df_copy['date_m'] = df_copy['date_m'].astype(int)

f, ax = plt.subplots(figsize=(15, 5))
sns.stripplot(data = df_copy, x='date_m', y='Price', jitter=.25)
plt.show()

df_copy['date_d'] = df_copy['date_d'].astype(int)
f, ax = plt.subplots(figsize=(15, 10))
sns.boxplot(data = df_copy, x='date_d', y='Price')
plt.show()

df_copy['date_y'] = df_copy['date_y'].astype(int)
f, ax = plt.subplots(figsize=(7, 10))
sns.violinplot(data = df_copy, x='date_y', y='Price')
plt.show()

# Lets see how many features we now have:
df_copy.info()

df_copy = df_copy.drop(['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Date', 'CouncilArea', 'Regionname'], axis=1)
df_copy.info()

# Let's see how our new features correlate to the price:
f, ax = plt.subplots(figsize=(10, 10))
corrmat = df_copy.corr()
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
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
	

plt_dtr = pd.DataFrame()

plt_dtr['mean_scores'] = scores
plt_dtr['depth'] = depth

f, ax = plt.subplots(figsize=(15, 5))
sns.barplot(data = plt_dtr, x='depth', y='mean_scores')
plt.show()

# The above bar graph represent number of max depth on the x-axis that the model will limit itself to and on the y-axis is the mean scores of the cross validation of 12 folds. As you can see even the worst mean score of approximately 4, this model guesses way better due to having more information.

# Let's try "Nearest Neighbors" once more:

knn = neighbors.KNeighborsRegressor(n_neighbors=2)
knn.fit(X, Y)

print(cross_val_score(knn, X, Y, cv=12))

knn = neighbors.KNeighborsRegressor(n_neighbors=2)
knn.fit(X, Y)

predicted = knn.predict(X)
residual = Y - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='purple')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='purple')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y, predicted))
print('RMSE:')
print(rmse)

# RMSE: 264179.577229
# It appears that the RMSE is higher this time around, but I believe this is due to higher priced homes does not follow the normal pricing pattern. It seems that $2,000,000 would be the threshold for home prices.
# Lets give "Lasso & Ridge" another shot:

lass = linear_model.Lasso(alpha = 1.15e8)
lass.fit (X, Y)

predicted = lass.predict(X)
residual = Y - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='pink')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='pink')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y, predicted))
print('RMSE:')
print(rmse)

lass = linear_model.Lasso(alpha = 1.15e8)
lass.fit (X, Y)

print('\nR²:')
print(lass.score(X, Y))
origparams = np.append(lass.coef_, lass.intercept_)
print('\nParameter features:')
print(origparams)

print('\nCross Validation Score:')
print(cross_val_score(lass, X, Y, cv=12))

rid = linear_model.Ridge(alpha = 90)
rid.fit (X, Y)

predicted = rid.predict(X)
residual = Y - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='teal')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='teal')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y, predicted))
print('RMSE:')
print(rmse)

rid = linear_model.Ridge(alpha = 90)
rid.fit (X, Y)

print('\nR²:')
print(rid.score(X, Y))
origparams_rid = np.append(rid.coef_, rid.intercept_)
print('\nParameter features:')
print(origparams_rid)

print('\nCross Validation Score:')
print(cross_val_score(rid, X, Y, cv=12))

# Ridge is the winner:
# Although the RMSE score is high look at the cross validation score, it ranges between 60-70%. This is really good. Lets take a look at the coefficients and see dollar to dollar what each feature is valued at.

cdf = pd.DataFrame(data = rid.coef_, index = X.columns, columns = ['Coefficients'])
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
	
plt_gbr = pd.DataFrame()

plt_gbr['mean_scores'] = mean_scores
plt_gbr['depth'] = deep
plt_gbr['R²'] = r_sq

f, ax = plt.subplots(figsize=(15, 5))
sns.barplot(data = plt_gbr, x='depth', y='R²')
plt.show()

f, ax = plt.subplots(figsize=(15, 5))
sns.barplot(data = plt_gbr, x='depth', y='mean_scores')
plt.show()

gbr = GradientBoostingRegressor(loss ='ls', max_depth=6)
gbr.fit (X, Y)
predicted = gbr.predict(X)
rmse = np.sqrt(mean_squared_error(Y, predicted))
scores = cross_val_score(gbr, X, Y, cv=12)

print('\nCross Validation Scores:')
print(scores)
print('\nMean Score:')
print(scores.mean())
print('\nRMSE:')
print(rmse)

# Boosting is as advertised!!!
# It reduced the bias and variances. This is evidenced by a higher mean score and a much lower RMSE than the ridge model.

cdf = pd.DataFrame(data = gbr.feature_importances_, index = X.columns, columns = ['Importance'])
cdf