import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

train = pd.read_csv("/Users/nickwalker/Desktop/Data Sets/House Price Prediction - Kaggle/train.csv")
train.set_index("Id")

test = pd.read_csv("/Users/nickwalker/Desktop/Data Sets/House Price Prediction - Kaggle/test.csv")
test.set_index("Id")

all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                      test.loc[:, 'MSSubClass':'SaleCondition']))


# Histogram showing that SalePrice can be normalized using a log transformation
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"Price":train["SalePrice"], "Log(Price + 1)":np.log1p(train["SalePrice"])})
prices.hist()
plt.show()

# So log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

# log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) # compute skew
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# convert categorical variables into dummy/indicator variables 
all_data = pd.get_dummies(all_data)


# Filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())


# Create Test and Train Matrices
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
Y = train.SalePrice


# MODELS
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score


# PCA
"""print (X_train.shape)
pca = PCA(n_components=250)
pca.fit(X_train)
X_train = pca.transform(X_train)
print (X_train.shape)
"""


# RMSE
n_folds = 5
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, Y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
model_ridge = Ridge()


# Ridge Regression (L2)
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "L2 Parameter Validation Score")
plt.xlabel("Alpha")
plt.ylabel("RMSE")
plt.show()

print (cv_ridge.min())


# Lasso Regression (L1)
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005], selection='random', max_iter=15000).fit(X_train, Y)
res = rmse_cv(model_lasso)
print("Mean:",res.mean())
print("Min: ",res.min())


# How many values LassoCV kept/eliminated:
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print ("Lasso picked ", str(sum(coef != 0)), "variables and eliminated the other ", str(sum(coef == 0)), "variables")


# Plotting feature importance
imp_coef = pd.concat([coef.sort_values().head(10),
                      coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()

