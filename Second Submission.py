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



# Feature Manipulation and Extraction

#all_data["1st2ndFloor"] = (all_data["1stFlrSF"] + all_data["2ndFlrSF"])
#all_data["AllLivArea"] = (all_data["1stFlrSF"] + all_data["2ndFlrSF"] + all_data["LowQualFinSF"] + all_data["GrLivArea"])
    # when these two are included, the min RSME for Ridge goes down, while for Lasso it goes up

# drop this one as 99.5% of streets are paved
all_data = all_data.drop('Street', axis=1)

# drop this one also as ony 6% of houses have alleys
all_data = all_data.drop('Alley', axis=1)

# 63% of lot shapes are regular, not big enough to separate them into bins
#print (all_data["LotShape"].value_counts())

# 89% of values are Level, so bin the other values together as not level
all_data['LevelContour'] = (all_data['LandContour'] == "Lvl") * 1
all_data = all_data.drop('LandContour', axis=1)

# only one value is not public utilities, so just drop this one all together
all_data = all_data.drop("Utilities", axis=1)

# 73% of values are inside lots, don't bin this one yet
#print (all_data["LotConfig"].value_counts())

# 95% of slopes are gentle, so basket the rest together
all_data["GtlLandSlope"] = (all_data["LandSlope"] == "Gtl") * 1
all_data = all_data.drop('LandSlope', axis=1)

# leave this one as its super diverse
# print (all_data["Neighborhood"].value_counts())

# 86% of values are Normal, so bin the other values together as not normal
#print (all_data['Condition1'].value_counts())
all_data["NormCondition1"] = (all_data["Condition1"] == "Norm") * 1
all_data = all_data.drop('Condition1', axis=1)

# 98.9% of values in Condition 2 are Norm, so treat the others as "not normal", and drop the old column
#print (all_data['Condition2'].value_counts())
all_data["NewCondition2"] = (all_data["Condition2"] == "Norm") * 1
all_data = all_data.drop("Condition2", axis=1)

# ????? - bin these into single-family detached as 83% of values fall into that bin
#print (all_data["BldgType"].value_counts())

# leave this b/c 50% is the biggest
#print (all_data["HouseStyle"].value_counts())

"""# leave this one as is -- can group into 5+ and then <5
print (all_data["OverallCond"].value_counts())
"""

# 79% of roof style is Gable so we'll replace this one and see if it has any effect
#all_data["GableRoofStyle"] = (all_data["RoofStyle"] == "Gable") * 1
#all_data = all_data.drop("RoofStyle")

# 98.5% of roof material is standard shingles, so bin the rest together
all_data["StandardRoofMaterial"] = (all_data["RoofMatl"] == "CompShg") * 1
all_data = all_data.drop("RoofMatl")

# leave this one at 35% biggest
#print (all_data["Exterior1st"].value_counts())

# leave this one at 34% biggest
#print (all_data["Exterior2nd"].value_counts())

# leave this one at 59% biggest
#print (all_data["MasVnrType"].value_counts())

# leave this one at 61% biggest
#print (all_data["ExterQual"].value_counts())

"""# 86% biggest.... ehh we can bin these
all_data["AvgExteriorCond"] = (all_data["ExteriorCond"] == "TA") * 1
all_data = all_data.drop("ExteriorCond")
"""

# this ones good at biggest 44%
#print (all_data["Foundation"].value_counts())

# this ones good at biggest of 43%
#print (all_data["BsmtQual"].value_counts())

# Typical basement condition 89% of total --- but other basement's aren't so pass on this one for now
#all_data["TypicalBsmtCond"] = (all_data["BsmtCond"] == "TA") * 1
#all_data.drop("BsmtCond", axis=1)

# this one's bueno at 65.2%
#print (all_data["BsmtExposure"].value_counts())

# This one's bueno for now
# print (all_data["BsmtFinType1"].value_counts())

# 85.4% of values are Unf
all_data["UnfBsmtFinType2"] = (all_data["BsmtFinType2"] == "Unf") * 1
all_data = all_data.drop("BsmtFinType2")

# 98.4% of Heating is GasA
all_data["GasAHeating"] = (all_data["Heating"] == "GasA") * 1
all_data = all_data.drop("Heating")

# Leave this for now
# print (all_data["HeatingQC"].value_counts())

# 93.2% yes, so bin the rest
all_data["YesCentralAir"] = (all_data["CentralAir"] == "Y") * 1
all_data = all_data.drop("CentralAir")

# 91.5% of Electrical systems are standard breakers, so bin the rest together
all_data["StdElectrical"] = (all_data["Electrical"] == "SBrkr") * 1
all_data = all_data.drop("Electrical")

# Leave Kitchen Quality b/c 52% is biggest value

#93% of Functionality is Typical, so bin together all the other values
all_data["TypicalFunctionality"] = (all_data["Functional"] == "Typ") * 1
all_data = all_data.drop("Functional", axis=1)

# Leave Fireplace Quality b/c 25% is biggest value
# Leave garage finish b/c 42% is biggest value

"""# 89.2% of Garage Quality is Average, so bin the lower ones into a separate one
all_data["AvgGarageQuality"] = np.any(all_data["GarageQual"] == "TA") * 1
all_data = all_data.drop("GarageCond")

# 90.9% of Garage conditions are Average, so bin the others together if they're lower
all_data["AvgGarageCondition"] = np.any(all_data["GarageCond"] == "TA") * 1
all_data = all_data.drop("GarageCond")
"""

# 90.4% of homes have a paved driveway, so bin the other ones together
all_data["PavedDriveway"] = (all_data["PavedDrive"] == "Y") * 1
all_data = all_data.drop("PavedDrive", axis=1)


# only 3.5% of homes have a Miscellaneous Feature, so drop this column
all_data = all_data.drop("MiscFeature", axis=1)

# 86.5% of SaleType is Warranty Deed - Conventional, so bin the other ones together
# print (all_data["SaleType"].value_counts())
all_data["WDSaleType"] = (all_data["SaleType"] == "WD") * 1
all_data = all_data.drop("SaleType", axis=1)


# 82.2% of SaleCondition is Normal, so leave this one for now
#print (all_data["SaleCondition"].value_counts())




# Heatmap of correlated variables:
corr = all_data.select_dtypes(include = ['float64', 'int64', 'object']).iloc[:, 1:].corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, vmax=1, square=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()



# Histogram showing that SalePrice can be normalized using a log transformation
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"Price":train["SalePrice"], "Log(Price + 1)":np.log1p(train["SalePrice"])})
prices.hist()
plt.show()

# So log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

# log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) # compute skew
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
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 8, 8.2, 9, 10, 11, 12, 13, 14, 15, 30, 50, 75]
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

