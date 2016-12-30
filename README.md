# House-Prices-Advanced-Regression-Techniques
Code submission for the Kaggle contest "House Prices: Advanced Regression Techniques". The contest involves using 79 predictor variables describing almost every aspect of residential homes in Ames, Iowa in order to build a predictive model to find the final price of each home. It can be found at the following link: https://www.kaggle.com/c/house-prices-advanced-regression-techniques 

Original submission:
includes Ridge and Lasso regression models designed to minimize the RMSE of the prediction variables. I did this by using a simple log transformation in order to reduce the skew of numeric features. I then converted all categorical variables into indicator variables and filled in all of the NaN's in order to build test and train matrices for sklearn. From this I constructed both models, using cross-validation to optimize my parameter values. 
The Lasso method proved surprisingly successful for a simple model without any feature engineering, producing a mean RMSE of .1231, kaggle submission score of .12092 and position of 661/3018 (Top 22%).

Second Submission:
Here I added a correlation heatmap and inspected the value counts of each predictor variable in order to bin together values that didn't occur frequently. For example, in the 'LandContour' predictor variable, 89% of the homes were placed on ground that had a Level Contour, so I separated these homes from homes that are placed on either a banked, hillside, or depressed contour, and grouped the latter homes together. After some experimentation, I found that I got the best results when grouping together variables with >85% of one dominant value. 
This method reduced the mean RSME to .1214 for the Lasso method, reduced kaggle submission score to .12082, and position to 609/3018 (Top 20%)


To Do: 
1.) Experiment with various data preprocessing techniques such as feature scaling and/or feature engineering on the predictor variables
2.) XGBoost to boost model performance




My original submission was constructed with reference to the following two kernels: 
https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models/comments
https://www.kaggle.com/yadavsarthak/house-prices-advanced-regression-techniques/you-got-this-feature-engineering-and-lasso
