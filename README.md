# credit_risk_analysis

## 1. Project Overview

Here, I used Scikit-learn and Imbalance-learn in a Jupyter Notebooks to compare the performance of different sampling and modeling methods
in predicting credit risk. Because high-risk loans are greatly outnumbered by low-risk loans, categorizing future risk is an inherently 
imbalanced classification problem, requiring any number of potential methodologies to effectively train predictive models. To this end, 
I used a dataset of credit card loans to compare the predictive accuracy of logistic regression models following different sampling schemes:
oversampling, undersampling, and combination (over and under) sampling. I then basic sampling to compare the performance of ensemble classifier 
methods: a balanced random forest classifier, and easy ensemble Adaboost classifier. To characterize model performance, I used balanced
accuracy scores, precision, and recall for each model.


## 2. Results

Raw performance scores are below:

- Model 1: logistic regression + random naive oversampling
	- 0.653 balanced accuracy score
	- 0.99 precision
	- 0.64 recall
	
- Model 2: logistic regression + SMOTE oversampling
	- 0.643 balanced accuracy score
	- 0.99 precision
	- 0.58 recall
	
- Model 3: logistic regression + cluster centroids undersampling
	- 0.532 balanced accuracy score
	- 0.99 precision
	- 0.45 recall
	
- Model 4: logistic regression + SMOTEENN combination sampling
	- 0.661 balanced accuracy score
	- 0.99 precision
	- 0.54 recall
	
- Model 5: balanced random forest classifier
	- 0.800 balanced accuracy score
	- 0.99 precision
	- 0.91 recall
	
- Model 6: AdaBoost classifier
	- 0.911 balanced accuracy score
	- 1.00 precision
	- 0.94 recall


## 3. Summary

In general, ensemble classifiers greatly outperformed logistic regression models in predicting credit risk, regardless of the sampling scheme
employed. All six models had very high precision (0.99-1.00), meaning this metric is not useful for model comparison in this case. However, 
both balanced accuracy scores and recall scores were consistently substantially higher for ensemble classifiers (mean accuracy score = 0.856, 
mean recall = 0.925) than logistic regression (accuracy = 0.623, recall = 0.553). Of the six methods tested, the AdaBoost classifier performed
best (accuracy = 0.911, recall = 0.94), making it the best choice for future efforts to model credit risk. 

