def get_ols_analysis(df, dep_variable, ind_variables1, ind_variables2):
	formula1 = get_formula(dep_variable, ind_variables1)
	model1 = ols(formula1, data=df).fit()
	formula2 = get_formula(dep_variable, ind_variables2)
	model2 = ols(formula2, data=df).fit()
	print(model1.summary())
	print(model2.summary())
	anova_table = sm.stats.anova_lm(model1, model2)
	print(anova_table)

def get_variable_importance(df, dep_variable, ind_variables):
	y = df[dep_variable]
	X = df[ind_variables]
	# assess variable importance using chi-squared test
	# higher scores and lower p-values indicate variables that are more influential in predicting the DV
	scores, p_values = chi2(X, y)
	model = RandomForestClassifier()
	model.fit(X, y)
	importances = model.feature_importances_
	results = pd.DataFrame({
		'variable': ind_variables,
		'score': scores,
		'p_value': p_values,
		'importance': importances
		})
	return(results)

def get_log_regression(df, dep_variable, ind_variables):
	y = df[dep_variable]
	X = df[ind_variables]
	# assess variable importance using F-test
	# higher scores and lower p-values indicate variables that are more influential in predicting the DV
	scores, p_values = f_regression(X, y)
	# logistic regression model to determine variable importance
	# coef's indicate direction and magnitude of influence of each IV on the target DV
	model = LogisticRegression()
	m = model.fit(X, y)
	coefficients = m.coef_[0]
	results = pd.DataFrame({
		'variable': ind_variables,
		'score': scores,
		'p_value': p_values,
		'coefficient': coefficients
		})
	return(m, results)

def get_mixed_effects_regression(df, dep_variable, ind_variables, random_variable, group_by):
	formula = get_formula(dep_variable, ind_variables)
	formula += ' + (1 | '+random_variable+')'
	model = smf.mixedlm(formula, data=df, groups = df[group_by])
	result = model.fit()
	return(result)

def get_formula(dep_variable, ind_variables):
	formula = dep_variable + ' ~ ' + ind_variables[0]
	x = 1
	while x in range(len(ind_variables)):
		formula += ' + ' + ind_variables[x]
		x +=1
	return(formula)

if __name__ == '__main__':
	import sys
	import pandas as pd
	from pandas import read_csv
	import researchpy as rp
	import statsmodels.api as sm
	import statsmodels.formula.api as smf
	import scipy.stats as stats
	from sklearn.feature_selection import chi2
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.feature_selection import f_regression
	from statsmodels.formula.api import ols


	datafile = sys.argv[1]
	df = pd.read_csv(datafile, sep="\t", lineterminator="\n")
	
	# NOTE: df['gender'] needs to be [0,1]
	
	# Sample data
	#data = pd.DataFrame({
	#	'gender': [1, 0, 1, 0, 1, 1, 0, 1, 0],
	#	'likes': [5, 2, 13, 4, 23, 24, 8, 9, 3],
	#	'score': [0.87, 0.42, 0.77, 0.23, 0.52, 0.60, 0.40, 0.48, 0.19],
	#	'ups': [17, 8, 23, 2, 10, 9, 3, 12, 1],
	#	'OP_id': [12322, 42312, 12322, 72994, 72994, 138284, 298820, 798920, 304091] 
	#	})

	## Example uses for each def:
	# get_ols_analysis(df, 'gender', ['likes'], ['likes', 'score'])
	# chi2_info = get_variable_importance(df, 'gender', ['likes','ups','score'])
	# model, regression_info = get_log_regression(df, 'gender', ['likes','ups','score'])
	# mixed_effects_info = get_mixed_effects_regression(df, 'gender', ['likes','ups','score'], 'OP_id', 'OP_id')
