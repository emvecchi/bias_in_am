def get_ols_analysis(df, dep_variable, ind_variables1, ind_variables2):
	formula1 = get_formula(dep_variable, ind_variables1)
	model1 = ols(formula1, data=df).fit()
	formula2 = get_formula(dep_variable, ind_variables2)
	model2 = ols(formula2, data=df).fit()
	print(model1.summary())
	print(model2.summary())
	anova=get_anova(model1,model2)
	print('ANOVA'+anova)
	return(model1,model2)

def get_ols_analysis_single_m(df, dep_variable, ind_variables):
	formula = get_formula(dep_variable, ind_variables)
	model = ols(formula, data=df).fit()
	print(model.summary())
	anova_table = sm.stats.anova_lm(model, typ=2)
	print(anova_table)
	return(model)

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
	Z=df[[dep_variable]+ind_variables]
	print('Variance Inflation Factor:')
	print(get_vif(Z))
	model = LogisticRegression()
	m = model.fit(X, y)
	results = pd.DataFrame({
		'variable': ind_variables,
		'f-reg_score': scores,
		'f_reg_p_value': p_values
		})
	summary = pd.DataFrame({
		'X': [ind_variables],
		'y': dep_variable,
		'intercept': m.intercept_,
		'coefficients': [m.coef_],
		'R-squared': m.score(X,y)
		})
	return(m, results, summary)

def get_vif(X):
	vif_data = pd.DataFrame()
	vif_data["feature"] = X.columns
	vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
	return(vif_data)

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

def get_anova(m1, m2):
	anova_table = sm.stats.anova_lm(m1,m2)
	return(anova_table)

def get_model_plot(model, outfile):
    y_pred = model.fittedvalues
    residuals = model.resid
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
    plt.savefig(outfile)


#TO-DO: add function to get stats of gender column in question, with distribution plot 

if __name__ == '__main__':
	import sys, json
	import pandas as pd
	import numpy as np
	from pandas import read_csv
	import statsmodels.api as sm
	import statsmodels.formula.api as smf
	import scipy.stats as stats
	from sklearn.feature_selection import chi2
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.feature_selection import f_regression
	from statsmodels.formula.api import ols
	from statsmodels.stats.outliers_influence import variance_inflation_factor

	#datafile='/mount/arbeitsdaten14/projekte/sfb-732/d8/falensaa/BiasInArguments/intermediate-data/tal_elal_2016/train_period_data.annotations.jsonlist'
	datafile = sys.argv[1]
	with open(datafile, 'r') as f:
		data = [json.loads(line) for line in f]

	df = pd.DataFrame(data)
	dep_variable=sys.argv[2]
	ind_variables=sys.argv[3:]

	print('-----------------------------')
	print('::Model::')
	print('Dependent Variable: '+dep_variable)
	print('Independent Variables: '+str(ind_variables)[1:-1])
	print('')

	# NOTE: df['gender'] needs to be [0,1]
	subset = df[df[dep_variable].isin(['M', 'F'])].copy()
	print('Distribution of Dependent Variable:')
	print(subset[dep_variable].describe(include='all'))
	print('')
	subset[dep_variable] = subset[dep_variable].map({'M': 0, 'F': 1})
	m,results,summary = get_log_regression(subset, dep_variable, ind_variables)
	
	print('')
	print('Variable Results:')
	print(results)
	print('')
	print('Summary:')
	print(summary)
	print('')

