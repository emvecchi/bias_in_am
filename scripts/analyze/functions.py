def get_tan_word_features(df):
    # This function adds the missing word features used in Tan et al (2016) to the feature df 
    import re
    import nltk
    import math
    df['tmp_text'] = df['text'].str.replace(r'\*Hello, users of CMV!.*', '', regex=True)
    #
    # definite & indefinite articles
    definite_articles = ['the']
    indefinite_articles = ['a','an']
    df['definite_article_count'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(definite_articles))) 
    df['definite_article_perc'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(definite_articles))) / df['nwords']
    df['indefinite_article_count'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(indefinite_articles)))
    df['indefinite_article_perc'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(indefinite_articles))) / df['nwords']
    #
    # second person pronoun
    second_person = ['you', 'your', 'yours', 'yourself', 'yourselves']
    df['second_person'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(second_person))) / df['nwords']
    #
    # first person plural
    first_person_pl = ['we', 'us', 'our', 'ours', 'ourselves']
    df['first_person_pl'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(first_person_pl))) / df['nwords']
    #
    # hedging
    hedges = ['maybe', 'perhaps', 'possibly', 'potentially', 'likely', 'probably', 'could', 'might', 'may', 'can', 'should']
    df['hedge_words'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(hedges)))
    df['hedge_words_perc'] = df['tmp_text'].str.lower().str.count(r'\b({})\b'.format('|'.join(hedges))) / df['nwords']
    #
    # .com|.edu|.org links
    df['com_link_count'] = df['tmp_text'].str.lower().str.count(r'\b[\w.-]+\.com\b')
    df['edu_link_count'] = df['tmp_text'].str.lower().str.count(r'\b[\w.-]+\.edu\b')
    df['org_link_count'] = df['tmp_text'].str.lower().str.count(r'\b[\w.-]+\.org\b')
    df['pdf_link_count'] = df['tmp_text'].str.lower().str.count(r'\b[\w.-]+\.pdf\b')
    #
    # punctuation
    df['num_quotations'] = df['tmp_text'].str.count(r'["\']')
    df['num_question'] = df['tmp_text'].str.count(r'\?')
    df['num_exclamation'] = df['tmp_text'].str.count(r'\!')
    #
    #  sentence boundaries
    nltk.download('punkt')
    df['sentence_count'] = df['tmp_text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    df['paragraph_count'] = df['tmp_text'].str.split(r'\n{2,}').apply(len)
    #
    # word entropy
    df['tokenized_text'] = df['tmp_text'].apply(nltk.word_tokenize)
    df['word_frequencies'] = df['tokenized_text'].apply(nltk.FreqDist)
    df['total_words'] = df['tokenized_text'].apply(len)
    df['word_probabilities'] = df['word_frequencies'].apply(lambda x: {word: freq / sum(x.values()) for word, freq in x.items()})
    df['entropy'] = df['word_probabilities'].apply(lambda x: -sum(prob * math.log2(prob) for prob in x.values()))
    df = df.drop(columns=['word_frequencies','total_words','word_probabilities'])
    #
    # examples
    df['example_count'] = df['tmp_text'].str.count(r'example', flags=re.IGNORECASE)
    #
    df = df.drop(columns=['tmp_text'])
    return(df)

def get_correlations_heatmap(df, variables, outfile, x, y):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.subplots(figsize=(x,y))
    correlation_matrix = df[variables].corr()
    heatmap = sns.heatmap(correlation_matrix, annot=True, xticklabels=1, yticklabels=1, cmap='Blues')
    heatmap.figure.savefig(outfile)

def get_box_plots(df, variable, outfile):
    plt.boxplot(df[variable])
    plt.title('Range of Data for '+variable)
    plt.xlabel(variable)
    plt.ylabel('Values')
    plt.savefig(outfile)

def get_histo_graph(df, variable, outfile):
    import numpy as np
    plt.hist(df[variable], bins=np.arange(0, 100, 10))
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.title('Distribution of '+variable)
    plt.savefig(outfile)

def get_ols_analysis(df, dep_variable, ind_variables1, ind_variables2):
    from statsmodels.formula.api import ols
    import statsmodels.api as sm
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
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    formula = get_formula(dep_variable, ind_variables)
    model = ols(formula, data=df).fit()
    print(model.summary())
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    return(model)

def get_variable_importance(df, dep_variable, ind_variables):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import chi2
    import scipy.stats as stats
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
    import pandas as pd
    import statsmodels.api as sm
    y = df[dep_variable]
    X = df[ind_variables]
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    result = model.fit()
    Z=df[[dep_variable]+ind_variables]
    print('Variance Inflation Factor:')
    print(get_vif(Z))
    print()
    print('LMG R^2 Decomposition:')
    get_lmg_r2_decomposition(X, y)
    return(model, result)

def get_lmg_r2_decomposition(X, y):
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X, y)
    feature_names = X.columns
    # Get the predicted probabilities
    y_pred_prob = clf.predict_proba(X)[:, 1]
    # Calculate the baseline AUC
    baseline_auc = roc_auc_score(y, y_pred_prob)
    # Calculate permutation feature importance
    perm_importance = permutation_importance(clf, X, y, scoring='roc_auc', n_repeats=10, random_state=0)
    # Calculate LMG R^2 decomposition
    lmg_r2_decomposition = perm_importance.importances_mean / baseline_auc
    # Print the LMG R^2 decomposition scores
    for i, feature in enumerate(feature_names):
        print(f"\t{feature}: \t\t{lmg_r2_decomposition[i]}")

def get_log_regression2(df, dep_variable, ind_variables):
    from sklearn.linear_model import LogisticRegression
    import scipy.stats as stats
    from sklearn.feature_selection import f_regression
    import pandas as pd
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
    import scipy.stats as stats
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor
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
    import statsmodels.api as sm
    anova_table = sm.stats.anova_lm(m1,m2)
    return(anova_table)

def load_dataframes(datafile, featurefile, dep_variable):
    import sys, json, os, ast
    import pandas as pd
    with open(datafile, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    subset = df[df[dep_variable].isin(['M', 'F'])].copy()
    subset[dep_variable] = subset[dep_variable].map({'M': 0, 'F': 1})
    subset['edited_binary'] = subset['edited'].apply(lambda x: 0 if x is False else 1 if pd.notnull(x) else None)
    tmp_feature_df = pd.read_csv(featurefile, lineterminator="\n")
    tmp_feature_df = tmp_feature_df.drop("Unnamed: 0",axis=1)
    tmp_feature_df = tmp_feature_df[tmp_feature_df['id'].isin(subset['id'])].copy()
    expanded_df = pd.DataFrame()
    for _, row in tmp_feature_df.iterrows():
        feature_set_str = row['feature_set']
        feature_set_dict = ast.literal_eval(feature_set_str)
        temp_df = pd.DataFrame.from_records([feature_set_dict], index=[row['id']])
        expanded_df = expanded_df.append(temp_df)
    feature_df = tmp_feature_df.drop('feature_set', axis=1).merge(expanded_df, left_on='id', right_index=True)
    feature_df = feature_df.rename(columns={'text_x':'text'})
    feature_df = feature_df.rename(columns={'type_x':'type'})
    return(subset, feature_df)

def get_acl2022_feature_subset(df,dep_variable):
    acl2022_features=['mtld_original_aw',
                    'mattr50_aw',
                    'hdd42_aw',
                    'ADV',
                    'AUX',
                    'entities',
                    'past_tense',
                    'PRON',
                    'SCONJ',
                    'nwords',
                    'characters_per_word',
                    'syll_per_word',
                    'long_words',
                    'flesch',
                    'gunningFog',
                    'action_component',
                    'affect_friends_and_family_component',
                    'certainty_component',
                    'economy_component',
                    'failure_component',
                    'fear_and_digust_component',
                    'joy_component',
                    'negative_adjectives_component',
                    'objects_component',
                    'polarity_nouns_component',
                    'polarity_verbs_component',
                    'politeness_component',
                    'positive_adjectives_component',
                    'positive_nouns_component',
                    'positive_verbs_component',
                    'respect_component',
                    'social_order_component',
                    'trust_verbs_component',
                    'virtue_adverbs_component',
                    'well_being_component']
    feature_subset = df[['id','text','type',dep_variable]+acl2022_features]
    return(feature_subset)

def get_feature_hierarch_correlations(df, correlation_threshold, outfile):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.cluster import hierarchy
    correlation_matrix = df.iloc[:, 4:].corr(method='spearman')
    #plt.figure(figsize=(12, 10))
    #sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    #plt.title("Pairwise Spearman Correlation")
    #plt.savefig('correlation_matrix.png')
    distance_matrix = 1 - correlation_matrix.abs()
    linkage = hierarchy.linkage(distance_matrix, method='complete')
    plt.figure(figsize=(20, 15))
    dendrogram = hierarchy.dendrogram(linkage, labels=df.iloc[:, 4:].columns, orientation='top')
    plt.title('Dendrogram of Feature Clustering')
    plt.ylabel('Spearman Distance')
    plt.savefig(outfile)
    clusters = hierarchy.fcluster(linkage, t=correlation_threshold, criterion='distance')
    subclusters = {}
    for feature, cluster in zip(df.iloc[:, 4:].columns, clusters):
        if cluster not in subclusters:
            subclusters[cluster] = []
        subclusters[cluster].append(feature)
    for cluster, features in subclusters.items():
        if len(features) > 1:
            print(f"Subcluster {cluster}:")
            print(features)
            print()

def get_stepwise_selection_OLS(df, dep_variable):
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # Separate the target variable from the predictor variables
    X = df.drop(dep_variable, axis=1)  # Predictor variables
    y = df[dep_variable]               # Target variable
    # Add a constant column to the predictor variables
    X = sm.add_constant(X)
    # Perform step-wise model selection using stepAIC
    model = sm.OLS(y, X)
    result = model.fit()
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["GVIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    selected_features = result.model.exog_names
    final_model = sm.OLS(y, X[selected_features])
    final_result = final_model.fit()
    return(selected_features, final_model, final_result, vif)

def get_stepwise_selection(df, dep_variable):
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.metrics import roc_auc_score
    # Separate the target variable from the predictor variables
    X = df.drop(dep_variable, axis=1)  # Predictor variables
    y = df[dep_variable]               # Target variable
    # Add a constant column to the predictor variables
    X = sm.add_constant(X)
    # Perform step-wise model selection using stepAIC
    model = sm.Logit(y, X)
    result = model.fit(maxiter=1000)
    aic = result.aic
    bic = result.bic
    y_pred_prob = result.predict(X)
    auc = roc_auc_score(y, y_pred_prob)
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["GVIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    selected_features = stepwise_selection(X, y)
    print()
    print('LMG R^2 Decomposition:')
    get_lmg_r2_decomposition(X, y)
    print()
    final_model = sm.Logit(y, X[selected_features])
    final_result = final_model.fit()
    #print_residual_plot(final_result, 'residual_plot.png')
    #print_vif_plot(final_model, X, y, 'vif_plot.png')
    return(selected_features, final_model, final_result, vif, aic, bic, auc)

# Perform step-wise model selection using stepAIC
def stepwise_selection(X, y, initial_list=[], threshold_in=0.05, threshold_out=0.1, verbose=True):
    import statsmodels.api as sm
    import pandas as pd
    import numpy as np
    included = list(initial_list)
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=np.float64)
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit(disp=0)
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Added feature: {best_feature}    p-value: {best_pval:.4f}')
        # Backward step
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(disp=0)
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f'Removed feature: {worst_feature}    p-value: {worst_pval:.4f}')
        if not changed:
            break
    return included

def get_relevant_dataframe(subset, feature_df, dep_variable):
    # returns the dataframe subset that has relevant CMV features, Tan et al (2016) text features, and ACL 2022 features
    import pandas as pd
    feature_df = get_tan_word_features(feature_df)
    subset = pd.merge(subset, feature_df, on='id')
    df_acl2022 = get_acl2022_feature_subset(subset, dep_variable)
    full_tan2016_features = [
        'definite_article_perc',
        'indefinite_article_perc',
        'hu_liu_pos_nwords',
        'second_person',
        'hu_liu_neg_nwords',
        'hedge_words_perc',
        'first_person',
        'first_person_pl',
        'com_link_count',
        'example_count',
        'num_question',     
        'pdf_link_count',
        'edu_link_count',
        'hu_liu_pos_perc',
        'num_quotations',
        'Valence',
        'entropy',
        'sentence_count',
        'ttr',
        'paragraph_count'
    ]
    df_acl2022_tan2016 = pd.merge(df_acl2022, feature_df[['id']+full_tan2016_features], on='id', how='left')
    df = pd.merge(df_acl2022_tan2016, subset[['id','num_comments', 'score', 'edited_binary']], on='id', how='left')
    return(df)

def print_residual_plot(result, outfile):
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance
    residuals = result.resid_response
    plt.scatter(result.fittedvalues, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.savefig(outfile)

def print_vif_plot(model, X, y, outfile):
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(model, X, y, scoring='accuracy')
    feature_names = X.columns
    importance_scores = perm_importance.importances_mean
    sorted_idx = np.argsort(importance_scores)
    plt.barh(range(len(feature_names)), importance_scores[sorted_idx], align='center')
    plt.yticks(range(len(feature_names)), np.array(feature_names)[sorted_idx])
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title('Variable Importance Plot')
    plt.savefig(outfile)