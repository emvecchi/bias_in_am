library(stats)
library(corrplot)
library(dplyr)
library(MASS)

gender_type<-'author_gender'

data <- read.csv('data/bias_in_AM/data_for_analysis.csv')

data$gender_source <- ifelse(data$gender_source == 'explicit', 1, 0)
data$sentiment <- ifelse(data$sentiment == 'negative', -1,
                       ifelse(data$sentiment == 'neutral', 0,
                              ifelse(data$sentiment == 'positive', 1, NA)))

# work only with author_gender (not considering explicit gender info)
df <- data[data$author_gender %in% c(0, 1), ]
df <- subset(df, select = -c(explicit_gender,masked_selftext)) #,aq_masked_score

# work only with explicit_gender (not considering implicit gender info)
#df <- data[data$explicit_gender %in% c(0, 1), ]
#df <- subset(df, select = -c(author_gender,aq_masked_score,masked_selftext,gender_source))

# remove columns that were shown to have colinearity issues (check script feature_selection.py)
colinear_feats <- c('basic_ntokens', 'edited_binary','basic_ntypes', 'basic_ncontent_tokens', 'basic_ncontent_types', 'basic_nfunction_tokens', 'basic_nfunction_types', 'entropy', 'sentence_count', 'ttr', 'syll_per_word', 'long_words', 'flesch', 'hu_liu_neg_nwords', 'hu_liu_pos_perc', 'perc_explicit_gender_in_comments', 'perc_author_gender_in_comments', 'toxicity_neutral', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral')
df <- df[, !(names(df) %in% colinear_feats)]

# get a subset of df that only contains 200 M explicit_gender to balance data
#random_samples <- subset(df, author_gender == 0)[sample(nrow(subset(df, author_gender == 0)), 400), ]
#subset_df <- rbind(subset(df, author_gender == 1), random_samples)
subset_df<-df   # use if you are not random sampling a subset of M instances

dependent_var <- 'gender_source'

filter_strings <- c('gender', 'score', 'num_comments')
filtered_columns <- grep(paste(filter_strings, collapse = '|'), names(subset_df))
filtered_column_names <- names(subset_df)[filtered_columns]
filtered_column_names_ivs <- setdiff(filtered_column_names, dependent_var)
#filtered_column_names_ivs <- setdiff(names(subset_df[, -c(1:3)]), dependent_var)


cmv_extracted_feats <- subset(subset_df, select=c(filtered_column_names))
# scale and center all IVs
#cmv_extracted_feats <- cmv_extracted_feats %>% mutate_if(is.numeric, scale, scale = T)

ling_feats <- subset(subset_df, select= -filtered_columns)
ling_feats <- ling_feats[, 4:ncol(ling_feats)]
# scale and center all IVs
#ling_feats <- ling_feats %>% mutate_if(is.numeric, scale, scale = T)

#combined_df <- data.frame(cmv_extracted_feats, ling_feats)

#model <- lm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs, collapse = " + "), '+ (author_gender*gender_source)'), data = cmv_extracted_feats)
#model <- glm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs, collapse = " + ")), data = cmv_extracted_feats, family=binomial)
#model <- glm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs, collapse = " + ")), subset_df[, 4:ncol(subset_df)], family=binomial)
#model <- lm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs, collapse = " + ")), data = cmv_extracted_feats)
#model <- lm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs, collapse = " + ")), data = subset_df[, 4:ncol(subset_df)])
result_stepAIC <- stepAIC(model)
model_formula <- formula(result_stepAIC$call)
#stepAICmodel <- lm(model_formula, data = cmv_extracted_feats)
#stepAICmodel <- glm(model_formula, data = subset_df[, 4:ncol(subset_df)])
summary(stepAICmodel)



library(sjPlot)
plot<-plot_model(stepAICmodel, show.values = TRUE, width = 0.1, value.size = 3, dot.size =1)+
  ylab("Group A+B Features")
pdf("author_gender_groupAB.pdf", width = 8, height = 6)
print(plot)
dev.off() 

