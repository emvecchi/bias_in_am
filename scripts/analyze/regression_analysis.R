library(stats)
library(corrplot)

data <- read.csv('data/bias_in_AM/data_for_analysis.csv')

data$gender_source <- ifelse(data$gender_source == 'explicit', 1, 0)
data$sentiment <- ifelse(data$sentiment == 'negative', -1,
                       ifelse(data$sentiment == 'neutral', 0,
                              ifelse(data$sentiment == 'positive', 1, NA)))


# work only with author_gender (not considering explicit gender info)
df <- data[data$author_gender %in% c(0, 1), ]
df <- subset(df, select = -c(explicit_gender,aq_masked_score,masked_selftext,perc_explicit_gender_in_comments_f))

random_samples <- subset(df, author_gender == 0)[sample(nrow(subset(df, author_gender == 0)), 400), ]
subset_df <- rbind(subset(df, author_gender == 1), random_samples)


dependent_var <- 'gender_source'

filter_strings <- c('aq', 'gender', 'toxicity', 'sentiment', 'score', 'comment', 'edited')
filtered_columns <- grep(paste(filter_strings, collapse = '|'), names(subset_df))
filtered_column_names <- names(subset_df)[filtered_columns]
filtered_column_names_ivs <- setdiff(filtered_column_names, dependent_var)

cmv_extracted_feats <- subset(subset_df, select=c(filtered_column_names))
# scale and center all IVs
#cmv_extracted_feats <- cmv_extracted_feats %>% mutate_if(is.numeric, scale, scale = T)

ling_feats <- subset(subset_df, select= -filtered_columns)
ling_feats <- ling_feats[, 4:ncol(ling_feats)]
# scale and center all IVs
#ling_feats <- ling_feats %>% mutate_if(is.numeric, scale, scale = T)


#model <- lm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs, collapse = " + "), '+ (author_gender*gender_source)'), data = cmv_extracted_feats)
model <- lm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs, collapse = " + ")), data = cmv_extracted_feats)
summary(model)



#### DV = author_gender
# 60.4% explained variance
stepAIC(model)
model <- lm(formula = author_gender ~ score + edited_binary + perc_author_gender_in_comments_m + 
    perc_author_gender_in_comments_f + toxicity_neutral + sentiment_positive + 
    sentiment_neutral + sentiment_negative + gender_source, data = cmv_extracted_feats)
summary(model)
# 60.7% explained var

#### DV = aq_score
# 9.2% explained variance
stepAIC(model)
model <- lm(formula = aq_score ~ toxicity_neutral + sentiment_negative + 
    sentiment + avg_comment_aq + gender_source, data = cmv_extracted_feats)
summary(model)
# 9.9% explained var

#### DV = avg_comment_aq
# 17.7% explained variance
stepAIC(model)
model <- lm(formula = avg_comment_aq ~ author_gender + score + perc_author_gender_in_comments_m + 
    avg_comment_toxicity + sentiment_positive + sentiment_neutral + 
    sentiment + avg_comment_sentiment + aq_score + gender_source, 
    data = cmv_extracted_feats)
summary(model)
# 17.9% explained var

#### DV = gender_source
# 7.9% explained variance
stepAIC(model)
model <- lm(formula = gender_source ~ author_gender + perc_explicit_gender_in_comments_m + 
    perc_author_gender_in_comments_f + avg_comment_toxicity + 
    sentiment_positive + aq_score + avg_comment_aq, data = cmv_extracted_feats)
summary(model)
# 8.8% explained var




