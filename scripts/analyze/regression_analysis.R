library(stats)
library(corrplot)
library(dplyr)
library(MASS)
library(xtable)
library(sjmisc)
library(sjPlot)  
library(car)

get_clusters <- function(cor_matrix, threshold) {
       dist_matrix <- as.dist(1 - cor_matrix)
       if (any(is.na(dist_matrix)) || any(is.infinite(dist_matrix))) {
              dist_matrix[is.na(dist_matrix)] <- mean(dist_matrix, na.rm = TRUE)
              dist_matrix[is.infinite(dist_matrix)] <- mean(dist_matrix, na.rm = TRUE)
       }
       hclust_res <- hclust(dist_matrix, method = "ward.D2")
       clusters <- cutree(hclust_res, h = threshold)
       #pdf("clustering_and_correlation_plots.pdf")
       #plot(hclust_res, hang = -1,
       #main = "Hierarchical Clustering of Features (Spearman Correlation)")
       #corrplot(cor_matrix, method = "color")
       #dev.off()
       #return(clusters)
}

run_stepAIC <- function(model) {
       result_stepAIC <- stepAIC(model)       
       model_formula <- formula(result_stepAIC$call)
       return(model_formula)
}

get_explained_variance <- function(model){
       fit <- anova(model)
       explained_variance <-(fit[["Sum Sq" ]]/sum(fit[["Sum Sq" ]]))*100
       fit$explvar <- explained_variance
       return(fit)
}

save_plot_to_pdf <- function(plot, file_name,w,h) {
  pdf(file = file_name, width = w, height = h)
  print(plot)
  dev.off()
}

save_summary <- function(model, dependent_var, file_path) {
       writeLines('#######################################################', file_path)
       writeLines(paste('Dependent Variable:',dependent_var), file_path)
       writeLines(capture.output(summary(model)$coef[,0]), file_path)
       writeLines(capture.output(summary(model)), file_path)
       writeLines(capture.output(vif(model)), file_path)
       fit <- get_explained_variance(model)
       writeLines(capture.output(fit[order(fit$explvar, decreasing = TRUE),]))
}

update_df_for_colinearity <- function(clusters, cor_matrix_spearman, df) {
       for (cluster_id in unique(clusters)) {
              if (length(colnames(cor_matrix_spearman)[clusters == cluster_id]) > 1){
                     list_length <- length(colnames(cor_matrix_spearman)[clusters == cluster_id])
                     # Drop either all but first or all but last items in clusters with colinearity
                     if (cluster_id == 53 || cluster_id == 59 || cluster_id == 20 || cluster_id == 30 || cluster_id == 34){ 
                     columns_to_drop <- colnames(cor_matrix_spearman)[clusters == cluster_id][1:(list_length - 1)]
                     }
              else if (cluster_id < 50 || cluster_id == 55 ){
                     columns_to_drop <- colnames(cor_matrix_spearman)[clusters == cluster_id][-1]
                     }
              df <- df[, !colnames(df) %in% columns_to_drop]
              cat("Cluster", cluster_id, ":\n")
              cat(colnames(cor_matrix_spearman)[clusters == cluster_id], "\n\n")
              }
       }
       return(df)
}

data <- read.csv('data/bias_in_AM/data_for_analysis.csv')
data$gender_source <- ifelse(data$gender_source == 'explicit', 1.0, 0.0)
data$sentiment <- ifelse(data$sentiment == 'negative', -1,
                       ifelse(data$sentiment == 'neutral', 0,
                              ifelse(data$sentiment == 'positive', 1, NA)))

df <- data[data$author_gender %in% c(0, 1), ]
df <- subset(df, select = -c(explicit_gender, imperative, aq_score, avg_comment_aq, personal_pronouns, aq_masked_score, topic, sentiment_positive, toxicity_neutral))#tokenized_text
df[, 4:ncol(df)] <- lapply(df[, 4:ncol(df)], function(x) as.numeric(as.character(x)))

# identify cols with zero standard deviation
#sds <- apply(df[, 4:ncol(df)], 2, sd)
#zero_sd_vars <- names(sds[sds == 0])
#print(zero_sd_vars)

cor_matrix_spearman <- cor(df[, 4:ncol(df)], method = "spearman")
correlation_threshold <- 0.5       # correlation threshold for colinearity check
clusters <- get_clusters(cor_matrix_spearman, correlation_threshold)
df <- update_df_for_colinearity(clusters, cor_matrix_spearman, df)

# get a subset of df that only contains 200 M explicit_gender to balance data
#random_samples <- subset(df, author_gender == 0)[sample(nrow(subset(df, author_gender == 0)), 400), ]
#df <- rbind(subset(df, author_gender == 1), random_samples)

filter_strings <- c('gender', 'score', 'num_comments')
filtered_columns <- grep(paste(filter_strings, collapse = '|'), names(df))
filtered_column_names <- names(df)[filtered_columns]

file_path <- "models_summary_author_gender.txt"
summary_file <- file(file_path, "a")
counter<-1
for (dependent_var in c('author_gender', 'score', 'perc_author_gender_in_comments_m', 'perc_author_gender_in_comments_f')){
       #dependent_var <- 'author_gender'
       filtered_column_names_ivs_a <- setdiff(filtered_column_names, dependent_var)
       cmv_extracted_feats <- subset(df, select=c(filtered_column_names))
       ling_feats <- subset(df, select= -filtered_columns)
       ling_feats <- ling_feats[, 4:ncol(ling_feats)]
       combined_df <- data.frame(cmv_extracted_feats, ling_feats)
       filtered_column_names_ivs_ab <- c(filtered_column_names_ivs_a, colnames(ling_feats))
       if (dependent_var %in% c('author_gender','gender_source')){
              modelA <- glm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs_a, collapse = " + "), '+ (score*num_comments)'), data = cmv_extracted_feats, family=binomial)
              stepAICmodelA <- glm(run_stepAIC(modelA), data = cmv_extracted_feats, family=binomial)
              modelAB <- glm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs_ab, collapse = " + "), '+ (score*num_comments)'), data = combined_df, family=binomial)
              stepAICmodelAB <- glm(run_stepAIC(modelAB), data = combined_df, family=binomial)
              sig_var <- "Pr(>|z|)"
       } else {
              modelA <- lm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs_a, collapse = " + ")), data = cmv_extracted_feats)
              stepAICmodelA <- lm(run_stepAIC(modelA), data = cmv_extracted_feats)
              modelAB <- lm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs_ab, collapse = " + ")), data = combined_df)
              stepAICmodelAB <- lm(run_stepAIC(modelAB), data = combined_df)
              sig_var <- "Pr(>|t|)"
       }
       significant_featuresA <- summary(stepAICmodelA)$coefficients[summary(stepAICmodelA)$coefficients[, sig_var] < 0.05, ]
       plotA<-plot_model(stepAICmodelA, type = "std", show.p = TRUE, show.values=TRUE, 
              width = 0.1, value.size = 3, dot.size =.5, sort.est = TRUE, 
              terms = c(rownames(significant_featuresA)), 
              title = '') + theme_sjplot()
       save_plot_to_pdf(plotA, paste('groupA_',counter,'.pdf', sep=''),5,1.8)
       significant_featuresAB <- summary(stepAICmodelAB)$coefficients[summary(stepAICmodelAB)$coefficients[, sig_var] < 0.05, ]
       plotAB<-plot_model(stepAICmodelAB, type='std', show.values = TRUE, 
              width = 0.1, value.size = 3, dot.size =.5, sort.est = TRUE, 
              terms = c(rownames(significant_featuresAB)), 
              title = '') + theme_sjplot()
       save_plot_to_pdf(plotAB, paste('groupAB_',counter,'.pdf', sep=''),5,4)
       print(summary(stepAICmodelA))
       save_summary(stepAICmodelA, dependent_var, summary_file)
       print(summary(stepAICmodelAB))
       save_summary(stepAICmodelAB, dependent_var, summary_file)
       counter<-counter+1
}
close(summary_file)

