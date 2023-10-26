library(stats)
library(corrplot)
library(dplyr)
library(MASS)
library(sjmisc)
library(sjPlot)  
library(car)
library(ggplot2)
require(rms)

get_clusters <- function(cor_matrix, threshold) {
       dist_matrix <- as.dist(1 - cor_matrix)
       if (any(is.na(dist_matrix)) || any(is.infinite(dist_matrix))) {
              dist_matrix[is.na(dist_matrix)] <- mean(dist_matrix, na.rm = TRUE)
              dist_matrix[is.infinite(dist_matrix)] <- mean(dist_matrix, na.rm = TRUE)
       }
       hclust_res <- hclust(dist_matrix, method = "ward.D2")
       clusters <- cutree(hclust_res, h = threshold)
       return(clusters)
}

run_stepAIC <- function(model) {
       result_stepAIC <- stepAIC(model)       
       model_formula <- formula(result_stepAIC$call)
       return(model_formula)
}
run_stepAIC_interaction <- function(model) {
       result_stepAIC <- stepAIC(model, scope = . ~ .^2)       
       model_formula <- formula(result_stepAIC$call)
       return(model_formula)
}

get_explained_variance <- function(model, dependent_var){
       fit <- anova(model)
       if (dependent_var == 'extCMVGender'){
              explained_variance <- (fit[["Deviance" ]]/deviance(model))*100
       } else {
              explained_variance <-(fit[["Sum Sq" ]]/sum(fit[["Sum Sq" ]]))*100
       }
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
       fit <- get_explained_variance(model, dependent_var)
       writeLines(capture.output(fit[order(fit$explvar, decreasing = TRUE),]))
}

update_df_for_colinearity <- function(clusters, cor_matrix_spearman, df) {
       for (cluster_id in unique(clusters)) {
              if (length(colnames(cor_matrix_spearman)[clusters == cluster_id]) > 1){
                     list_length <- length(colnames(cor_matrix_spearman)[clusters == cluster_id])
                     # Drop either all but first or all but last items in clusters with colinearity
                     if (cluster_id == 14 || cluster_id == 41 || cluster_id == 57 || cluster_id == 63){ 
                     #if (cluster_id == 53 || cluster_id == 59 || cluster_id == 20 || cluster_id == 30 || cluster_id == 34){ 
                            columns_to_drop <- colnames(cor_matrix_spearman)[clusters == cluster_id][1:(list_length - 1)]
                     }
                     else if (cluster_id < 54 || cluster_id == 59 ){
                     #else if (cluster_id < 50 || cluster_id == 55 ){
                            columns_to_drop <- colnames(cor_matrix_spearman)[clusters == cluster_id][-1]
                     }
              df <- df[, !colnames(df) %in% columns_to_drop]
              cat("Cluster", cluster_id, ":\n")
              cat(colnames(cor_matrix_spearman)[clusters == cluster_id], "\n\n")
              }
       }
       return(df)
}

get_plot <- function(model, significant_features, dependent_var){
       plot<-plot_model(model, type = 'std', sort.est = TRUE, 
              show.values = TRUE, width = 0.05, 
              vline.color="black", terms = c(rownames(significant_features))) + 
              theme(
                     text = element_text(size = 10),
                     axis.text.y = element_text(size = 15),
                     panel.background = element_rect(fill = 'white', color = 'white'),
                     strip.text = element_text(size = 15), # Set text size for the strip (above the plot)
                     strip.background = element_blank(),    # Remove background for the strip
                     strip.text.x = element_text(hjust = 0), # Adjust horizontal alignment of strip text
                     panel.grid.major = element_line(color = "gray", linetype = "dashed"), # Add major grid lines
                     panel.grid.minor = element_blank()
                     ) +
              labs(title = dependent_var)  # Custom title above the grid plot
       return(plot)
}


data <- read.csv('data/bias_in_AM/data_for_analysis2.csv')
data$gender_source <- ifelse(data$gender_source == 'explicit', 1.0, 0.0)
data$sentiment <- ifelse(data$sentiment == 'negative', -1,
                       ifelse(data$sentiment == 'neutral', 0,
                              ifelse(data$sentiment == 'positive', 1, NA)))

df <- data[data$author_gender %in% c(0, 1), ]
# remove cols that are irrelevant (eg gender_source, *aq*) or those with zero standard deviation (eg pdf_link_count, imperative, PRON)
df <- subset(df, select = -c(explicit_gender, aq_score, avg_comment_aq, personal_pronouns, aq_masked_score, topic, sentiment_positive, toxicity_neutral, deltas, avg_comment_deltas, author_flair_text))

# identify cols with zero standard deviation
#sds <- apply(df[, 4:ncol(df)], 2, sd)
#zero_sd_vars <- names(sds[sds == 0])
#print(zero_sd_vars)

cor_matrix_spearman <- cor(df[, 4:ncol(df)], method = "spearman")
correlation_threshold <- 0.5       # correlation threshold for colinearity check
clusters <- get_clusters(cor_matrix_spearman, correlation_threshold)
df <- update_df_for_colinearity(clusters, cor_matrix_spearman, df)

# For printout of analyses using approp feature names
df <- df %>% 
       rename("extCMVGender" = "author_gender",
               "extCMVGender_in_comments_f" = "perc_author_gender_in_comments_f",
               "extCMVGender_in_comments_m"= "perc_author_gender_in_comments_m",
               "CMVGender_in_comments_m" = "perc_explicit_gender_in_comments_m",
               "CMVGender_in_comments_f" = "perc_explicit_gender_in_comments_f",
               "edited" = "edited_binary",
               "Lex_Decision_Accuracy" = "LD_Mean_Accuracy",
               "hypernomy_verb_noun" = "hyper_verb_noun_Sav_Pav",
               "KL_divergence_rel_entropy" = "McD_CD"
               )

filter_strings <- c('ender', 'score', 'num_comments')
filtered_columns <- grep(paste(filter_strings, collapse = '|'), names(df))
filtered_column_names <- names(df)[filtered_columns]

file_path <- "models_summary_extCMVGender.txt"
summary_file <- file(file_path, "a")
counter<-1
for (dependent_var in c('extCMVGender', 'score', 'extCMVGender_in_comments_m', 'extCMVGender_in_comments_f')){
       filtered_column_names_ivs_a <- setdiff(filtered_column_names, dependent_var)
       cmv_extracted_feats <- subset(df, select=c(filtered_column_names))
       ling_feats <- subset(df, select= -filtered_columns)
       ling_feats <- ling_feats[, 4:ncol(ling_feats)]
       combined_df <- data.frame(cmv_extracted_feats, ling_feats)
       filtered_column_names_ivs_ab <- c(filtered_column_names_ivs_a, colnames(ling_feats))
       if (dependent_var %in% c('extCMVGender','gender_source')){
              modelA <- glm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs_a, collapse = " + "), '+ (score*num_comments)'), data = cmv_extracted_feats, family=binomial)
              stepAICmodelA <- glm(run_stepAIC_interaction(modelA), data = cmv_extracted_feats, family=binomial)
              pR2A = 1 - stepAICmodelA$deviance / stepAICmodelA$null.deviance 
              modelAB <- glm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs_ab, collapse = " + "), '+ (score*num_comments)'), data = combined_df, family=binomial)
              stepAICmodelAB <- glm(run_stepAIC(modelAB), data = combined_df, family=binomial)
              pR2AB = 1 - stepAICmodelAB$deviance / stepAICmodelAB$null.deviance 
       } else {
              modelA <- lm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs_a, collapse = " + ")), data = cmv_extracted_feats)
              stepAICmodelA <- lm(run_stepAIC_interaction(modelA), data = cmv_extracted_feats)
              modelAB <- lm(formula = paste(dependent_var, ' ~ ', paste(filtered_column_names_ivs_ab, collapse = " + ")), data = combined_df)
              stepAICmodelAB <- lm(run_stepAIC(modelAB), data = combined_df)
       }
       significant_featuresA <- summary(stepAICmodelA)$coefficients[summary(stepAICmodelA)$coefficients[, 4] < 0.01, ]
       plotA<-get_plot(stepAICmodelA, significant_featuresA, dependent_var)
       save_plot_to_pdf(plotA, paste('groupA_',counter,'.pdf', sep=''),6,3)
       significant_featuresAB <- summary(stepAICmodelAB)$coefficients[summary(stepAICmodelAB)$coefficients[, 4] < 0.05, ]
       features_to_remove <- c('positive_verbs_component', 'well_being_component')
       significant_featuresAB <- significant_featuresAB[!(rownames(significant_featuresAB) %in% features_to_remove), , drop = FALSE]
       plotAB<-get_plot(stepAICmodelAB, significant_featuresAB, dependent_var)
       save_plot_to_pdf(plotAB, paste('groupAB_',counter,'.pdf', sep=''),8,5.5)
       print(summary(stepAICmodelA))
       save_summary(stepAICmodelA, dependent_var, summary_file)
       print(summary(stepAICmodelAB))
       save_summary(stepAICmodelAB, dependent_var, summary_file)
       counter<-counter+1
}
close(summary_file)
