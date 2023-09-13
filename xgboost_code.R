
library(xgboost)
library(caret)
library(DiagrammeR)
library(tidyverse)

#### near zero variance and 0.75 correlation cutoff
df_cor <- all_data %>% 
  select( -c(stcofips, st, state, st_abbr, county, alz_spending, pop, glyphosate, dicamba, eal_score,
             resl_score, veteran, vacant, resp, ptsdf, ozone, dslpm))

cormat_df <- round(cor(df_cor, use = "pairwise.complete.obs"), 2)

remove_cols <- nearZeroVar(df_cor, names = TRUE, freqCut = 2, uniqueCut = 20)
remove_cols

findCorrelation(cormat_df, cutoff = 0.75, names = T, verbose = T)

###hospital, incarcerate, syphyllis, teeth, housingcost, smoking, hbp, nocomputer, hincome, snap

str(all_data)

df_cor$physician

summary(df_cor)


#split train and test 80% and 20%
alz_model_cut <- df_cor %>% 
  select(-c(hospital, incarcerate, syphilis, teeth, housingcost, mental, hbp, hincome, nocomputer, snap, log_homevalue))

dmy_cut <- dummyVars(" ~ .", data = alz_model_cut,fullRank = T)
alz_model_cut <- data.frame(predict(dmy_cut, newdata = alz_model_cut))
alz_model_data_complete_cut <- alz_model_cut %>% 
  na.omit()

#encoding categorical metro

##################                              ############################
##################Random Forest Added by Request############################
##################                              ############################
############################################################################



install.packages("randomForest")
library(randomForest)

# Build the random forest regression model
set.seed(500)
rffit <- randomForest(
  formula = alzheimers_dementia ~ .,
  data = alz_train_cut,
  ntree = 2000,
  mtry = 5,
  importance = TRUE
)

# Make predictions on the testing data
predictions_rf <- predict(rffit, newdata = alz_test_cut)

# Calculate the RMSE and R-squared
predicted_rf = predict(rffit, newdata = alz_test_cut )
residuals_rf = alz_test_y_cut - predicted_rf
RMSE_rf = sqrt(mean(residuals^2))
cat('The root mean square error of the test data is ', round(RMSE_rf,3),'\n')

y_test_mean_rf = mean(alz_test_y_cut)
# Calculate total sum of squares
tss_rf =  sum((alz_test_y_cut - y_test_mean_rf)^2 )
# Calculate residual sum of squares
rss_rf =  sum(residuals_rf^2)
# Calculate R-squared
rsq_rf  =  1 - (rss_rf/tss_rf)
cat('The R-square of the test data is ', round(rsq_rf,3), '\n')

imp_rf <- as.data.frame(importance(rffit))
imp_rf$Var.Names <- row.names(imp_rf)
importance_rf <- as_tibble(imp_rf) %>% 
  rename(Gain = `IncNodePurity`,
         Feature = Var.Names) %>% 
  arrange(desc(Gain)) %>% 
  filter(row_number() <= 15 )

importance_plot_rf <- ggplot(importance_rf, aes(x = Gain, y = reorder(Feature, Gain)))+
  geom_col(fill = "grey", color = "black")+
  theme_classic()+
  labs(x = "Importance (Gain)", y = "Feature")+
  ggtitle("Feature Importance")+
  theme(plot.title = element_text(hjust = 0.5),
        axis.text=element_text(size=12),
        axis.title=element_text(size=12))+
  scale_x_continuous(labels = scales::percent)
importance_plot_rf

############################################################################
############################################################################
############################################################################
############################################################################



#split train and test 80% and 20%
set.seed(0978)
intrain_alz_cut <-  createDataPartition(alz_model_data_complete_cut$alzheimers_dementia, p = 0.8, list = F)
alz_train_cut <-  alz_model_data_complete_cut[intrain_alz_cut, ]
alz_test_cut <-  alz_model_data_complete_cut[-intrain_alz_cut, ]

alz_train_x_cut <-  data.matrix(alz_train_cut[,-1])
alz_train_y_cut <-  alz_train_cut[,1]

alz_test_x_cut <-  data.matrix(alz_test_cut[,-1])
alz_test_y_cut <-  alz_test_cut[,1]

#create matrix for xgboost
alz_xgboost_train_cut = xgb.DMatrix(data=alz_train_x_cut, label=alz_train_y_cut)
alz_xgboost_test_cut = xgb.DMatrix(data=alz_test_x_cut, label=alz_test_y_cut)

#default parameters xgboost
xgb_params_cut = list(
  booster = "gbtree",
  eta = 0.3, 
  gamma = 0,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 1,
  colsample_bytree = 1,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
# rmse for train and test displayed
watchlist_cut <- list(train = alz_xgboost_train_cut, test = alz_xgboost_test_cut)

#xgboost for model
xgb_model_cut <- xgb.train(
  data = alz_xgboost_train_cut,
  nrounds = 1000,
  watchlist = watchlist_cut,
  params = xgb_params_cut
)

xgb_model_cut
#model importance
importance_cut <- xgb.importance(feature_names = colnames(alz_train_x_cut), model = xgb_model_cut)

pred_cut <- predict(xgb_model_cut, alz_xgboost_test_cut)
RMSE(pred_cut,alz_test_y_cut )
xgb.plot.importance(importance_cut)



elog_cut <- xgb_model_cut$evaluation_log
ggplot(data = elog_cut, aes(x = iter)) +
  geom_line(aes(y = elog_cut$train_rmse), col = "blue") +
  geom_line(aes(y = elog_cut$test_rmse), col = "red")

predicted = predict(xgb_model_cut, alz_test_x_cut )
residuals = alz_test_y_cut - predicted
RMSE = sqrt(mean(residuals^2))
cat('The root mean square error of the test data is ', round(RMSE,3),'\n')

y_test_mean = mean(alz_test_y_cut)
# Calculate total sum of squares
tss =  sum((alz_test_y_cut - y_test_mean)^2 )
# Calculate residual sum of squares
rss =  sum(residuals^2)
# Calculate R-squared
rsq  =  1 - (rss/tss)
cat('The R-square of the test data is ', round(rsq,3), '\n')

#################################
###now tuning the dataset. iterative grid search starting with primary tuning of eta and max_depth
set.seed=(777)
xgb_grid_1_cut = expand.grid(
  nrounds = c(5000),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)



xgb_trcontrol_1_cut = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE,
  search = "grid"
)

xgb_train_1_cut = train(
  x = alz_train_x_cut,
  y = alz_train_y_cut,
  trControl = xgb_trcontrol_1_cut,
  tuneGrid = xgb_grid_1_cut,
  method = "xgbTree"
)





#now tuning min child weight with parameters from search 1
xgb_grid_2_cut <- expand.grid(
  nrounds = xgb_train_1_cut$bestTune$nrounds,
  eta = xgb_train_1_cut$bestTune$eta,
  max_depth = ifelse(xgb_train_1_cut$bestTune$max_depth == 2,
                     c(xgb_train_1_cut$bestTune$max_depth:4),
                     xgb_train_1_cut$bestTune$max_depth - 1:xgb_train_1_cut$bestTune$max_depth + 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3,4,5),
  subsample = 1
)
?train
xgb_train_2_cut = train(
  x = alz_train_x_cut,
  y = alz_train_y_cut,
  trControl = xgb_trcontrol_1_cut,
  tuneGrid = xgb_grid_2_cut,
  method = "xgbTree"
)

#now tuning subsample and col sample by tree using parameters from search 1 and 2
xgb_grid_3_cut <- expand.grid(
  nrounds = xgb_train_2_cut$bestTune$nrounds,
  eta = xgb_train_2_cut$bestTune$eta,
  max_depth = xgb_train_2_cut$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_train_2_cut$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_train_3_cut = train(
  x = alz_train_x_cut,
  y = alz_train_y_cut,
  trControl = xgb_trcontrol_1_cut,
  tuneGrid = xgb_grid_3_cut,
  method = "xgbTree"
)

#now tuning gamma using parameters from search 1, 2, and 3
xgb_grid_4_cut <- expand.grid(
  nrounds = xgb_train_3_cut$bestTune$nrounds,
  eta = xgb_train_3_cut$bestTune$eta,
  max_depth = xgb_train_3_cut$bestTune$max_depth,
  gamma = c(0, 1, 2, 3, 4, 5, 6),
  colsample_bytree = xgb_train_3_cut$bestTune$colsample_bytree,
  min_child_weight = xgb_train_3_cut$bestTune$min_child_weight,
  subsample = xgb_train_3_cut$bestTune$subsample
)


xgb_train_4_cut = train(
  x = alz_train_x_cut,
  y = alz_train_y_cut,
  trControl = xgb_trcontrol_1_cut,
  tuneGrid = xgb_grid_4_cut,
  method = "xgbTree"
)


#now further tuning of eta using parameters from searches 1-4
xgb_grid_5_cut <- expand.grid(
  nrounds = xgb_train_4_cut$bestTune$nrounds,
  eta = c(0.001, 0.005,0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_train_4_cut$bestTune$max_depth,
  gamma = xgb_train_4_cut$bestTune$gamma,
  colsample_bytree = xgb_train_4_cut$bestTune$colsample_bytree,
  min_child_weight = xgb_train_4_cut$bestTune$min_child_weight,
  subsample = xgb_train_4_cut$bestTune$subsample
)

xgb_train_5_cut = train(
  x = alz_train_x_cut,
  y = alz_train_y_cut,
  trControl = xgb_trcontrol_1_cut,
  tuneGrid = xgb_grid_5_cut,
  method = "xgbTree"
)


xgb_train_5_cut



### cut tune xgboost

###chosen parameters


xgb_params_cut_tune = list(
  booster = "gbtree",
  eta = 0.015,
  gamma = 6,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.75,
  colsample_bytree = 0.4,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
# rmse for train and test displayed
watchlist_cut_tune <- list(train = alz_xgboost_train_cut, test = alz_xgboost_test_cut)

set.seed(500)
#xgboost for model
xgb_model_cut_tune <- xgb.train(
  data = alz_xgboost_train_cut,
  nrounds = 5000,
  watchlist = watchlist_cut_tune,
  params = xgb_params_cut_tune,
  early_stopping_rounds = 50
)

xgb_model_cut_tune


#model importance
importance_cut_tune <- xgb.importance(feature_names = colnames(alz_train_x_cut), model = xgb_model_cut_tune)


predicted = predict(xgb_model_cut_tune, alz_test_x_cut )
residuals = alz_test_y_cut - predicted
RMSE = sqrt(mean(residuals^2))
cat('The root mean square error of the test data is ', round(RMSE,3),'\n')

y_test_mean = mean(alz_test_y_cut)
# Calculate total sum of squares
tss =  sum((alz_test_y_cut - y_test_mean)^2 )
# Calculate residual sum of squares
rss =  sum(residuals^2)
# Calculate R-squared
rsq  =  1 - (rss/tss)
cat('The R-square of the test data is ', round(rsq,3), '\n')

library(ggpubr)

my_data = as.data.frame(cbind(predicted = predicted,
                              observed = alz_test_y_cut))
# Plot predictions vs test data
ggplot(my_data,aes(predicted, observed)) + geom_point(color = "black", alpha = 0.5) + 
  geom_smooth(method=lm, color = "black", alpha = 0.5) +
  xlab("Predicted ADRD Prevalence ") + ylab("Observed ADRD Prevalence") + 
  theme(plot.title = element_text(color="black",size=16,hjust = 0.5),
        axis.text.y = element_text(size=12), axis.text.x = element_text(size=12,hjust=.5),
        axis.title.x = element_text(size=14), axis.title.y = element_text(size=14))+
  stat_cor(aes(label =  ..rr.label..))+
  scale_y_continuous(labels = scales::label_percent(scale = 1),
                     breaks = seq(0, 24, by = 2))+
  scale_x_continuous(labels = scales::label_percent(scale = 1),
                     breaks = seq(0, 24, by = 2))
ggsave(filename = "Figure 3 try 10.tiff",
       dpi = 1200)
dev.off()
?stat_cor
library(ggplot2)

xgb.plot.tree(model = xgb_model_cut_tune, trees = c(25,39))
?xgb.plot.tree

elog_tune <- xgb_model_cut_tune$evaluation_log %>% 
  rename(train = "train_rmse",
         test = "test_rmse") %>% 
  pivot_longer(c(train, test), names_to = "Data", values_to = "RMSE")

elog_tune_cut <- ggplot(elog_tune, aes(x = iter, y= RMSE, group = Data))+
  geom_line(aes(color = Data))+
  scale_color_manual(values=c("#eeb211", "#46166b", "#eeb211"))+
  scale_y_continuous(n.breaks = 10)+
  theme_classic()+
  labs(x = "Number of Trees")+
  ggtitle("Learning Curves for Training and Testing Data")+
  theme(plot.title = element_text(hjust = 0.5))

#learning curves
elog_tune_cut


importance_cut_tune_df <- as_tibble(importance_cut_tune) %>% 
  arrange(desc(Gain)) %>% 
  filter(row_number() <= 15 )

importance_plot_cut <- ggplot(importance_cut_tune_df, aes(x = Gain, y = reorder(Feature, Gain)))+
  geom_col(fill = "grey", color = "black")+
  theme_classic()+
  labs(x = "Importance (Gain)", y = "Feature")+
  ggtitle("Feature Importance")+
  theme(plot.title = element_text(hjust = 0.5),
        axis.text=element_text(size=12),
        axis.title=element_text(size=12))+
  scale_x_continuous(labels = scales::percent)
importance_plot_cut


xgb.plot.tree( model =xgb_model_cut_tune, trees = c(12:13) )
installed.packages("SHAPforxgboost")
library(SHAPforxgboost)

dev.off()

?shap.plot.summary

shap_alz_tune <- shap.prep(xgb_model_cut_tune, X_train = alz_train_x_cut)

shap.plot.summary(shap_alz_tune)
#shap.plot.summary.wrap1(xgb_model_cut_tune, alz_train_x_cut, top_n = 20)
#?shap.plot.summary.wrap1
shap.plot.dependence(data_long = shap_alz_tune, x="sleep",
                     add_hist = F, dilute = F)

shap.plot.dependence(data_long = shap_alz_tune, x="fruit",
                     add_hist = F, dilute = F)

shap.plot.dependence(data_long = shap_alz_tune, x="lesshighschool",
                     add_hist = F, dilute = F)

shap.plot.dependence(data_long = shap_alz_tune, x="black",
                     add_hist = F, dilute = F)

shap.plot.dependence(data_long = shap_alz_tune, x="socialvul",
                     add_hist = F, dilute = F)

shap.plot.dependence(data_long = shap_alz_tune, x="socialvul",
                     add_hist = F, dilute = F)

shap.plot.dependence(data_long = shap_alz_tune, x="white",
                     add_hist = TRUE, dilute = F)

shap.plot.dependence(data_long = shap_alz_tune, x="social_assoc",
                     add_hist = TRUE, dilute = F)

shap.plot.dependence(data_long = shap_alz_tune, x="pre1960pct",
                     add_hist = F, dilute = F)

shap.plot.dependence(data_long = shap_alz_tune, x="hisp",
                     add_hist = F, dilute = F)

shap_int <- shap.prep.interaction(xgb_model_cut_tune, X_train = alz_train_x_cut)

shap.plot.dependence(data_long = shap_alz_tune,
                     data_int = shap_int,
                     x="black",
                     y = "hisp",
                     color_feature = "hisp")

?shap.plot.dependence


shap.plot.summary.wrap1(xgb_model_cut_tune, alz_test_x_cut, top_n = 20)
shap_alz_importance <- shap.importance(shap_alz_tune, names_only = FALSE, top_n = 15)
shap_alz_importance

importance_plot_shap <- ggplot(shap_alz_importance, aes(x = mean_abs_shap, y = reorder(variable, mean_abs_shap)))+
  geom_col(fill = "#eeb211", color = "#46166b")+
  theme_classic()+
  labs(x = "Importance (mean|SHAP|)", y = "Feature")+
  ggtitle("Feature Importance")+
  theme(plot.title = element_text(hjust = 0.5),
        axis.text=element_text(size=12),
        axis.title=element_text(size=12))
importance_plot_shap


########################################


#     Analysis with poverty quantiles #


########################################


all_data_poverty_quantiles <- all_data %>% 
  mutate(poverty_quantile = ntile(poverty, 4)) 


poverty_high <- all_data_poverty_quantiles %>% 
  filter(poverty_quantile == 4)

poverty_high_cor <- poverty_high %>% 
  select( -c(stcofips, st, state, st_abbr, county, alz_spending, pop, glyphosate, dicamba, eal_score,
             resl_score, veteran, vacant, resp, ptsdf, ozone, dslpm, poverty_quantile, poverty))



alz_pov_high_cut <- poverty_high_cor %>% 
  select(-c(hospital, incarcerate, syphilis, teeth, housingcost, mental, hbp, hincome, nocomputer, snap, log_homevalue))


dmy_cut_pov_high <- dummyVars(" ~ .", data = alz_pov_high_cut,fullRank = T)
alz_pov_high_cut <- data.frame(predict(dmy_cut_pov_high, newdata = alz_pov_high_cut))
alz_pov_high_cut_complete <- alz_pov_high_cut%>% 
  na.omit()


set.seed(900)
intrain_pov_high_cut <-  createDataPartition(alz_pov_high_cut_complete$alzheimers_dementia, p = 0.8, list = F)
alz_train_cut_pov_high <-  alz_pov_high_cut_complete[intrain_pov_high_cut, ]
alz_test_cut_pov_high <-  alz_pov_high_cut_complete[-intrain_pov_high_cut, ]

alz_train_x_cut_pov_high <-  data.matrix(alz_train_cut_pov_high[,-1])
alz_train_y_cut_pov_high <-  alz_train_cut_pov_high[,1]

alz_test_x_cut_pov_high <-  data.matrix(alz_test_cut_pov_high[,-1])
alz_test_y_cut_pov_high <-  alz_test_cut_pov_high[,1]

#create matrix for xgboost
alz_xgboost_train_cut_pov_high = xgb.DMatrix(data=alz_train_x_cut_pov_high, label=alz_train_y_cut_pov_high)
alz_xgboost_test_cut_pov_high = xgb.DMatrix(data=alz_test_x_cut_pov_high, label=alz_test_y_cut_pov_high)

#default parameters xgboost
xgb_params_cut_pov_high = list(
  booster = "gbtree",
  eta = 0.3, 
  gamma = 0,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 1,
  colsample_bytree = 1,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
# rmse for train and test displayed
watchlist_cut_pov_high <- list(train = alz_xgboost_train_cut_pov_high, test = alz_xgboost_test_cut_pov_high)

#xgboost for model
xgb_model_cut_pov_high <- xgb.train(
  data = alz_xgboost_train_cut_pov_high,
  nrounds = 1000,
  watchlist = watchlist_cut_pov_high,
  params = xgb_params_cut_pov_high
)

xgb_model_cut_pov_high
#model importance
importance_cut_pov_high <- xgb.importance(feature_names = colnames(alz_train_x_cut_pov_high), model = xgb_model_cut_pov_high)



set.seed=(1234)
xgb_grid_1_cut_pov_high = expand.grid(
  nrounds = c(5000),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)



xgb_trcontrol_1_cut_pov_high = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE,
  search = "grid"
)

xgb_train_1_cut_pov_high = train(
  x = alz_train_x_cut_pov_high,
  y = alz_train_y_cut_pov_high,
  trControl = xgb_trcontrol_1_cut_pov_high,
  tuneGrid = xgb_grid_1_cut_pov_high,
  method = "xgbTree"
)






xgb_grid_2_cut_pov_high <- expand.grid(
  nrounds = xgb_train_1_cut_pov_high$bestTune$nrounds,
  eta = xgb_train_1_cut_pov_high$bestTune$eta,
  max_depth = ifelse(xgb_train_1_cut_pov_high$bestTune$max_depth == 2,
                     c(xgb_train_1_cut_pov_high$bestTune$max_depth:4),
                     xgb_train_1_cut_pov_high$bestTune$max_depth - 1:xgb_train_1_cut_pov_high$bestTune$max_depth + 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3,4,5),
  subsample = 1
)

xgb_train_2_cut_pov_high = train(
  x = alz_train_x_cut_pov_high,
  y = alz_train_y_cut_pov_high,
  trControl = xgb_trcontrol_1_cut_pov_high,
  tuneGrid = xgb_grid_2_cut_pov_high,
  method = "xgbTree"
)


xgb_grid_3_cut_pov_high <- expand.grid(
  nrounds = xgb_train_2_cut_pov_high$bestTune$nrounds,
  eta = xgb_train_2_cut_pov_high$bestTune$eta,
  max_depth = xgb_train_2_cut_pov_high$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_train_2_cut_pov_high$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_train_3_cut_pov_high = train(
  x = alz_train_x_cut_pov_high,
  y = alz_train_y_cut_pov_high,
  trControl = xgb_trcontrol_1_cut_pov_high,
  tuneGrid = xgb_grid_3_cut_pov_high,
  method = "xgbTree"
)


xgb_grid_4_cut_pov_high <- expand.grid(
  nrounds = xgb_train_3_cut_pov_high$bestTune$nrounds,
  eta = xgb_train_3_cut_pov_high$bestTune$eta,
  max_depth = xgb_train_3_cut_pov_high$bestTune$max_depth,
  gamma = c(0,1,2,3,4,5,6),
  colsample_bytree = xgb_train_3_cut_pov_high$bestTune$colsample_bytree,
  min_child_weight = xgb_train_3_cut_pov_high$bestTune$min_child_weight,
  subsample = xgb_train_3_cut_pov_high$bestTune$subsample
)


xgb_train_4_cut_pov_high = train(
  x = alz_train_x_cut_pov_high,
  y = alz_train_y_cut_pov_high,
  trControl = xgb_trcontrol_1_cut_pov_high,
  tuneGrid = xgb_grid_4_cut_pov_high,
  method = "xgbTree"
)



xgb_grid_5_cut_pov_high <- expand.grid(
  nrounds = xgb_train_4_cut_pov_high$bestTune$nrounds,
  eta = c(0.001, 0.005,0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_train_4_cut_pov_high$bestTune$max_depth,
  gamma = xgb_train_4_cut_pov_high$bestTune$gamma,
  colsample_bytree = xgb_train_4_cut_pov_high$bestTune$colsample_bytree,
  min_child_weight = xgb_train_4_cut_pov_high$bestTune$min_child_weight,
  subsample = xgb_train_4_cut_pov_high$bestTune$subsample
)

xgb_train_5_cut_pov_high = train(
  x = alz_train_x_cut_pov_high,
  y = alz_train_y_cut_pov_high,
  trControl = xgb_trcontrol_1_cut_pov_high,
  tuneGrid = xgb_grid_5_cut_pov_high,
  method = "xgbTree"
)





### cut tune xgboost

xgb_params_cut_tune_pov_high = list(
  booster = "gbtree",
  eta = xgb_train_5_cut_pov_high$bestTune$eta, 
  gamma = xgb_train_5_cut_pov_high$bestTune$gamma,
  max_depth = xgb_train_5_cut_pov_high$bestTune$max_depth,
  min_child_weight =xgb_train_5_cut_pov_high$bestTune$min_child_weight,
  subsample = xgb_train_5_cut_pov_high$bestTune$subsample,
  colsample_bytree = xgb_train_5_cut_pov_high$bestTune$colsample_bytree,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
# rmse for train and test displayed
watchlist_cut_tune_pov_high <- list(train = alz_xgboost_train_cut_pov_high, test = alz_xgboost_test_cut_pov_high)

set.seed(8009)
#xgboost for model
xgb_model_cut_tune_pov_high <- xgb.train(
  data = alz_xgboost_train_cut_pov_high,
  nrounds = 5000,
  watchlist = watchlist_cut_tune_pov_high,
  params = xgb_params_cut_tune_pov_high,
  early_stopping_rounds = 50
)

xgb_model_cut_tune_pov_high

#model importance
importance_cut_tune_pov_high <- xgb.importance(feature_names = colnames(alz_train_x_cut_pov_high), model = xgb_model_cut_tune_pov_high)




elog_pov_high <- xgb_model_cut_tune_pov_high$evaluation_log %>% 
  rename(train = "train_rmse",
         test = "test_rmse") %>% 
  pivot_longer(c(train, test), names_to = "Data", values_to = "RMSE")

elog_tune_cut_pov_high <- ggplot(elog_pov_high, aes(x = iter, y= RMSE, group = Data))+
  geom_line(aes(color=Data))+
  scale_color_manual(values=c("#eeb211", "#46166b", "#eeb211"))+
  scale_y_continuous(n.breaks = 10)+
  theme_classic()+
  labs(x = "Training Iterations")+
  ggtitle("Learning Curves for Training and Testing Data Stratified for High Poverty %")+
  theme(plot.title = element_text(hjust = 0.5))


elog_tune_cut_pov_high




importance_cut_tune_pov_high_df <- as_tibble(importance_cut_tune_pov_high) %>% 
  arrange(desc(Gain)) %>% 
  filter(row_number() <= 15)

importance_plot_cut_pov_high <- ggplot(importance_cut_tune_pov_high_df, aes(x = Gain, y = reorder(Feature, Gain)))+
  geom_col(fill = "grey", color = "black")+
  theme_classic()+
  labs(x = "Importance (Gain)", y = "Feature")+
  theme(plot.title = element_text(hjust = 0.5),
        axis.text=element_text(size=12),
        axis.title=element_text(size=12))

importance_plot_cut_pov_high


shap_pov_high <- shap.prep(xgb_model_cut_tune_pov_high, X_train = alz_train_x_cut_pov_high)

shap.plot.summary.wrap1(xgb_model_cut_tune_pov_high, alz_test_x_cut_pov_high, top_n = 20)











########## low poverty

poverty_low <- all_data_poverty_quantiles %>% 
  filter(poverty_quantile == 1)

poverty_low_cor <- poverty_low %>% 
  select( -c(stcofips, st, state, st_abbr, county, alz_spending, pop, glyphosate, dicamba, eal_score,
             resl_score, veteran, vacant, resp, ptsdf, ozone, dslpm, poverty_quantile, poverty))



alz_pov_low_cut <- poverty_low_cor %>% 
  select(-c(hospital, incarcerate, syphilis, teeth, housingcost, mental, hbp, hincome, nocomputer, snap, log_homevalue))


dmy_cut_pov_low <- dummyVars(" ~ .", data = alz_pov_low_cut,fullRank = T)
alz_pov_low_cut <- data.frame(predict(dmy_cut_pov_low, newdata = alz_pov_low_cut))
alz_pov_low_cut_complete <- alz_pov_low_cut%>% 
  na.omit()


set.seed(900)
intrain_pov_low_cut <-  createDataPartition(alz_pov_low_cut_complete$alzheimers_dementia, p = 0.8, list = F)
alz_train_cut_pov_low <-  alz_pov_low_cut_complete[intrain_pov_low_cut, ]
alz_test_cut_pov_low <-  alz_pov_low_cut_complete[-intrain_pov_low_cut, ]

alz_train_x_cut_pov_low <-  data.matrix(alz_train_cut_pov_low[,-1])
alz_train_y_cut_pov_low <-  alz_train_cut_pov_low[,1]

alz_test_x_cut_pov_low <-  data.matrix(alz_test_cut_pov_low[,-1])
alz_test_y_cut_pov_low <-  alz_test_cut_pov_low[,1]

#create matrix for xgboost
alz_xgboost_train_cut_pov_low = xgb.DMatrix(data=alz_train_x_cut_pov_low, label=alz_train_y_cut_pov_low)
alz_xgboost_test_cut_pov_low = xgb.DMatrix(data=alz_test_x_cut_pov_low, label=alz_test_y_cut_pov_low)


watchlist_cut_pov_low <- list(train = alz_xgboost_train_cut_pov_low, test = alz_xgboost_test_cut_pov_low)




set.seed=(123)
xgb_grid_1_cut_pov_low = expand.grid(
  nrounds = c(5000),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)



xgb_trcontrol_1_cut_pov_low = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE,
  search = "grid"
)

xgb_train_1_cut_pov_low = train(
  x = alz_train_x_cut_pov_low,
  y = alz_train_y_cut_pov_low,
  trControl = xgb_trcontrol_1_cut_pov_low,
  tuneGrid = xgb_grid_1_cut_pov_low,
  method = "xgbTree"
)






xgb_grid_2_cut_pov_low <- expand.grid(
  nrounds = xgb_train_1_cut_pov_low$bestTune$nrounds,
  eta = xgb_train_1_cut_pov_low$bestTune$eta,
  max_depth = ifelse(xgb_train_1_cut_pov_low$bestTune$max_depth == 2,
                     c(xgb_train_1_cut_pov_low$bestTune$max_depth:4),
                     xgb_train_1_cut_pov_low$bestTune$max_depth - 1:xgb_train_1_cut_pov_low$bestTune$max_depth + 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3,4,5),
  subsample = 1
)

xgb_train_2_cut_pov_low = train(
  x = alz_train_x_cut_pov_low,
  y = alz_train_y_cut_pov_low,
  trControl = xgb_trcontrol_1_cut_pov_low,
  tuneGrid = xgb_grid_2_cut_pov_low,
  method = "xgbTree"
)


xgb_grid_3_cut_pov_low <- expand.grid(
  nrounds = xgb_train_2_cut_pov_low$bestTune$nrounds,
  eta = xgb_train_2_cut_pov_low$bestTune$eta,
  max_depth = xgb_train_2_cut_pov_low$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_train_2_cut_pov_low$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_train_3_cut_pov_low = train(
  x = alz_train_x_cut_pov_low,
  y = alz_train_y_cut_pov_low,
  trControl = xgb_trcontrol_1_cut_pov_low,
  tuneGrid = xgb_grid_3_cut_pov_low,
  method = "xgbTree"
)


xgb_grid_4_cut_pov_low <- expand.grid(
  nrounds = xgb_train_3_cut_pov_low$bestTune$nrounds,
  eta = xgb_train_3_cut_pov_low$bestTune$eta,
  max_depth = xgb_train_3_cut_pov_low$bestTune$max_depth,
  gamma = c(0,1,2,3,4,5,6),
  colsample_bytree = xgb_train_3_cut_pov_low$bestTune$colsample_bytree,
  min_child_weight = xgb_train_3_cut_pov_low$bestTune$min_child_weight,
  subsample = xgb_train_3_cut_pov_low$bestTune$subsample
)


xgb_train_4_cut_pov_low = train(
  x = alz_train_x_cut_pov_low,
  y = alz_train_y_cut_pov_low,
  trControl = xgb_trcontrol_1_cut_pov_low,
  tuneGrid = xgb_grid_4_cut_pov_low,
  method = "xgbTree"
)



xgb_grid_5_cut_pov_low <- expand.grid(
  nrounds = xgb_train_4_cut_pov_low$bestTune$nrounds,
  eta = c(0.001, 0.005,0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_train_4_cut_pov_low$bestTune$max_depth,
  gamma = xgb_train_4_cut_pov_low$bestTune$gamma,
  colsample_bytree = xgb_train_4_cut_pov_low$bestTune$colsample_bytree,
  min_child_weight = xgb_train_4_cut_pov_low$bestTune$min_child_weight,
  subsample = xgb_train_4_cut_pov_low$bestTune$subsample
)

xgb_train_5_cut_pov_low = train(
  x = alz_train_x_cut_pov_low,
  y = alz_train_y_cut_pov_low,
  trControl = xgb_trcontrol_1_cut_pov_low,
  tuneGrid = xgb_grid_5_cut_pov_low,
  method = "xgbTree"
)





### cut tune xgboost

xgb_params_cut_tune_pov_low = list(
  booster = "gbtree",
  eta = xgb_train_5_cut_pov_low$bestTune$eta, 
  gamma = xgb_train_5_cut_pov_low$bestTune$gamma,
  max_depth = xgb_train_5_cut_pov_low$bestTune$max_depth,
  min_child_weight =xgb_train_5_cut_pov_low$bestTune$min_child_weight,
  subsample = xgb_train_5_cut_pov_low$bestTune$subsample,
  colsample_bytree = xgb_train_5_cut_pov_low$bestTune$colsample_bytree,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
# rmse for train and test displayed
watchlist_cut_tune_pov_low <- list(train = alz_xgboost_train_cut_pov_low, test = alz_xgboost_test_cut_pov_low)

set.seed(600)
#xgboost for model
xgb_model_cut_tune_pov_low <- xgb.train(
  data = alz_xgboost_train_cut_pov_low,
  nrounds = 5000,
  watchlist = watchlist_cut_tune_pov_low,
  params = xgb_params_cut_tune_pov_low,
  early_stopping_rounds = 50
)

xgb_model_cut_tune_pov_low

#model importance
importance_cut_tune_pov_low <- xgb.importance(feature_names = colnames(alz_train_x_cut_pov_low), model = xgb_model_cut_tune_pov_low)




elog_pov_low <- xgb_model_cut_tune_pov_low$evaluation_log %>% 
  rename(train = "train_rmse",
         test = "test_rmse") %>% 
  pivot_longer(c(train, test), names_to = "Data", values_to = "RMSE")

elog_tune_cut_pov_low <- ggplot(elog_pov_low, aes(x = iter, y= RMSE, group = Data))+
  geom_line(aes(color=Data))+
  scale_color_manual(values=c("#eeb211", "#46166b", "#eeb211"))+
  scale_y_continuous(n.breaks = 10)+
  theme_classic()+
  labs(x = "Training Iterations")+
  ggtitle("Learning Curves for Training and Testing Data Stratified for Low Poverty %")+
  theme(plot.title = element_text(hjust = 0.5))


elog_tune_cut_pov_low




importance_cut_tune_pov_low_df <- as_tibble(importance_cut_tune_pov_low) %>% 
  arrange(desc(Gain)) %>% 
  filter(row_number() <= 15)

importance_plot_cut_pov_low <- ggplot(importance_cut_tune_pov_low_df, aes(x = Gain, y = reorder(Feature, Gain)))+
  geom_col(fill = "grey", color = "black")+
  theme_classic()+
  labs(x = "Importance (Gain)", y = "Feature")+
  theme(plot.title = element_text(hjust = 0.5),
        axis.text=element_text(size=12),
        axis.title=element_text(size=12))

importance_plot_cut_pov_low
library(SHAPforxgboost)

shap_pov_low <- shap.prep(xgb_model_cut_tune_pov_low, X_train = alz_train_x_cut_pov_low)

shap.plot.summary.wrap1(xgb_model_cut_tune_pov_low, alz_test_x_cut_pov_low, top_n = 20)


xgb.plot.shap(alz_train_x_cut_pov_low, model = xgb_model_cut_tune_pov_low, features = "male")




########################################


#     Analysis with social vulnerability index quantiles #


########################################

all_data_svi_quantiles <- all_data %>% 
  mutate(svi_quantile = ntile(socialvul, 4))


svi_high <- all_data_svi_quantiles %>% 
  filter(svi_quantile == 4)

svi_high_cor <- svi_high %>% 
  select( -c(stcofips, st, state, st_abbr, county, alz_spending, pop, glyphosate, dicamba, eal_score,
             resl_score, veteran, vacant, resp, ptsdf, ozone, dslpm, svi_quantile, socialvul))



alz_svi_high_cut <- svi_high_cor %>% 
  select(-c(hospital, incarcerate, syphilis, teeth, housingcost, mental, hbp, hincome, nocomputer, snap, log_homevalue))


dmy_cut_svi_high <- dummyVars(" ~ .", data = alz_svi_high_cut,fullRank = T)
alz_svi_high_cut <- data.frame(predict(dmy_cut_svi_high, newdata = alz_svi_high_cut))
alz_svi_high_cut_complete <- alz_svi_high_cut%>% 
  na.omit()


set.seed(900)
intrain_svi_high_cut <-  createDataPartition(alz_svi_high_cut_complete$alzheimers_dementia, p = 0.8, list = F)
alz_train_cut_svi_high <-  alz_svi_high_cut_complete[intrain_svi_high_cut, ]
alz_test_cut_svi_high <-  alz_svi_high_cut_complete[-intrain_svi_high_cut, ]

alz_train_x_cut_svi_high <-  data.matrix(alz_train_cut_svi_high[,-1])
alz_train_y_cut_svi_high <-  alz_train_cut_svi_high[,1]

alz_test_x_cut_svi_high <-  data.matrix(alz_test_cut_svi_high[,-1])
alz_test_y_cut_svi_high <-  alz_test_cut_svi_high[,1]

#create matrix for xgboost
alz_xgboost_train_cut_svi_high = xgb.DMatrix(data=alz_train_x_cut_svi_high, label=alz_train_y_cut_svi_high)
alz_xgboost_test_cut_svi_high = xgb.DMatrix(data=alz_test_x_cut_svi_high, label=alz_test_y_cut_svi_high)



set.seed=(1234)
xgb_grid_1_cut_svi_high = expand.grid(
  nrounds = c(5000),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)



xgb_trcontrol_1_cut_svi_high = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE,
  search = "grid"
)

xgb_train_1_cut_svi_high = train(
  x = alz_train_x_cut_svi_high,
  y = alz_train_y_cut_svi_high,
  trControl = xgb_trcontrol_1_cut_svi_high,
  tuneGrid = xgb_grid_1_cut_svi_high,
  method = "xgbTree"
)






xgb_grid_2_cut_svi_high <- expand.grid(
  nrounds = xgb_train_1_cut_svi_high$bestTune$nrounds,
  eta = xgb_train_1_cut_svi_high$bestTune$eta,
  max_depth = ifelse(xgb_train_1_cut_svi_high$bestTune$max_depth == 2,
                     c(xgb_train_1_cut_svi_high$bestTune$max_depth:4),
                     xgb_train_1_cut_svi_high$bestTune$max_depth - 1:xgb_train_1_cut_svi_high$bestTune$max_depth + 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3,4,5),
  subsample = 1
)

xgb_train_2_cut_svi_high = train(
  x = alz_train_x_cut_svi_high,
  y = alz_train_y_cut_svi_high,
  trControl = xgb_trcontrol_1_cut_svi_high,
  tuneGrid = xgb_grid_2_cut_svi_high,
  method = "xgbTree"
)


xgb_grid_3_cut_svi_high <- expand.grid(
  nrounds = xgb_train_2_cut_svi_high$bestTune$nrounds,
  eta = xgb_train_2_cut_svi_high$bestTune$eta,
  max_depth = xgb_train_2_cut_svi_high$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_train_2_cut_svi_high$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_train_3_cut_svi_high = train(
  x = alz_train_x_cut_svi_high,
  y = alz_train_y_cut_svi_high,
  trControl = xgb_trcontrol_1_cut_svi_high,
  tuneGrid = xgb_grid_3_cut_svi_high,
  method = "xgbTree"
)


xgb_grid_4_cut_svi_high <- expand.grid(
  nrounds = xgb_train_3_cut_svi_high$bestTune$nrounds,
  eta = xgb_train_3_cut_svi_high$bestTune$eta,
  max_depth = xgb_train_3_cut_svi_high$bestTune$max_depth,
  gamma = c(0,1,2,3,4,5,6),
  colsample_bytree = xgb_train_3_cut_svi_high$bestTune$colsample_bytree,
  min_child_weight = xgb_train_3_cut_svi_high$bestTune$min_child_weight,
  subsample = xgb_train_3_cut_svi_high$bestTune$subsample
)


xgb_train_4_cut_svi_high = train(
  x = alz_train_x_cut_svi_high,
  y = alz_train_y_cut_svi_high,
  trControl = xgb_trcontrol_1_cut_svi_high,
  tuneGrid = xgb_grid_4_cut_svi_high,
  method = "xgbTree"
)



xgb_grid_5_cut_svi_high <- expand.grid(
  nrounds = xgb_train_4_cut_svi_high$bestTune$nrounds,
  eta = c(0.001, 0.005,0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_train_4_cut_svi_high$bestTune$max_depth,
  gamma = xgb_train_4_cut_svi_high$bestTune$gamma,
  colsample_bytree = xgb_train_4_cut_svi_high$bestTune$colsample_bytree,
  min_child_weight = xgb_train_4_cut_svi_high$bestTune$min_child_weight,
  subsample = xgb_train_4_cut_svi_high$bestTune$subsample
)

xgb_train_5_cut_svi_high = train(
  x = alz_train_x_cut_svi_high,
  y = alz_train_y_cut_svi_high,
  trControl = xgb_trcontrol_1_cut_svi_high,
  tuneGrid = xgb_grid_5_cut_svi_high,
  method = "xgbTree"
)





### cut tune xgboost

xgb_params_cut_tune_svi_high = list(
  booster = "gbtree",
  eta = xgb_train_5_cut_svi_high$bestTune$eta, 
  gamma = xgb_train_5_cut_svi_high$bestTune$gamma,
  max_depth = xgb_train_5_cut_svi_high$bestTune$max_depth,
  min_child_weight =xgb_train_5_cut_svi_high$bestTune$min_child_weight,
  subsample = xgb_train_5_cut_svi_high$bestTune$subsample,
  colsample_bytree = xgb_train_5_cut_svi_high$bestTune$colsample_bytree,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
# rmse for train and test displayed
watchlist_cut_tune_svi_high <- list(train = alz_xgboost_train_cut_svi_high, test = alz_xgboost_test_cut_svi_high)

set.seed(8009)
#xgboost for model
xgb_model_cut_tune_svi_high <- xgb.train(
  data = alz_xgboost_train_cut_svi_high,
  nrounds = 5000,
  watchlist = watchlist_cut_tune_svi_high,
  params = xgb_params_cut_tune_svi_high,
  early_stopping_rounds = 50
)

xgb_model_cut_tune_svi_high

#model importance
importance_cut_tune_svi_high <- xgb.importance(feature_names = colnames(alz_train_x_cut_svi_high), model = xgb_model_cut_tune_svi_high)


elog_svi_high <- xgb_model_cut_tune_svi_high$evaluation_log %>% 
  rename(train = "train_rmse",
         test = "test_rmse") %>% 
  pivot_longer(c(train, test), names_to = "Data", values_to = "RMSE")

elog_tune_cut_svi_high <- ggplot(elog_svi_high, aes(x = iter, y= RMSE, group = Data))+
  geom_line(aes(color=Data))+
  scale_color_manual(values=c("#eeb211", "#46166b", "#eeb211"))+
  scale_y_continuous(n.breaks = 10)+
  theme_classic()+
  labs(x = "Training Iterations")+
  ggtitle("Learning Curves for Training and Testing Data Stratified for High Poverty %")+
  theme(plot.title = element_text(hjust = 0.5))


elog_tune_cut_svi_high




importance_cut_tune_svi_high_df <- as_tibble(importance_cut_tune_svi_high) %>% 
  arrange(desc(Gain)) %>% 
  filter(row_number() <= 15)

importance_plot_cut_svi_high <- ggplot(importance_cut_tune_svi_high_df, aes(x = Gain, y = reorder(Feature, Gain)))+
  geom_col(fill = "grey", color = "black")+
  theme_classic()+
  labs(x = "Importance (Gain)", y = "Feature")+
  theme(plot.title = element_text(hjust = 0.5),
        axis.text=element_text(size=12),
        axis.title=element_text(size=12))

importance_plot_cut_svi_high


shap_svi_high <- shap.prep(xgb_model_cut_tune_svi_high, X_train = alz_train_x_cut_svi_high)

shap.plot.summary.wrap1(xgb_model_cut_tune_svi_high, alz_test_x_cut_svi_high, top_n = 20)

####
######         svi low
######
######

svi_low <- all_data_svi_quantiles %>% 
  filter(svi_quantile == 1)

svi_low_cor <- svi_low %>% 
  select( -c(stcofips, st, state, st_abbr, county, alz_spending, pop, glyphosate, dicamba, eal_score,
             resl_score, veteran, vacant, resp, ptsdf, ozone, dslpm, svi_quantile, socialvul))



alz_svi_low_cut <- svi_low_cor %>% 
  select(-c(hospital, incarcerate, syphilis, teeth, housingcost, mental, hbp, hincome, nocomputer, snap, log_homevalue))


dmy_cut_svi_low <- dummyVars(" ~ .", data = alz_svi_low_cut,fullRank = T)
alz_svi_low_cut <- data.frame(predict(dmy_cut_svi_low, newdata = alz_svi_low_cut))
alz_svi_low_cut_complete <- alz_svi_low_cut%>% 
  na.omit()


set.seed(900)
intrain_svi_low_cut <-  createDataPartition(alz_svi_low_cut_complete$alzheimers_dementia, p = 0.8, list = F)
alz_train_cut_svi_low <-  alz_svi_low_cut_complete[intrain_svi_low_cut, ]
alz_test_cut_svi_low <-  alz_svi_low_cut_complete[-intrain_svi_low_cut, ]

alz_train_x_cut_svi_low <-  data.matrix(alz_train_cut_svi_low[,-1])
alz_train_y_cut_svi_low <-  alz_train_cut_svi_low[,1]

alz_test_x_cut_svi_low <-  data.matrix(alz_test_cut_svi_low[,-1])
alz_test_y_cut_svi_low <-  alz_test_cut_svi_low[,1]

#create matrix for xgboost
alz_xgboost_train_cut_svi_low = xgb.DMatrix(data=alz_train_x_cut_svi_low, label=alz_train_y_cut_svi_low)
alz_xgboost_test_cut_svi_low = xgb.DMatrix(data=alz_test_x_cut_svi_low, label=alz_test_y_cut_svi_low)



set.seed=(1234)
xgb_grid_1_cut_svi_low = expand.grid(
  nrounds = c(5000),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)



xgb_trcontrol_1_cut_svi_low = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE,
  search = "grid"
)

xgb_train_1_cut_svi_low = train(
  x = alz_train_x_cut_svi_low,
  y = alz_train_y_cut_svi_low,
  trControl = xgb_trcontrol_1_cut_svi_low,
  tuneGrid = xgb_grid_1_cut_svi_low,
  method = "xgbTree"
)






xgb_grid_2_cut_svi_low <- expand.grid(
  nrounds = xgb_train_1_cut_svi_low$bestTune$nrounds,
  eta = xgb_train_1_cut_svi_low$bestTune$eta,
  max_depth = ifelse(xgb_train_1_cut_svi_low$bestTune$max_depth == 2,
                     c(xgb_train_1_cut_svi_low$bestTune$max_depth:4),
                     xgb_train_1_cut_svi_low$bestTune$max_depth - 1:xgb_train_1_cut_svi_low$bestTune$max_depth + 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3,4,5),
  subsample = 1
)

xgb_train_2_cut_svi_low = train(
  x = alz_train_x_cut_svi_low,
  y = alz_train_y_cut_svi_low,
  trControl = xgb_trcontrol_1_cut_svi_low,
  tuneGrid = xgb_grid_2_cut_svi_low,
  method = "xgbTree"
)


xgb_grid_3_cut_svi_low <- expand.grid(
  nrounds = xgb_train_2_cut_svi_low$bestTune$nrounds,
  eta = xgb_train_2_cut_svi_low$bestTune$eta,
  max_depth = xgb_train_2_cut_svi_low$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_train_2_cut_svi_low$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_train_3_cut_svi_low = train(
  x = alz_train_x_cut_svi_low,
  y = alz_train_y_cut_svi_low,
  trControl = xgb_trcontrol_1_cut_svi_low,
  tuneGrid = xgb_grid_3_cut_svi_low,
  method = "xgbTree"
)


xgb_grid_4_cut_svi_low <- expand.grid(
  nrounds = xgb_train_3_cut_svi_low$bestTune$nrounds,
  eta = xgb_train_3_cut_svi_low$bestTune$eta,
  max_depth = xgb_train_3_cut_svi_low$bestTune$max_depth,
  gamma = c(0,1,2,3,4,5,6),
  colsample_bytree = xgb_train_3_cut_svi_low$bestTune$colsample_bytree,
  min_child_weight = xgb_train_3_cut_svi_low$bestTune$min_child_weight,
  subsample = xgb_train_3_cut_svi_low$bestTune$subsample
)


xgb_train_4_cut_svi_low = train(
  x = alz_train_x_cut_svi_low,
  y = alz_train_y_cut_svi_low,
  trControl = xgb_trcontrol_1_cut_svi_low,
  tuneGrid = xgb_grid_4_cut_svi_low,
  method = "xgbTree"
)



xgb_grid_5_cut_svi_low <- expand.grid(
  nrounds = xgb_train_4_cut_svi_low$bestTune$nrounds,
  eta = c(0.001, 0.005,0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_train_4_cut_svi_low$bestTune$max_depth,
  gamma = xgb_train_4_cut_svi_low$bestTune$gamma,
  colsample_bytree = xgb_train_4_cut_svi_low$bestTune$colsample_bytree,
  min_child_weight = xgb_train_4_cut_svi_low$bestTune$min_child_weight,
  subsample = xgb_train_4_cut_svi_low$bestTune$subsample
)

xgb_train_5_cut_svi_low = train(
  x = alz_train_x_cut_svi_low,
  y = alz_train_y_cut_svi_low,
  trControl = xgb_trcontrol_1_cut_svi_low,
  tuneGrid = xgb_grid_5_cut_svi_low,
  method = "xgbTree"
)





### cut tune xgboost

xgb_params_cut_tune_svi_low = list(
  booster = "gbtree",
  eta = xgb_train_5_cut_svi_low$bestTune$eta, 
  gamma = xgb_train_5_cut_svi_low$bestTune$gamma,
  max_depth = xgb_train_5_cut_svi_low$bestTune$max_depth,
  min_child_weight =xgb_train_5_cut_svi_low$bestTune$min_child_weight,
  subsample = xgb_train_5_cut_svi_low$bestTune$subsample,
  colsample_bytree = xgb_train_5_cut_svi_low$bestTune$colsample_bytree,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
# rmse for train and test displayed
watchlist_cut_tune_svi_low <- list(train = alz_xgboost_train_cut_svi_low, test = alz_xgboost_test_cut_svi_low)

set.seed(8009)
#xgboost for model
xgb_model_cut_tune_svi_low <- xgb.train(
  data = alz_xgboost_train_cut_svi_low,
  nrounds = 5000,
  watchlist = watchlist_cut_tune_svi_low,
  params = xgb_params_cut_tune_svi_low,
  early_stopping_rounds = 50
)

xgb_model_cut_tune_svi_low

#model importance
importance_cut_tune_svi_low <- xgb.importance(feature_names = colnames(alz_train_x_cut_svi_low), model = xgb_model_cut_tune_svi_low)

elog_svi_low <- xgb_model_cut_tune_svi_low$evaluation_log %>% 
  rename(train = "train_rmse",
         test = "test_rmse") %>% 
  pivot_longer(c(train, test), names_to = "Data", values_to = "RMSE")

elog_tune_cut_svi_low <- ggplot(elog_svi_low, aes(x = iter, y= RMSE, group = Data))+
  geom_line(aes(color=Data))+
  scale_color_manual(values=c("#eeb211", "#46166b", "#eeb211"))+
  scale_y_continuous(n.breaks = 10)+
  theme_classic()+
  labs(x = "Training Iterations")+
  ggtitle("Learning Curves for Training and Testing Data Stratified for Low Poverty %")+
  theme(plot.title = element_text(hjust = 0.5))


elog_tune_cut_svi_low




importance_cut_tune_svi_low_df <- as_tibble(importance_cut_tune_svi_low) %>% 
  arrange(desc(Gain)) %>% 
  filter(row_number() <= 15)

importance_plot_cut_svi_low <- ggplot(importance_cut_tune_svi_low_df, aes(x = Gain, y = reorder(Feature, Gain)))+
  geom_col(fill = "grey", color = "black")+
  theme_classic()+
  labs(x = "Importance (Gain)", y = "Feature")+
  theme(plot.title = element_text(hjust = 0.5),
        axis.text=element_text(size=12),
        axis.title=element_text(size=12))

importance_plot_cut_svi_low
library(SHAPforxgboost)

shap_svi_low <- shap.prep(xgb_model_cut_tune_svi_low, X_train = alz_train_x_cut_svi_low)

shap.plot.summary.wrap1(xgb_model_cut_tune_svi_low, alz_test_x_cut_svi_low, top_n = 20)


xgb.plot.shap(alz_train_x_cut_svi_low, model = xgb_model_cut_tune_svi_low, features = "male")

########################################


#     Analysis with racial segregation quantiles #


########################################


all_data_rs_quantiles <- all_data %>% 
  mutate(rs_quantile = ntile(racialseg, 4))

summary(all_data_rs_quantiles)
rs_high <- all_data_rs_quantiles %>% 
  filter(rs_quantile == 4)

rs_high_cor <- rs_high %>% 
  select( -c(stcofips, st, state, st_abbr, county, alz_spending, pop, glyphosate, dicamba, eal_score,
             resl_score, veteran, vacant, resp, ptsdf, ozone, dslpm, rs_quantile, racialseg))



alz_rs_high_cut <- rs_high_cor %>% 
  select(-c(hospital, incarcerate, syphilis, teeth, housingcost, mental, hbp, hincome, nocomputer, snap, log_homevalue))


dmy_cut_rs_high <- dummyVars(" ~ .", data = alz_rs_high_cut,fullRank = T)
alz_rs_high_cut <- data.frame(predict(dmy_cut_rs_high, newdata = alz_rs_high_cut))
alz_rs_high_cut_complete <- alz_rs_high_cut%>% 
  na.omit()


set.seed(900)
intrain_rs_high_cut <-  createDataPartition(alz_rs_high_cut_complete$alzheimers_dementia, p = 0.8, list = F)
alz_train_cut_rs_high <-  alz_rs_high_cut_complete[intrain_rs_high_cut, ]
alz_test_cut_rs_high <-  alz_rs_high_cut_complete[-intrain_rs_high_cut, ]

alz_train_x_cut_rs_high <-  data.matrix(alz_train_cut_rs_high[,-1])
alz_train_y_cut_rs_high <-  alz_train_cut_rs_high[,1]

alz_test_x_cut_rs_high <-  data.matrix(alz_test_cut_rs_high[,-1])
alz_test_y_cut_rs_high <-  alz_test_cut_rs_high[,1]

#create matrix for xgboost
alz_xgboost_train_cut_rs_high = xgb.DMatrix(data=alz_train_x_cut_rs_high, label=alz_train_y_cut_rs_high)
alz_xgboost_test_cut_rs_high = xgb.DMatrix(data=alz_test_x_cut_rs_high, label=alz_test_y_cut_rs_high)



set.seed=(1234)
xgb_grid_1_cut_rs_high = expand.grid(
  nrounds = c(5000),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)



xgb_trcontrol_1_cut_rs_high = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE,
  search = "grid"
)

xgb_train_1_cut_rs_high = train(
  x = alz_train_x_cut_rs_high,
  y = alz_train_y_cut_rs_high,
  trControl = xgb_trcontrol_1_cut_rs_high,
  tuneGrid = xgb_grid_1_cut_rs_high,
  method = "xgbTree"
)






xgb_grid_2_cut_rs_high <- expand.grid(
  nrounds = xgb_train_1_cut_rs_high$bestTune$nrounds,
  eta = xgb_train_1_cut_rs_high$bestTune$eta,
  max_depth = ifelse(xgb_train_1_cut_rs_high$bestTune$max_depth == 2,
                     c(xgb_train_1_cut_rs_high$bestTune$max_depth:4),
                     xgb_train_1_cut_rs_high$bestTune$max_depth - 1:xgb_train_1_cut_rs_high$bestTune$max_depth + 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3,4,5),
  subsample = 1
)

xgb_train_2_cut_rs_high = train(
  x = alz_train_x_cut_rs_high,
  y = alz_train_y_cut_rs_high,
  trControl = xgb_trcontrol_1_cut_rs_high,
  tuneGrid = xgb_grid_2_cut_rs_high,
  method = "xgbTree"
)


xgb_grid_3_cut_rs_high <- expand.grid(
  nrounds = xgb_train_2_cut_rs_high$bestTune$nrounds,
  eta = xgb_train_2_cut_rs_high$bestTune$eta,
  max_depth = xgb_train_2_cut_rs_high$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_train_2_cut_rs_high$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_train_3_cut_rs_high = train(
  x = alz_train_x_cut_rs_high,
  y = alz_train_y_cut_rs_high,
  trControl = xgb_trcontrol_1_cut_rs_high,
  tuneGrid = xgb_grid_3_cut_rs_high,
  method = "xgbTree"
)


xgb_grid_4_cut_rs_high <- expand.grid(
  nrounds = xgb_train_3_cut_rs_high$bestTune$nrounds,
  eta = xgb_train_3_cut_rs_high$bestTune$eta,
  max_depth = xgb_train_3_cut_rs_high$bestTune$max_depth,
  gamma = c(0,1,2,3,4,5,6),
  colsample_bytree = xgb_train_3_cut_rs_high$bestTune$colsample_bytree,
  min_child_weight = xgb_train_3_cut_rs_high$bestTune$min_child_weight,
  subsample = xgb_train_3_cut_rs_high$bestTune$subsample
)


xgb_train_4_cut_rs_high = train(
  x = alz_train_x_cut_rs_high,
  y = alz_train_y_cut_rs_high,
  trControl = xgb_trcontrol_1_cut_rs_high,
  tuneGrid = xgb_grid_4_cut_rs_high,
  method = "xgbTree"
)



xgb_grid_5_cut_rs_high <- expand.grid(
  nrounds = xgb_train_4_cut_rs_high$bestTune$nrounds,
  eta = c(0.001, 0.005,0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_train_4_cut_rs_high$bestTune$max_depth,
  gamma = xgb_train_4_cut_rs_high$bestTune$gamma,
  colsample_bytree = xgb_train_4_cut_rs_high$bestTune$colsample_bytree,
  min_child_weight = xgb_train_4_cut_rs_high$bestTune$min_child_weight,
  subsample = xgb_train_4_cut_rs_high$bestTune$subsample
)

xgb_train_5_cut_rs_high = train(
  x = alz_train_x_cut_rs_high,
  y = alz_train_y_cut_rs_high,
  trControl = xgb_trcontrol_1_cut_rs_high,
  tuneGrid = xgb_grid_5_cut_rs_high,
  method = "xgbTree"
)





### cut tune xgboost

xgb_params_cut_tune_rs_high = list(
  booster = "gbtree",
  eta = xgb_train_5_cut_rs_high$bestTune$eta, 
  gamma = xgb_train_5_cut_rs_high$bestTune$gamma,
  max_depth = xgb_train_5_cut_rs_high$bestTune$max_depth,
  min_child_weight =xgb_train_5_cut_rs_high$bestTune$min_child_weight,
  subsample = xgb_train_5_cut_rs_high$bestTune$subsample,
  colsample_bytree = xgb_train_5_cut_rs_high$bestTune$colsample_bytree,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
# rmse for train and test displayed
watchlist_cut_tune_rs_high <- list(train = alz_xgboost_train_cut_rs_high, test = alz_xgboost_test_cut_rs_high)

set.seed(8009)
#xgboost for model
xgb_model_cut_tune_rs_high <- xgb.train(
  data = alz_xgboost_train_cut_rs_high,
  nrounds = 5000,
  watchlist = watchlist_cut_tune_rs_high,
  params = xgb_params_cut_tune_rs_high,
  early_stopping_rounds = 50
)

xgb_model_cut_tune_rs_high

#model importance
importance_cut_tune_rs_high <- xgb.importance(feature_names = colnames(alz_train_x_cut_rs_high), model = xgb_model_cut_tune_rs_high)


elog_rs_high <- xgb_model_cut_tune_rs_high$evaluation_log %>% 
  rename(train = "train_rmse",
         test = "test_rmse") %>% 
  pivot_longer(c(train, test), names_to = "Data", values_to = "RMSE")

elog_tune_cut_rs_high <- ggplot(elog_rs_high, aes(x = iter, y= RMSE, group = Data))+
  geom_line(aes(color=Data))+
  scale_color_manual(values=c("#eeb211", "#46166b", "#eeb211"))+
  scale_y_continuous(n.breaks = 10)+
  theme_classic()+
  labs(x = "Training Iterations")+
  ggtitle("Learning Curves for Training and Testing Data Stratified for High Poverty %")+
  theme(plot.title = element_text(hjust = 0.5))


elog_tune_cut_rs_high




importance_cut_tune_rs_high_df <- as_tibble(importance_cut_tune_rs_high) %>% 
  arrange(desc(Gain)) %>% 
  filter(row_number() <= 15)

importance_plot_cut_rs_high <- ggplot(importance_cut_tune_rs_high_df, aes(x = Gain, y = reorder(Feature, Gain)))+
  geom_col(fill = "grey", color = "black")+
  theme_classic()+
  labs(x = "Importance (Gain)", y = "Feature")+
  theme(plot.title = element_text(hjust = 0.5),
        axis.text=element_text(size=12),
        axis.title=element_text(size=12))

importance_plot_cut_rs_high


shap_rs_high <- shap.prep(xgb_model_cut_tune_rs_high, X_train = alz_train_x_cut_rs_high)

shap.plot.summary.wrap1(xgb_model_cut_tune_rs_high, alz_test_x_cut_rs_high, top_n = 20)

###############3
################   low racial seg
##############


rs_low <- all_data_rs_quantiles %>% 
  filter(rs_quantile == 1)

rs_low_cor <- rs_low %>% 
  select( -c(stcofips, st, state, st_abbr, county, alz_spending, pop, glyphosate, dicamba, eal_score,
             resl_score, veteran, vacant, resp, ptsdf, ozone, dslpm, rs_quantile, racialseg))



alz_rs_low_cut <- rs_low_cor %>% 
  select(-c(hospital, incarcerate, syphilis, teeth, housingcost, mental, hbp, hincome, nocomputer, snap, log_homevalue))


dmy_cut_rs_low <- dummyVars(" ~ .", data = alz_rs_low_cut,fullRank = T)
alz_rs_low_cut <- data.frame(predict(dmy_cut_rs_low, newdata = alz_rs_low_cut))
alz_rs_low_cut_complete <- alz_rs_low_cut%>% 
  na.omit()


set.seed(900)
intrain_rs_low_cut <-  createDataPartition(alz_rs_low_cut_complete$alzheimers_dementia, p = 0.8, list = F)
alz_train_cut_rs_low <-  alz_rs_low_cut_complete[intrain_rs_low_cut, ]
alz_test_cut_rs_low <-  alz_rs_low_cut_complete[-intrain_rs_low_cut, ]

alz_train_x_cut_rs_low <-  data.matrix(alz_train_cut_rs_low[,-1])
alz_train_y_cut_rs_low <-  alz_train_cut_rs_low[,1]

alz_test_x_cut_rs_low <-  data.matrix(alz_test_cut_rs_low[,-1])
alz_test_y_cut_rs_low <-  alz_test_cut_rs_low[,1]

#create matrix for xgboost
alz_xgboost_train_cut_rs_low = xgb.DMatrix(data=alz_train_x_cut_rs_low, label=alz_train_y_cut_rs_low)
alz_xgboost_test_cut_rs_low = xgb.DMatrix(data=alz_test_x_cut_rs_low, label=alz_test_y_cut_rs_low)



set.seed=(1234)
xgb_grid_1_cut_rs_low = expand.grid(
  nrounds = c(5000),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)



xgb_trcontrol_1_cut_rs_low = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE,
  search = "grid"
)

xgb_train_1_cut_rs_low = train(
  x = alz_train_x_cut_rs_low,
  y = alz_train_y_cut_rs_low,
  trControl = xgb_trcontrol_1_cut_rs_low,
  tuneGrid = xgb_grid_1_cut_rs_low,
  method = "xgbTree"
)






xgb_grid_2_cut_rs_low <- expand.grid(
  nrounds = xgb_train_1_cut_rs_low$bestTune$nrounds,
  eta = xgb_train_1_cut_rs_low$bestTune$eta,
  max_depth = ifelse(xgb_train_1_cut_rs_low$bestTune$max_depth == 2,
                     c(xgb_train_1_cut_rs_low$bestTune$max_depth:4),
                     xgb_train_1_cut_rs_low$bestTune$max_depth - 1:xgb_train_1_cut_rs_low$bestTune$max_depth + 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3,4,5),
  subsample = 1
)

xgb_train_2_cut_rs_low = train(
  x = alz_train_x_cut_rs_low,
  y = alz_train_y_cut_rs_low,
  trControl = xgb_trcontrol_1_cut_rs_low,
  tuneGrid = xgb_grid_2_cut_rs_low,
  method = "xgbTree"
)


xgb_grid_3_cut_rs_low <- expand.grid(
  nrounds = xgb_train_2_cut_rs_low$bestTune$nrounds,
  eta = xgb_train_2_cut_rs_low$bestTune$eta,
  max_depth = xgb_train_2_cut_rs_low$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_train_2_cut_rs_low$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_train_3_cut_rs_low = train(
  x = alz_train_x_cut_rs_low,
  y = alz_train_y_cut_rs_low,
  trControl = xgb_trcontrol_1_cut_rs_low,
  tuneGrid = xgb_grid_3_cut_rs_low,
  method = "xgbTree"
)


xgb_grid_4_cut_rs_low <- expand.grid(
  nrounds = xgb_train_3_cut_rs_low$bestTune$nrounds,
  eta = xgb_train_3_cut_rs_low$bestTune$eta,
  max_depth = xgb_train_3_cut_rs_low$bestTune$max_depth,
  gamma = c(0,1,2,3,4,5,6),
  colsample_bytree = xgb_train_3_cut_rs_low$bestTune$colsample_bytree,
  min_child_weight = xgb_train_3_cut_rs_low$bestTune$min_child_weight,
  subsample = xgb_train_3_cut_rs_low$bestTune$subsample
)


xgb_train_4_cut_rs_low = train(
  x = alz_train_x_cut_rs_low,
  y = alz_train_y_cut_rs_low,
  trControl = xgb_trcontrol_1_cut_rs_low,
  tuneGrid = xgb_grid_4_cut_rs_low,
  method = "xgbTree"
)



xgb_grid_5_cut_rs_low <- expand.grid(
  nrounds = xgb_train_4_cut_rs_low$bestTune$nrounds,
  eta = c(0.001, 0.005,0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_train_4_cut_rs_low$bestTune$max_depth,
  gamma = xgb_train_4_cut_rs_low$bestTune$gamma,
  colsample_bytree = xgb_train_4_cut_rs_low$bestTune$colsample_bytree,
  min_child_weight = xgb_train_4_cut_rs_low$bestTune$min_child_weight,
  subsample = xgb_train_4_cut_rs_low$bestTune$subsample
)

xgb_train_5_cut_rs_low = train(
  x = alz_train_x_cut_rs_low,
  y = alz_train_y_cut_rs_low,
  trControl = xgb_trcontrol_1_cut_rs_low,
  tuneGrid = xgb_grid_5_cut_rs_low,
  method = "xgbTree"
)





### cut tune xgboost

xgb_params_cut_tune_rs_low = list(
  booster = "gbtree",
  eta = xgb_train_5_cut_rs_low$bestTune$eta, 
  gamma = xgb_train_5_cut_rs_low$bestTune$gamma,
  max_depth = xgb_train_5_cut_rs_low$bestTune$max_depth,
  min_child_weight =xgb_train_5_cut_rs_low$bestTune$min_child_weight,
  subsample = xgb_train_5_cut_rs_low$bestTune$subsample,
  colsample_bytree = xgb_train_5_cut_rs_low$bestTune$colsample_bytree,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
# rmse for train and test displayed
watchlist_cut_tune_rs_low <- list(train = alz_xgboost_train_cut_rs_low, test = alz_xgboost_test_cut_rs_low)

set.seed(8009)
#xgboost for model
xgb_model_cut_tune_rs_low <- xgb.train(
  data = alz_xgboost_train_cut_rs_low,
  nrounds = 5000,
  watchlist = watchlist_cut_tune_rs_low,
  params = xgb_params_cut_tune_rs_low,
  early_stopping_rounds = 50
)

xgb_model_cut_tune_rs_low

#model importance
importance_cut_tune_rs_low <- xgb.importance(feature_names = colnames(alz_train_x_cut_rs_low), model = xgb_model_cut_tune_rs_low)


elog_rs_low <- xgb_model_cut_tune_rs_low$evaluation_log %>% 
  rename(train = "train_rmse",
         test = "test_rmse") %>% 
  pivot_longer(c(train, test), names_to = "Data", values_to = "RMSE")

elog_tune_cut_rs_low <- ggplot(elog_rs_low, aes(x = iter, y= RMSE, group = Data))+
  geom_line(aes(color=Data))+
  scale_color_manual(values=c("#eeb211", "#46166b", "#eeb211"))+
  scale_y_continuous(n.breaks = 10)+
  theme_classic()+
  labs(x = "Training Iterations")+
  ggtitle("Learning Curves for Training and Testing Data Stratified for Low Poverty %")+
  theme(plot.title = element_text(hjust = 0.5))


elog_tune_cut_rs_low




importance_cut_tune_rs_low_df <- as_tibble(importance_cut_tune_rs_low) %>% 
  arrange(desc(Gain)) %>% 
  filter(row_number() <= 15)

importance_plot_cut_rs_low <- ggplot(importance_cut_tune_rs_low_df, aes(x = Gain, y = reorder(Feature, Gain)))+
  geom_col(fill = "grey", color = "black")+
  theme_classic()+
  labs(x = "Importance (Gain)", y = "Feature")+
  theme(plot.title = element_text(hjust = 0.5),
        axis.text=element_text(size=12),
        axis.title=element_text(size=12))

importance_plot_cut_rs_low
library(SHAPforxgboost)

shap_rs_low <- shap.prep(xgb_model_cut_tune_rs_low, X_train = alz_train_x_cut_rs_low)

shap.plot.summary.wrap1(xgb_model_cut_tune_rs_low, alz_test_x_cut_rs_low, top_n = 20)


xgb.plot.shap(alz_train_x_cut_rs_low, model = xgb_model_cut_tune_rs_low, features = "male")

########################################


#     Analysis with urban/rural quantiles #


########################################
summary(all_data$metro)

all_data_urbanrural <- all_data %>% 
  mutate(urban_rural = case_when(
    metro %in% c(1, 2, 3) ~ "urban",
    metro == 4 ~ "rural"
  )) 


urban <- all_data_urbanrural %>% 
  filter(urban_rural == "urban")

urban_cor <- urban %>% 
  select( -c(stcofips, st, state, st_abbr, county, alz_spending, pop, glyphosate, dicamba, eal_score,
             resl_score, veteran, vacant, resp, ptsdf, ozone, dslpm, metro, urban_rural))



alz_urban <- urban_cor %>% 
  select(-c(hospital, incarcerate, syphilis, teeth, housingcost, mental, hbp, hincome, nocomputer, snap, log_homevalue))


dmy_urban <- dummyVars(" ~ .", data = alz_urban,fullRank = T)
alz_urban <- data.frame(predict(dmy_urban, newdata = alz_urban))
alz_urban_complete <- alz_urban%>% 
  na.omit()


set.seed(900)
intrain_urban_cut <-  createDataPartition(alz_urban_complete$alzheimers_dementia, p = 0.8, list = F)
alz_train_cut_urban <-  alz_urban_complete[intrain_urban_cut, ]
alz_test_cut_urban <-  alz_urban_complete[-intrain_urban_cut, ]

alz_train_x_cut_urban <-  data.matrix(alz_train_cut_urban[,-1])
alz_train_y_cut_urban <-  alz_train_cut_urban[,1]

alz_test_x_cut_urban <-  data.matrix(alz_test_cut_urban[,-1])
alz_test_y_cut_urban <-  alz_test_cut_urban[,1]

#create matrix for xgboost
alz_xgboost_train_cut_urban = xgb.DMatrix(data=alz_train_x_cut_urban, label=alz_train_y_cut_urban)
alz_xgboost_test_cut_urban = xgb.DMatrix(data=alz_test_x_cut_urban, label=alz_test_y_cut_urban)



set.seed=(1234)
xgb_grid_1_cut_urban = expand.grid(
  nrounds = c(5000),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)



xgb_trcontrol_1_cut_urban = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE,
  search = "grid"
)

xgb_train_1_cut_urban = train(
  x = alz_train_x_cut_urban,
  y = alz_train_y_cut_urban,
  trControl = xgb_trcontrol_1_cut_urban,
  tuneGrid = xgb_grid_1_cut_urban,
  method = "xgbTree"
)






xgb_grid_2_cut_urban <- expand.grid(
  nrounds = xgb_train_1_cut_urban$bestTune$nrounds,
  eta = xgb_train_1_cut_urban$bestTune$eta,
  max_depth = ifelse(xgb_train_1_cut_urban$bestTune$max_depth == 2,
                     c(xgb_train_1_cut_urban$bestTune$max_depth:4),
                     xgb_train_1_cut_urban$bestTune$max_depth - 1:xgb_train_1_cut_urban$bestTune$max_depth + 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3,4,5),
  subsample = 1
)

xgb_train_2_cut_urban = train(
  x = alz_train_x_cut_urban,
  y = alz_train_y_cut_urban,
  trControl = xgb_trcontrol_1_cut_urban,
  tuneGrid = xgb_grid_2_cut_urban,
  method = "xgbTree"
)


xgb_grid_3_cut_urban <- expand.grid(
  nrounds = xgb_train_2_cut_urban$bestTune$nrounds,
  eta = xgb_train_2_cut_urban$bestTune$eta,
  max_depth = xgb_train_2_cut_urban$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_train_2_cut_urban$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_train_3_cut_urban = train(
  x = alz_train_x_cut_urban,
  y = alz_train_y_cut_urban,
  trControl = xgb_trcontrol_1_cut_urban,
  tuneGrid = xgb_grid_3_cut_urban,
  method = "xgbTree"
)


xgb_grid_4_cut_urban <- expand.grid(
  nrounds = xgb_train_3_cut_urban$bestTune$nrounds,
  eta = xgb_train_3_cut_urban$bestTune$eta,
  max_depth = xgb_train_3_cut_urban$bestTune$max_depth,
  gamma = c(0,1,2,3,4,5,6),
  colsample_bytree = xgb_train_3_cut_urban$bestTune$colsample_bytree,
  min_child_weight = xgb_train_3_cut_urban$bestTune$min_child_weight,
  subsample = xgb_train_3_cut_urban$bestTune$subsample
)


xgb_train_4_cut_urban = train(
  x = alz_train_x_cut_urban,
  y = alz_train_y_cut_urban,
  trControl = xgb_trcontrol_1_cut_urban,
  tuneGrid = xgb_grid_4_cut_urban,
  method = "xgbTree"
)



xgb_grid_5_cut_urban <- expand.grid(
  nrounds = xgb_train_4_cut_urban$bestTune$nrounds,
  eta = c(0.001, 0.005,0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_train_4_cut_urban$bestTune$max_depth,
  gamma = xgb_train_4_cut_urban$bestTune$gamma,
  colsample_bytree = xgb_train_4_cut_urban$bestTune$colsample_bytree,
  min_child_weight = xgb_train_4_cut_urban$bestTune$min_child_weight,
  subsample = xgb_train_4_cut_urban$bestTune$subsample
)

xgb_train_5_cut_urban = train(
  x = alz_train_x_cut_urban,
  y = alz_train_y_cut_urban,
  trControl = xgb_trcontrol_1_cut_urban,
  tuneGrid = xgb_grid_5_cut_urban,
  method = "xgbTree"
)





### cut tune xgboost

xgb_params_cut_tune_urban = list(
  booster = "gbtree",
  eta = xgb_train_5_cut_urban$bestTune$eta, 
  gamma = xgb_train_5_cut_urban$bestTune$gamma,
  max_depth = xgb_train_5_cut_urban$bestTune$max_depth,
  min_child_weight =xgb_train_5_cut_urban$bestTune$min_child_weight,
  subsample = xgb_train_5_cut_urban$bestTune$subsample,
  colsample_bytree = xgb_train_5_cut_urban$bestTune$colsample_bytree,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
# rmse for train and test displayed
watchlist_cut_tune_urban <- list(train = alz_xgboost_train_cut_urban, test = alz_xgboost_test_cut_urban)

set.seed(8009)
#xgboost for model
xgb_model_cut_tune_urban <- xgb.train(
  data = alz_xgboost_train_cut_urban,
  nrounds = 5000,
  watchlist = watchlist_cut_tune_urban,
  params = xgb_params_cut_tune_urban,
  early_stopping_rounds = 50
)

xgb_model_cut_tune_urban

#model importance
importance_cut_tune_urban <- xgb.importance(feature_names = colnames(alz_train_x_cut_urban), model = xgb_model_cut_tune_urban)


elog_urban <- xgb_model_cut_tune_urban$evaluation_log %>% 
  rename(train = "train_rmse",
         test = "test_rmse") %>% 
  pivot_longer(c(train, test), names_to = "Data", values_to = "RMSE")

elog_tune_cut_urban <- ggplot(elog_urban, aes(x = iter, y= RMSE, group = Data))+
  geom_line(aes(color=Data))+
  scale_color_manual(values=c("#eeb211", "#46166b", "#eeb211"))+
  scale_y_continuous(n.breaks = 10)+
  theme_classic()+
  labs(x = "Training Iterations")+
  ggtitle("Learning Curves for Training and Testing Data Stratified for Low Poverty %")+
  theme(plot.title = element_text(hjust = 0.5))


elog_tune_cut_urban




importance_cut_tune_urban_df <- as_tibble(importance_cut_tune_urban) %>% 
  arrange(desc(Gain)) %>% 
  filter(row_number() <= 15)

importance_plot_cut_urban <- ggplot(importance_cut_tune_urban_df, aes(x = Gain, y = reorder(Feature, Gain)))+
  geom_col(fill = "grey", color = "black")+
  theme_classic()+
  labs(x = "Importance (Gain)", y = "Feature")+
  theme(plot.title = element_text(hjust = 0.5),
        axis.text=element_text(size=12),
        axis.title=element_text(size=12))

importance_plot_cut_urban
library(SHAPforxgboost)

shap_urban <- shap.prep(xgb_model_cut_tune_urban, X_train = alz_train_x_cut_urban)

shap.plot.summary.wrap1(xgb_model_cut_tune_urban, alz_test_x_cut_urban, top_n = 20)


xgb.plot.shap(alz_train_x_cut_urban, model = xgb_model_cut_tune_urban, features = "male")



#######
#######   rural
#######

rural <- all_data_urbanrural %>% 
  filter(urban_rural == "rural")

rural_cor <- rural %>% 
  select( -c(stcofips, st, state, st_abbr, county, alz_spending, pop, glyphosate, dicamba, eal_score,
             resl_score, veteran, vacant, resp, ptsdf, ozone, dslpm, metro, urban_rural))



alz_rural <- rural_cor %>% 
  select(-c(hospital, incarcerate, syphilis, teeth, housingcost, mental, hbp, hincome, nocomputer, snap, log_homevalue))


dmy_rural <- dummyVars(" ~ .", data = alz_rural,fullRank = T)
alz_rural <- data.frame(predict(dmy_rural, newdata = alz_rural))
alz_rural_complete <- alz_rural%>% 
  na.omit()


set.seed(900)
intrain_rural_cut <-  createDataPartition(alz_rural_complete$alzheimers_dementia, p = 0.8, list = F)
alz_train_cut_rural <-  alz_rural_complete[intrain_rural_cut, ]
alz_test_cut_rural <-  alz_rural_complete[-intrain_rural_cut, ]

alz_train_x_cut_rural <-  data.matrix(alz_train_cut_rural[,-1])
alz_train_y_cut_rural <-  alz_train_cut_rural[,1]

alz_test_x_cut_rural <-  data.matrix(alz_test_cut_rural[,-1])
alz_test_y_cut_rural <-  alz_test_cut_rural[,1]

#create matrix for xgboost
alz_xgboost_train_cut_rural = xgb.DMatrix(data=alz_train_x_cut_rural, label=alz_train_y_cut_rural)
alz_xgboost_test_cut_rural = xgb.DMatrix(data=alz_test_x_cut_rural, label=alz_test_y_cut_rural)



set.seed=(1234)
xgb_grid_1_cut_rural = expand.grid(
  nrounds = c(5000),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)



xgb_trcontrol_1_cut_rural = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE,
  search = "grid"
)

xgb_train_1_cut_rural = train(
  x = alz_train_x_cut_rural,
  y = alz_train_y_cut_rural,
  trControl = xgb_trcontrol_1_cut_rural,
  tuneGrid = xgb_grid_1_cut_rural,
  method = "xgbTree"
)






xgb_grid_2_cut_rural <- expand.grid(
  nrounds = xgb_train_1_cut_rural$bestTune$nrounds,
  eta = xgb_train_1_cut_rural$bestTune$eta,
  max_depth = ifelse(xgb_train_1_cut_rural$bestTune$max_depth == 2,
                     c(xgb_train_1_cut_rural$bestTune$max_depth:4),
                     xgb_train_1_cut_rural$bestTune$max_depth - 1:xgb_train_1_cut_rural$bestTune$max_depth + 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3,4,5),
  subsample = 1
)

xgb_train_2_cut_rural = train(
  x = alz_train_x_cut_rural,
  y = alz_train_y_cut_rural,
  trControl = xgb_trcontrol_1_cut_rural,
  tuneGrid = xgb_grid_2_cut_rural,
  method = "xgbTree"
)


xgb_grid_3_cut_rural <- expand.grid(
  nrounds = xgb_train_2_cut_rural$bestTune$nrounds,
  eta = xgb_train_2_cut_rural$bestTune$eta,
  max_depth = xgb_train_2_cut_rural$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_train_2_cut_rural$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_train_3_cut_rural = train(
  x = alz_train_x_cut_rural,
  y = alz_train_y_cut_rural,
  trControl = xgb_trcontrol_1_cut_rural,
  tuneGrid = xgb_grid_3_cut_rural,
  method = "xgbTree"
)


xgb_grid_4_cut_rural <- expand.grid(
  nrounds = xgb_train_3_cut_rural$bestTune$nrounds,
  eta = xgb_train_3_cut_rural$bestTune$eta,
  max_depth = xgb_train_3_cut_rural$bestTune$max_depth,
  gamma = c(0,1,2,3,4,5,6),
  colsample_bytree = xgb_train_3_cut_rural$bestTune$colsample_bytree,
  min_child_weight = xgb_train_3_cut_rural$bestTune$min_child_weight,
  subsample = xgb_train_3_cut_rural$bestTune$subsample
)


xgb_train_4_cut_rural = train(
  x = alz_train_x_cut_rural,
  y = alz_train_y_cut_rural,
  trControl = xgb_trcontrol_1_cut_rural,
  tuneGrid = xgb_grid_4_cut_rural,
  method = "xgbTree"
)



xgb_grid_5_cut_rural <- expand.grid(
  nrounds = xgb_train_4_cut_rural$bestTune$nrounds,
  eta = c(0.001, 0.005,0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_train_4_cut_rural$bestTune$max_depth,
  gamma = xgb_train_4_cut_rural$bestTune$gamma,
  colsample_bytree = xgb_train_4_cut_rural$bestTune$colsample_bytree,
  min_child_weight = xgb_train_4_cut_rural$bestTune$min_child_weight,
  subsample = xgb_train_4_cut_rural$bestTune$subsample
)

xgb_train_5_cut_rural = train(
  x = alz_train_x_cut_rural,
  y = alz_train_y_cut_rural,
  trControl = xgb_trcontrol_1_cut_rural,
  tuneGrid = xgb_grid_5_cut_rural,
  method = "xgbTree"
)





### cut tune xgboost

xgb_params_cut_tune_rural = list(
  booster = "gbtree",
  eta = xgb_train_5_cut_rural$bestTune$eta, 
  gamma = xgb_train_5_cut_rural$bestTune$gamma,
  max_depth = xgb_train_5_cut_rural$bestTune$max_depth,
  min_child_weight =xgb_train_5_cut_rural$bestTune$min_child_weight,
  subsample = xgb_train_5_cut_rural$bestTune$subsample,
  colsample_bytree = xgb_train_5_cut_rural$bestTune$colsample_bytree,
  objective = "reg:squarederror",
  eval_metric = "rmse"
)
# rmse for train and test displayed
watchlist_cut_tune_rural <- list(train = alz_xgboost_train_cut_rural, test = alz_xgboost_test_cut_rural)

set.seed(8009)
#xgboost for model
xgb_model_cut_tune_rural <- xgb.train(
  data = alz_xgboost_train_cut_rural,
  nrounds = 5000,
  watchlist = watchlist_cut_tune_rural,
  params = xgb_params_cut_tune_rural,
  early_stopping_rounds = 50
)

xgb_model_cut_tune_rural

#model importance
importance_cut_tune_rural <- xgb.importance(feature_names = colnames(alz_train_x_cut_rural), model = xgb_model_cut_tune_rural)

elog_rural <- xgb_model_cut_tune_rural$evaluation_log %>% 
  rename(train = "train_rmse",
         test = "test_rmse") %>% 
  pivot_longer(c(train, test), names_to = "Data", values_to = "RMSE")

elog_tune_cut_rural <- ggplot(elog_rural, aes(x = iter, y= RMSE, group = Data))+
  geom_line(aes(color=Data))+
  scale_color_manual(values=c("#eeb211", "#46166b", "#eeb211"))+
  scale_y_continuous(n.breaks = 10)+
  theme_classic()+
  labs(x = "Training Iterations")+
  ggtitle("Learning Curves for Training and Testing Data Stratified for Low Poverty %")+
  theme(plot.title = element_text(hjust = 0.5))


elog_tune_cut_rural




importance_cut_tune_rural_df <- as_tibble(importance_cut_tune_rural) %>% 
  arrange(desc(Gain)) %>% 
  filter(row_number() <= 15)

importance_plot_cut_rural <- ggplot(importance_cut_tune_rural_df, aes(x = Gain, y = reorder(Feature, Gain)))+
  geom_col(fill = "grey", color = "black")+
  theme_classic()+
  labs(x = "Importance (Gain)", y = "Feature")+
  theme(plot.title = element_text(hjust = 0.5),
        axis.text=element_text(size=12),
        axis.title=element_text(size=12))

importance_plot_cut_rural
library(SHAPforxgboost)

shap_rural <- shap.prep(xgb_model_cut_tune_rural, X_train = alz_train_x_cut_rural)

shap.plot.summary.wrap1(xgb_model_cut_tune_rural, alz_test_x_cut_rural, top_n = 20)


xgb.plot.shap(alz_train_x_cut_rural, model = xgb_model_cut_tune_rural, features = "male")



session_info()
