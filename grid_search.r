library(modeldata)
library(tidymodels)
data(cells)
cells <- cells %>%select(-case)

#create folds for 10 fold cross validation
set.seed(33)
cell_folds <- vfold_cv(cells)

#recipe to cleanup correlated predictors with unequal variance
mlp_rec <-
  recipe(class ~ ., data = cells) %>%
  step_YeoJohnson(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), num_comp = tune()) %>% 
  step_normalize(all_predictors())

#neural net model specs
mlp_spec <- 
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>% 
  set_engine("nnet", trace = 0) %>% 
  set_mode("classification")

#create workflow object
mlp_wflow <- 
  workflow() %>% 
  add_model(mlp_spec) %>% 
  add_recipe(mlp_rec)

#update parameters for grid search
mlp_param <- 
  mlp_wflow %>% 
  parameters() %>% 
  update(
    epochs = epochs(c(50, 200)),
    num_comp = num_comp(c(0, 40))
  )

#perform grid searc
roc_res <- metric_set(roc_auc)
set.seed(99)

mlp_reg_tune <-
  mlp_wflow %>%
  tune_grid(
    cell_folds,
    grid = mlp_param %>% grid_regular(levels = 3),
    metrics = roc_res
  )

#plot and show best hyperparameter combination
autoplot(mlp_reg_tune) + theme(legend.position = "top")
show_best(mlp_reg_tune) %>% select(-.estimator)

#use maximum entropy approach with 20 candidate values
set.seed(99)
mlp_sfd_tune <-
  mlp_wflow %>%
  tune_grid(
    cell_folds,
    grid = 20,
    # Pass in the parameter object to use the appropriate range: 
    param_info = mlp_param,
    metrics = roc_res
  )
mlp_sfd_tune

#plot results
autoplot(mlp_sfd_tune)

#show the best combination with more exhaustive approach
show_best(mlp_sfd_tune) %>% select(-.estimator)