library(tidymodels)
library(patchwork)
library(splines)
data(ames)
ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))

set.seed(123)
ames_split <- initial_split(ames, prob = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)


ames_rec <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude,
         data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10, id = "my_id") %>% 
  step_other(Neighborhood, threshold = tune()) %>% 
  step_dummy(all_nominal())%>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Longitude, deg_free = tune("longitude df")) %>% 
  step_ns(Latitude,  deg_free = tune("latitude df"))

#fit a linear regression model
lm_model <- linear_reg() %>% set_engine("lm")

#fit a neural net model
neural_net_spec <- 
  mlp(hidden_units = tune()) %>% 
  set_engine("keras")

#fit a random forest model
rf_spec <- 
  rand_forest(mtry = tune()) %>% 
  set_engine("ranger", regularization.factor = tune("regularization"))


lm_wflow <- 
  workflow() %>% 
  add_model(lm_model) %>%
  add_recipe(ames_rec)



wflow_param_nn <- 
  workflow() %>% 
  add_recipe(ames_rec) %>% 
  add_model(neural_net_spec) %>% 
  parameters()


parameters(ames_rec) %>% 
  update(threshold = threshold(c(0.8, 1.0)))



wflow_param_rf <- 
  workflow() %>% 
  add_recipe(ames_rec) %>% 
  add_model(rf_spec) %>% 
  parameters()

rf_param <- parameters(rf_spec)

#update rf parameters
rf_param %>% 
  update(mtry = mtry(c(1, 70)))



pca_rec <- 
  recipe(Sale_Price ~ ., data = ames_train) %>% 
  # Select the square-footage predictors and extract their PCA components:
  step_normalize(contains("SF")) %>% 
  # Select the number of components needed to capture 95% of
  # the variance in the predictors. 
  step_pca(contains("SF"), threshold = .95)

updated_param <- 
  workflow() %>% 
  add_model(rf_spec) %>% 
  add_recipe(pca_rec) %>% 
  parameters() %>% 
  finalize(ames_train)
