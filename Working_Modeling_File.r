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
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal())%>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude,Longitude, deg_free = 20)

#fit a linear regression model
lm_model <- linear_reg() %>% set_engine("lm")

lm_wflow <- 
  workflow() %>% 
  add_model(lm_model) %>%
  add_recipe(ames_rec)

lm_fit <- fit(lm_wflow,ames_train)


# fit a random forest model
rf_model <- rand_forest(trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

rf_wflow <- workflow() %>%
  add_formula(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude) %>%
  add_model(rf_model)

set.seed(55)
ames_folds <- vfold_cv(ames_train, v = 10)

keep_pred <- control_resamples(save_pred = TRUE)

set.seed(130)
rf_res <- rf_wflow %>% fit_resamples(resamples = ames_folds, control = keep_pred)
