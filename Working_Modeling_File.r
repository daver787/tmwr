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
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal())%>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude,Longitude, deg_free = 20)


ames_rec_prepped <- prep(ames_rec, training = ames_train)
ames_train_prepped <-bake(ames_rec_prepped, new_data = NULL)
ames_test_prepped <-bake(ames_rec_prepped, new_data = ames_test)

#use preprocessing from recipe and fit linear regression model
lm_fit <-lm(Sale_Price ~.,data=ames_train_prepped)
glance(lm_fit)
tidy(lm_fit)

#predict using model object and test data
predict(lm_fit, ames_test_prepped %>% head())