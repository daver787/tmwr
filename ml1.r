#packages ####

#resampling,splitting, and validation
library(rsample)
#feature engineering,preprocessing
library(recipes)
#specifying models
library(parsnip)
#tuning
library(tune)
#tuning parameters
library(dials)
#measure model performance
library(yardstick)
#variable importance plots
library(vip)
#combining feature engineering and model specification
library(workflows)
#data manipulation
library(dplyr)
library(purrr)
library(tidyr)
#plotting
library(ggplot2)
#parallelism
library(doFuture)
library(parallel)
#timing
library(tictoc)
#viz
library(skimr)



# Data ####
data(credit_data,package='modeldata')
credit <- credit_data %>%as_tibble()
credit
?modeldata::credit_data

#EDA
ggplot(credit_data,aes(x=Status))+
    geom_bar()


ggplot(credit,aes(x=Status,y=Amount))+geom_violin()

ggplot(credit,aes(x=Status,y=Age))+
    geom_violin()

ggplot(credit,aes(x=Status,y=Income))+
    geom_violin()


ggplot(credit,aes(x=Age,y=Income,color=Status))+
    geom_jitter()

ggplot(credit,aes(x=Age,y=Income,color=Status))+
    geom_hex()+ 
    facet_wrap(~Status)+
    theme(legend.position='bottom')


# Split the Data ####
set.seed(8261)
#from {rsample}

credit_split <-initial_split(credit,prop=0.8,strata='Status')
train <- training(credit_split)
test <- testing(credit_split)

#Feature Engineering ####
#also called preprocessing

# from {recipes}
#goal is to relate outcome to inputs
#outcomes: response,y,label,target,output,result,dependent variable,known,event
#inputs:predictors,x,features,covariates,variables,data,independent variables,attributes,descriptors


#two balanced ways to deal with inbalanced data:
#1) Upsample the minority
#2) downsample the majority class
rec1 <-recipe(Status~.,data=train)%>%
    #xgboost can handle this
    themis::step_downsample(Status,under_ratio = 1.2)%>%
    #not really needed
    step_normalize(Age,Price)%>%
    step_other(all_nominal(),-Status,other='Misc')%>%
    #remove variables with very little variance
    step_nzv(all_predictors())%>%
    #imputation:filling in missing values
    step_modeimpute(all_nominal(),-Status) %>%
    step_knnimpute(all_numeric())%>%
    step_dummy(all_nominal(),-Status,one_hot=TRUE)

#aside for dummy variables
library(useful)
colors1 <- tibble(Color=c('blue','green','blue','red','red','yellow','green'))
build.x(~Color,data=colors1)
build.x(~Color,data=colors1,contrasts=FALSE)
    
#to see all your steps
rec1%>%prep()%>%juice()%>%View


#model specification ####

#from {parsnip}
#model types
#model modes
#engines
#parameters
#show_engines('boost_tree')
xg_spec1<- boost_tree(trees=100)%>%
    set_engine('xgboost')%>%
    set_mode('classification')



# Workflows ####
#from {workflows}

flow1 <- workflow()%>%
    add_recipe(rec1)%>%
    add_model(xg_spec1)

fit1 <- fit(flow1,data=train)

#variable importance plot
fit1 %>%extract_model()%>%vip()


# How Did We Do? ####

#AIC,accuracy,logloss,AUC

#from {yardstick}
loss_fn <- metric_set(accuracy,roc_auc,mn_log_loss)

val_split <-validation_split(data=train,prop=0.8,strata='Status')

#from {tune}

val1 <- fit_resamples(object=flow1,resamples=val_split,metrics=loss_fn)
val1 %>%collect_metrics()

#cross validation
cv_split <- vfold_cv(data=train,v=10,strata='Status')

cv1 <- fit_resamples(object=flow1,resamples=cv_split,metrics=loss_fn)
cv1%>%collect_metrics()

xg_spec2 <- boost_tree(trees=300)%>%
    set_engine('xgboost')%>%
    set_mode('classification')


flow2 <- flow1 %>%
    update_model(xg_spec2)

val2 <- fit_resamples(object=flow2,resamples=val_split,metrics=loss_fn)

val2 %>% collect_metrics()
val1 %>% collect_metrics()


xg_spec3 <- boost_tree(trees=300,learn_rate=0.15)%>%
    set_engine('xgboost')%>%
    set_mode('classification')

flow3 <-
    flow2 %>%
    update_model(xg_spec3)

val3<- fit_resamples(object=flow3,resamples=val_split,metrics=loss_fn)
val3%>%collect_metrics()


#More Recipes ####
rec2 <-recipe(Status~.,data=train)%>%
    #xgboost can handle this
    themis::step_downsample(Status,under_ratio = 1.2)%>%
    #not really needed
    step_other(all_nominal(),-Status,other='Misc')%>%
    #remove variables with very little variance
    step_nzv(all_predictors())%>%
    #imputation:filling in missing values
    step_dummy(all_nominal(),-Status,one_hot=TRUE)


flow4 <-
    flow3 %>%
    update_recipe(rec2)
val4 <- fit_resamples(flow4, resamples=val_split,metrics=loss_fn)
val4 %>%collect_metrics()



rec3 <-recipe(Status~.,data=train)%>%
    #not really needed
    step_other(all_nominal(),-Status,other='Misc')%>%
    #remove variables with very little variance
    step_nzv(all_predictors())%>%
    #imputation:filling in missing values
    step_dummy(all_nominal(),-Status,one_hot=TRUE)


#scale_pos_weight
scaler <- train%>%count(Status)%>%pull(n)%>%rev()%>%reduce(`/`)

xg_spec5 <- boost_tree(mode='classification',trees=300,learn_rate=0.15)%>%
    set_engine('xgboost',scale_pos_weight=!!scaler)

flow5 <- flow4 %>%
    update_model(xg_spec5)

val5 <- fit_resamples(flow5, resamples=val_split,metrics=loss_fn)
val5 %>%collect_metrics()

#Tune Parameters ####
# from {tune} and {dials} package

xg_spec6 <- boost_tree(mode='classification',learn_rate=0.15,tree_depth=4,trees=tune())%>%
    set_engine('xgboost',scale_pos_weight=!!scaler)

flow6 <- flow5 %>%
    update_model(xg_spec6)


registerDoFuture()
cl <- makeCluster(6)
plan(cluster,workers=cl)
options(tidymodels.dark=TRUE)



tic()
tune6_val <- tune_grid(
    flow6,
    resamples=val_split,
    grid=30,
    metrics=loss_fn,
    control=control_grid(verbose=TRUE,allow_par=TRUE)
)

toc()
tune6_val %>% show_best(metric='roc_auc',n=30)%>%View()


tic()
tune6_cv <- tune_grid(
    flow6,
    resamples=cv_split,
    grid=30,
    metrics=loss_fn,
    control=control_grid(verbose=TRUE,allow_par=TRUE)
)

toc()
tune6_cv %>% show_best(metric='roc_auc')

xg_spec7 <- boost_tree(
    mode='classification',
    trees=tune(),
    learn_rate=0.15,
    tree_depth=tune(),
    sample_size=tune()
) %>% 
    set_engine('xgboost', scale_pos_weight=!!scaler)

flow7 <- flow6 %>% 
    update_model(xg_spec7)




params7 <- flow7 %>% 
    parameters() %>% 
    update(
        trees=trees(range=c(20,800)),
        tree_depth=tree_depth(range=c(2, 8)),
        sample_size=sample_prop(range=c(0.3, 1))
    )

params7 %>% pull(object)

tic()
tune7_val <- tune_grid(
    flow7,
    resamples=val_split,
    param_info=params7,
    grid=40,
    control=control_grid(verbose=TRUE, allow_par=TRUE)
)
toc()

tune7_val %>% show_best(metric='roc_auc')

grid7 <- grid_max_entropy(params7,size=80)
tic()
tune7_val.1 <- tune_grid(
    flow7,
    resamples=val_split,
    metrics=loss_fn,
    control=control_grid(verbose=TRUE,allow_par=TRUE),
    grid=grid7
)
toc()

tune7_val.1 %>% show_best(metric='roc_auc')


tic()
tune7_val.2 <- tune_bayes(
    flow7,
    resamples=val_split,
    iter=30,
    metrics=loss_fn,
    parm_info=params7,
    control=control_bayes(verbose=TRUE,no_improve=8)
    
    
)
toc()
tune7_val.2 %>% show_best(metric='roc_auc')



# simulated annealing from {finetune}
boost_tree(
    mode='classification',
    learn_rate=0.15,
    trees=157,
    tree_depth=2,
    sample_size=0.958
)%>%
    set_engine('xgboost',scale_pos_weight=!!scaler)

#Finalize Model ####


rec8 <-recipe(Status~.,data=train)%>%
    #not really needed
    step_other(all_nominal(),-Status,other='Misc')%>%
    #remove variables with very little variance
    step_nzv(all_predictors(),freq_cut=tune())%>%
    #imputation:filling in missing values
    step_dummy(all_nominal(),-Status,one_hot=TRUE)

flow8 <- flow7 %>%
    update_recipe(rec8)

params8 <- flow8 %>%
    parameters()%>%
    update(
        trees=trees(range=c(20,800)),
        tree_depth=tree_depth(range=c(2,8)),
        sample_size=sample_prop(range=c(0.3,1)),
        freq_cut=freq_cut(range=c(5,25))
    )


grid8 <- grid_max_entropy(params8,size=80)

tic()
tune8_val <- tune_grid(
    flow8,resamples=val_split,
    grid=grid8,
    metrics=loss_fn,
    contol_grid(verbose=TRUE,allow_par=TRUE)
)
toc()


best_params <- tune8_val %>% select_best(metric='roc_auc')

flow8_final <- flow8 %>% finalize_workflow(best_params)


val8 <-fit_resamples(flow8_final,resamples = val_split,metrics=loss_fn)
val8 %>% collect_metrics()


#Last Fit ####
results8 <- last_fit(flow8_final,split=credit_split,metrics=loss_fn)
results8%>% collect_metrics()


#Fit on entire dataset ####
fit8 <- fit(flow8_final,data=credit)
#preds8 <- predict(fit8,new_data=insert_data_here)
