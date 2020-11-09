# -*- coding: utf-8 -*-
"""
Created on 22-10-2020

@author: GK
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb1
import xgboost as xgb1
import catboost as cb1
import pickle
import os
import gc
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from xgboost import plot_importance
from matplotlib import pyplot
gc.enable()
from sklearn.model_selection import GridSearchCV

## Please change the path of directories to run the file on any other PC.

data_directory = "C:\\Users\\Dell 3450\\Desktop\\INDEED_ASSIGNMENT\\"
investigation_directory = "C:\\Users\\Dell 3450\\Desktop\\INDEED_ASSIGNMENT\\Investigation\\"

### This method calculates the proporation of columns with missing values.
def cal_missing_percentage(x_df,x_str):
    percent_missing = (x_df.isnull().sum() / len(x_df))*100
    percent_missing_df = percent_missing.to_frame().reset_index().rename(index=str,columns={'index':'Columns',0:'Missing_Percentage'})
    percent_missing_df.to_csv(investigation_directory + "percent_missing" + x_str +"_df.csv")
    percent_missing_df.plot(title=x_str)
    
    
## This method plots the proporation of rows with missing values.
def plot_missing_Rows(x_df,x_str):
    rows_missing_list = x_df.isnull().sum(axis=1)
    bins = np.arange(0, 50, 5) # fixed bin size
    plt.title(x_str)
    plt.xlim([min(rows_missing_list)-5, max(rows_missing_list)+5])
    plt.hist(rows_missing_list, bins=bins, alpha=0.5)
    
## This method plots the histogram of the numeric columns, to check their general distribution
def plot_distributions_hist(x_df,col_list):
    for col in col_list:
        x_df[[col]].hist(bins=10) 
        
        
## remove invalid rows, to remove zero or negative salary. as well as any negaitve miesFromMetropolis
## as well as
def remove_inValid_rows(x_df,x_str):
    if(x_str == 'train'):
         x_df = x_df.loc[x_df.salary > 0,:]
    x_df = x_df.loc[x_df.milesFromMetropolis >= 0,:]
    x_df = x_df.loc[x_df.yearsExperience >= 0,:]
    return x_df
        

### This method plots the boxplots of numeric columns with respect to the dependent variable,
### to check the which are the columns which affects the classification broadly.
def plotBoxPlotsByDV(x_df,cType):
    
    print("columns in the x_df ",list(x_df))
    if(cType == 'machine_status'):
        for col in list(x_df):
#            plot_1 = plt.figure()
            if(col != 'machine_status'):
                by_cols = [col,'machine_status']
                x_df[by_cols].boxplot(by=['machine_status'])
#            plot_1.savefig(col + ".svg",format="svg")
#            
    

## This method does the modelling using Gradient Boosting Regressor with GridsearchCV.
## This method has not been used in the final solution, but i was trying this as well.
def fit_GradientBoostingRegressor(tr_X,te_X,tr_Y,te_Y):
    param_grid={'n_estimators':[1000,2000], 'learning_rate': [0.1,0.05],# 0.05, 0.02, 0.01],
                'max_depth':[6,4],#4,6],
                'min_samples_leaf':[3,5],#,5,9,17],
                'max_features':[1.0,0.3],#,0.3]#,0.1]
                }
    n_jobs=4
    
    estimator = GradientBoostingRegressor() #Choose cross-validation generator - let's choose ShuffleSplit which randomly shuffles and selects Train and CV sets #for each iteration. There are other methods like the KFold split.
#    cv = ShuffleSplit(tr_X.shape[0], n_iter=10, test_size=0.2) #Apply the cross-validation iterator on the Training set using GridSearchCV. This will run the classifier on the #different train/cv splits using parameters specified and return the model that has the best results #Note that we are tuning based on the F1 score 2PR/P+R where P is Precision and R is Recall. This may not always be #the best score to tune our model on. I will explore this area further in a seperate exercise. For now, we'll use F1.
    regressor = GridSearchCV(estimator=estimator,cv = 5, param_grid=param_grid, n_jobs=n_jobs) #Also note that we're feeding multiple neighbors to the GridSearch to try out. #We'll now fit the training dataset to this classifier
    regressor.fit(tr_X, tr_Y) #Let's look at the best estimator that was found by GridSearchCV print "Best Estimator learned through GridSearch" print classifier.best_estimator_ - See more at: https://shankarmsy.github.io/stories/gbrt-sklearn.html#sthash.PARlmKFc.dpuf
    
    best_grid = regressor.best_estimator_
    
    print("Best Estimator learned through GridSearch")
    print(best_grid)
    
    mse_rep_gbr = mean_squared_error(te_Y, best_grid.predict(te_X))
    print("Value of mse_rep_gbr ====>>> ",mse_rep_gbr)
    
    return mse_rep_gbr,best_grid

## This method trains the Lightbgm with Grid search for Regreesion
def fit_LightGradientBoostGridSearch(X_fit, y_fit, X_val, y_val, counter, lgb_path, name):
    
    print("ENTER Inside Light Gradien Boost with Grid Search========>>>>>>>> ")
    
    param_grid={'n_estimators':[100], 'learning_rate': [0.1,0.05],# 0.05, 0.02, 0.01],
                'max_depth':[6,4],#4,6],
                'min_samples_leaf':[3,5],#,5,9,17],
                'max_features':[1.0,0.3],#,0.3]#,0.1]
                }
    n_jobs=3
    
    estimator = lgb1.LGBMRegressor()
    
    regressor = GridSearchCV(estimator=estimator,cv = 5, param_grid=param_grid, n_jobs=n_jobs,verbose = 2) #Also note that we're feeding multiple neighbors to the GridSearch to try out. #We'll now fit the training dataset to this classifier
    regressor.fit(X_fit, y_fit) #Let's look at the best estimator that was found by GridSearchCV print "Best Estimator learned through GridSearch" print classifier.best_estimator_ - See more at: https://shankarmsy.github.io/stories/gbrt-sklearn.html#sthash.PARlmKFc.dpuf
    
    best_grid = regressor.best_estimator_
    
#    model = lgb1.LGBMRegressor(max_depth=-1,
#                              n_estimators=999999,
#                              learning_rate=0.005,
#                              colsample_bytree=0.4,
#                              num_leaves=5,
#                              metric='rmse',
#                              objective='regression', 
#                              n_jobs=-1)
#     
#    model.fit(X_fit, y_fit, 
#              eval_set=[(X_val, y_val)],
#              verbose=0, 
#              early_stopping_rounds=50)
                  
    cv_val = best_grid.predict(X_val)
    
    #Save LightGBM Model
    save_to = '{}{}_fold{}.txt'.format(lgb_path, name, counter+1)
    best_grid.booster_.save_model(save_to)
    print("DONE Inside Light Gradien Boost with Grid Search========>>>>>>>> ")
    return cv_val



## This method trains the XGBoost with Grid search for Regreesion
def fit_XtremeGradientBoostGridSearch(X_fit, y_fit, X_val, y_val, counter, xgb_path, name):
    print("Inside Xtreme Gradien Boost with Grid Search========>>>>>>>> ")
    

    estimator = xgb1.XGBRegressor()
    
    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [0.05,0.04], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [100], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}
    
    n_jobs = 3

    regressor = GridSearchCV(estimator=estimator,cv = 5, param_grid=parameters, n_jobs=n_jobs,verbose = 2) #Also note that we're feeding multiple neighbors to the GridSearch to try out. #We'll now fit the training dataset to this classifier
    regressor.fit(X_fit, y_fit) #Let's look at the best estimator that was found by GridSearchCV print "Best Estimator learned through GridSearch" print classifier.best_estimator_ - See more at: https://shankarmsy.github.io/stories/gbrt-sklearn.html#sthash.PARlmKFc.dpuf
    
    best_grid = regressor.best_estimator_

    
#    clf.fit(X_fit, y_fit)
              
    cv_val = best_grid.predict(X_val)
    
    #Save XGBoost Model
    save_to = '{}{}_fold{}.dat'.format(xgb_path, name, counter+1)
    pickle.dump(best_grid, open(save_to, "wb"))
    return cv_val
    

    
## This method trains the catboost model
def fit_KataBoost(X_fit, y_fit, X_val, y_val, counter, cb_path, name):
    
    model = cb1.CatBoostRegressor(iterations=100,
                                  learning_rate=0.005,
                                  colsample_bylevel=0.03,
                                  objective="RMSE")
                                  
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)], 
              verbose=0, early_stopping_rounds=100)
              
    cv_val = model.predict(X_val)
    
    #Save Catboost Model          
    save_to = "{}{}_fold{}.mlmodel".format(cb_path, name, counter+1)
    model.save_model(save_to, format="coreml")
    
    return cv_val

## This method trains the 3 models lightbgm, xgboost and catboost
def step_train_models(df_path, lgb_path, xgb_path, cb_path,df_train_features):
    
    
     
    df_train = pd.get_dummies(df_train_features,columns=['companyId','jobType','degree','major','industry'])
   
    print('\nShape of Train Data====>>>>>> : {}'.format(df_train.shape))
    
    y_df = np.array(df_train['salary'])                        
    df_ids = np.array(df_train.jobId)                     
    df_train.drop(['jobId', 'salary'], axis=1, inplace=True)
    
    lgb_cv_result = np.zeros(df_train.shape[0])
    xgb_cv_result = np.zeros(df_train.shape[0])
    cb_cv_result  = np.zeros(df_train.shape[0])
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=23)
    skf.get_n_splits(df_ids, y_df)
    
    print('\n Start Fitting the Models=========>>>>>>>> \n ')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('\n Cross Validation FoldNumber {}'.format(counter+1))
        X_fit, y_fit = df_train.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df_train.values[ids[1]], y_df[ids[1]]
    
        print('Light Gradient Boost ========>>>>>>> \n ')
        
        lgb_cv_result[ids[1]] += fit_LightGradientBoostGridSearch(X_fit, y_fit, X_val, y_val, counter, lgb_path, name='lgb')
        
        
        print('Xtreme Gradient Boost ========>>>>>> \n ')
        
        xgb_cv_result[ids[1]] += fit_XtremeGradientBoostGridSearch(X_fit, y_fit, X_val, y_val, counter, xgb_path, name='xgb')
        
        
        print('Cata Boost ===========>>>>>>>>>> \n ')
        cb_cv_result[ids[1]]  += fit_KataBoost(X_fit,  y_fit, X_val, y_val, counter, cb_path,  name='cb')
        
        del X_fit, X_val, y_fit, y_val
        gc.collect()
    
    rmse_lgb  = round(math.sqrt(mean_squared_error(y_df, lgb_cv_result)),4)
    rmse_xgb  = round(math.sqrt(mean_squared_error(y_df, xgb_cv_result)),4)
    rmse_cb   = round(math.sqrt(mean_squared_error(y_df, cb_cv_result)), 4)
    rmse_mean = round(math.sqrt(mean_squared_error(y_df, (lgb_cv_result+xgb_cv_result+cb_cv_result)/3)), 4)
    rmse_mean_lgb_cb = round(math.sqrt(mean_squared_error(y_df, (lgb_cv_result+cb_cv_result)/2)), 4)
    
    print('\nLightGBM RMSE======>>>: {}'.format(rmse_lgb))
    print('XGBoost  VAL RMSE ======>>>>: {}'.format(rmse_xgb))
    print('Catboost VAL RMSE ====>>>>>: {}'.format(rmse_cb))
    print('Mean Catboost Plus LightGBM RMSE: =========>>>> {}'.format(rmse_mean_lgb_cb))
    print('Mean XGBoost and Catboost and LightGBM, VAL RMSE:=======>>>>>  {}\n'.format(rmse_mean))
    
    return 0
    

## this method creates the ensemble model using the trained and saved models light bgm, xgboost and catboost.
## First prediction is made from trained data from the 3 models, then they are fed into the stacking ensemble
## After that it creates final prediction for test data.
def step_stack_ensemble_and_predict(train_path, lgb_model_dir, xgb_model_dir, cb_model_dir,test_path):
    
    lgb_model_weightabge = 0.3
    xgb_model_weightabge = 0.3
    cb_model_weightabge = 0.4
    
    ###################################################################################33
    ############################# PREDICTING FOR TRAIN DATA BY USING TRAINED MODELS ##############################
    ####################################################################################
    
    
   
    print('Load Train Data. for Stacking====>>>>>>>')
    df_train_features = pd.read_csv(train_path)
    df_train = pd.get_dummies(df_train_features,columns=['companyId','jobType','degree','major','industry'])
    df_target = pd.read_csv(target_data_path)
    df_train = pd.merge(df_train,df_target)
    print('\nShape of Train Data:===>>>> {}'.format(df_train.shape))
    
    
    ## Remove any NA, Infiniy
    df_train = remove_inValid_rows(df_train,'train')
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_train.dropna(inplace=True)
    
    
    
    ## Samppling just to make it fast
#    df_train = df_train.sample(n = 500000)
    print("Shape of df_train====>>>> ",str(df_train.shape))
    
    ### Added so that length remains correct after removing some rows with issues
    df_target = df_train[['jobId','salary']]
    
    df_train.drop(['jobId','salary'], axis=1, inplace=True)
#    
    
    lgb_saved_models = sorted(os.listdir(lgb_model_dir))
    xgb_saved_models = sorted(os.listdir(xgb_model_dir))
    cb_saved_models  = sorted(os.listdir(cb_model_dir))
    
    lgb_result = np.zeros(df_train.shape[0])
    xgb_result = np.zeros(df_train.shape[0])
    cb_result  = np.zeros(df_train.shape[0])
    
    print('\nMake predictions===>>>> \n')
    
    print('START With Train Data Light GBM====.>>>>.')
    for this_model in lgb_saved_models:
        #Load LightGBM Model
        model = lgb1.Booster(model_file='{}{}'.format(lgb_model_dir, this_model))
        lgb_result += model.predict(df_train.values)
    print('END With Train Data Light GBM====.>>>>.')
        
    print('START With Train Data XG Boost =====>>>>>>')    
    for this_model in xgb_saved_models:
        #Load XGBOOST Model
        model = pickle.load(open('{}{}'.format(xgb_model_dir, this_model), "rb"))
        xgb_result += model.predict(df_train.values)
    print('END With Train Data XG Boost =====>>>>>>')
    
    print('START With Train Data Cat Boost ======>>>>>>')        
    for this_model in cb_saved_models:
        #Load Catboost Model
        model = cb1.CatBoostRegressor()
        model = model.load_model('{}{}'.format(cb_model_dir, this_model), format = 'coreml')
        cb_result += model.predict(df_train.values)
    print('END With Train Data Cat Boost ======>>>>>>')
    
    lgb_result = lgb_result / len(lgb_saved_models)
    xgb_result = xgb_result / len(xgb_saved_models)
    cb_result  = cb_result / len(cb_saved_models)
    
    
    ensemble_submiss = pd.DataFrame()
    
    
    
    ensemble_submiss['target'] = df_target['salary']
    
    print("Length of lgb_result ", len(lgb_result))
    print("Length of xgb_result ", len(xgb_result))
    print("Length of cb_result ", len(cb_result))
    
    ensemble_submiss['weighted_ensemble'] = (lgb_model_weightabge*lgb_result)+(xgb_model_weightabge*xgb_result)+(cb_model_weightabge*cb_result)
    ensemble_submiss['simple_ensemble'] = (lgb_result + cb_result + xgb_result)/3
    ensemble_submiss['xgb_result'] = xgb_result
    ensemble_submiss['lgb_result'] = lgb_result
    ensemble_submiss['cb_result'] = cb_result
    print('\nSimple Ensemble value Root Mean Square Error===>>>: {}'.format(round(math.sqrt(mean_squared_error(df_target['salary'], ensemble_submiss['simple_ensemble'])),4)))
    print('\nWeighted Ensemble value Root Mean Square Error===>>>: {}'.format(round(math.sqrt(mean_squared_error(df_target['salary'], ensemble_submiss['weighted_ensemble'])),4)))
    print('\nLightGBM value Root Mean Square Error===>>: {}'.format(round(math.sqrt(mean_squared_error(df_target['salary'], ensemble_submiss['lgb_result'])),4)))
    print('\nXGBOOST value Root Mean Square Error==>>>: {}'.format(round(math.sqrt(mean_squared_error(df_target['salary'], ensemble_submiss['xgb_result'])),4)))
    print('\nCATBOOST value Root Mean Square Error==>>>: {}'.format(round(math.sqrt(mean_squared_error(df_target['salary'], ensemble_submiss['cb_result'])),4)))
    ensemble_submiss.to_csv(data_directory + 'train_submission_of_ensembles.csv')
    
    
    print('Loading now Model Predictions======>>>>>>>')
    print('\nShape of Predictions Data Data=====>>>>>>: {}'.format(ensemble_submiss.shape))
    print('Stacking Using H2O Deep Learning =====>>>>>>>')
    
    ###################################################################################33
    ############################# STACKING ENSEMBLE OF TRAINED MODELS ##############################
    ####################################################################################
    
    
    h2o.init(ip="localhost", port=54321)
    train = h2o.H2OFrame(ensemble_submiss)
    train,valid,test = train.split_frame(ratios=[0.7, 0.15], seed=42)
    y = 'target'
    X = list(train.columns)
    X.remove(y)
    

    print('Training Deep Learning Model==========>>>>>')
    stacked_ensemble_model =  H2ODeepLearningEstimator(training_frame=train,
                  validation_frame=valid,
                  stopping_rounds=10,
                  stopping_tolerance=0.0005,
                  epochs = 5000,
                  adaptive_rate = True,  
                  stopping_metric="rmse",
                  hidden=[128,128,128],      
                  balance_classes= False,
                  standardize = True,  
                  loss = "absolute",
                  activation =  'RectifierWithDropout',
                  input_dropout_ratio =  0.04,
                  l1 = 0.00002,
                  l2 = 0.00002,
                  max_w2 = 10.0,
                  hidden_dropout_ratios = [0.01,0.01,0.01])
    
    stacked_ensemble_model.train(X,y,train)
    print('Deep Learning Model Performance on Train and Validation============>>>>>>')
    stacked_ensemble_model
    print('Deep Learning Model Performance on Test Partition==========>>>>>>>>>>>')
    stacked_ensemble_model.model_performance(test)
    
    stacked_model_df = stacked_ensemble_model.score_history()
    
    plt.plot(stacked_model_df['training_rmse'], label="training_rmse")
    plt.plot(stacked_model_df['validation_rmse'], label="validation_rmse")
    plt.title("Stacked Deep models (Tuned)")
    plt.legend();
    
    ###################################################################################33
    ############################# PREDICTION ON TEST DATA ##############################
    ####################################################################################
    
    print('Beginning Final Prediction for Submission for test dataset ======>>>>>> ')
    print('Loading Test Data========>>>>>>> ')
    df_test_features = pd.read_csv(test_path)
    df_test = pd.get_dummies(df_test_features,columns=['companyId','jobType','degree','major','industry'])
    test_id = df_test.jobId
    
    
    print('\nShape of Test Data:====>>>>>>>  {}'.format(df_test.shape))
    df_test.drop(['jobId'], axis=1, inplace=True)
   

    
    lgb_test_result = np.zeros(df_test.shape[0])
    xgb_test_result = np.zeros(df_test.shape[0])
    cb_test_result  = np.zeros(df_test.shape[0])
    
    print('\Start predicting for final test dataset======>>>>>\n')
    
    print('START With Test Data Light GBM ====>>>>>')
    for this_lgb_model in lgb_saved_models:
        #Load the saved LightGBM Model
        model = lgb1.Booster(model_file='{}{}'.format(lgb_model_dir, this_lgb_model))
        lgb_test_result += model.predict(df_test.values)
    print('END With Test Data Light GBM ====>>>>>')
        
    print('START With Test Data XGBoost =====>>>>> ')    
    for this_xgb_model in xgb_saved_models:
        #Load the saved XGBOOST Model
        model = pickle.load(open('{}{}'.format(xgb_model_dir, this_xgb_model), "rb"))
        xgb_test_result += model.predict(df_test.values)
    print('END With Test Data XGBoost =====>>>>> ') 
    
    print('START With Test Data CatBoost =====>>>>')        
    for this_cb_model in cb_saved_models:
        #Load the saved Catboost Model
        model = cb1.CatBoostRegressor()
        model = model.load_model('{}{}'.format(cb_model_dir, this_cb_model), format = 'coreml')
        cb_test_result += model.predict(df_test.values)
    print('END With Test Data CatBoost =====>>>>') 
    
    lgb_test_result /= len(lgb_saved_models)
    xgb_test_result /= len(xgb_saved_models)
    cb_test_result  /= len(cb_saved_models)
    
    
    test_submiss = pd.DataFrame()
    test_submiss['weighted_ensemble'] = (lgb_model_weightabge*lgb_test_result)+(xgb_model_weightabge*xgb_test_result)+(cb_model_weightabge*cb_test_result)
    test_submiss['simple_ensemble'] = (lgb_test_result + cb_test_result + xgb_test_result)/3
    test_submiss['xgb_result'] = xgb_test_result
    test_submiss['lgb_result'] = lgb_test_result
    test_submiss['cb_result'] = cb_test_result
    
    test_submission = h2o.H2OFrame(test_submiss)
    pred = stacked_ensemble_model.predict(test_submission).as_data_frame(use_pandas=True)
    submission_test = pd.DataFrame()
    submission_test['jobId'] = test_id
    submission_test['salary'] = pred
    submission_test.to_csv(investigation_directory + 'test_salaries.csv',index=False)
    

## this method calculates variable importance based on random forest and plots the variable importance
## Calculates and plots variable importance based on Random forest Modelling.
def calculatePlot_Variable_Importance(tr_x,tr_y):
    regr = RandomForestRegressor(max_depth=4,random_state=0,n_estimators=100)
    regr.fit(tr_x,tr_y)

    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking=======>>>>>>>>>>>>>>>")

    for f in range(tr_x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances=======>>>>>>>>>>>>>")
    plt.bar(range(tr_x.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
    plt.xticks(range(tr_x.shape[1]), indices)
    plt.xlim([-1, tr_x.shape[1]])
    plt.show()
    
    return indices, importances

## This method id used to do basic cleanings and EDA and finding the variable importance of features
## using Random Forest.
def eda_And_Feature_Selection():
    print('Load Train Data====================>>>>>>>>>>')
    df_train = pd.read_csv(train_data_path)
    
    df_target = pd.read_csv(target_data_path)
    
    
    df_train_1 = pd.merge(df_train,df_target)
    cal_missing_percentage(df_train_1,'Indeed_Salary')
    ## Above shows that there is no NA present in the dataset.
    
    
    df_train_2 = remove_inValid_rows(df_train_1,'train')
    df_train_2 = df_train_2.replace([np.inf, -np.inf], np.nan)
    df_train_2.dropna(inplace=True)
    
    
    plot_distributions_hist(df_train_2,['yearsExperience','milesFromMetropolis'])
    
    
    
    df_train_2.describe()
    df_train_3 = pd.get_dummies(df_train_2[['jobType','degree','major','industry','companyId']],columns=['jobType','degree','major','industry','companyId'])
    #df_numeric_1 = 
    df_train_4 = pd.concat([df_train_3.reset_index(drop = True),df_train_2[['yearsExperience','milesFromMetropolis']].reset_index(drop=True)],axis=1)
       
    
    
    indices_list, importance_list = calculatePlot_Variable_Importance(df_train_4,df_train_2['salary'].tolist())
    col_List = list(df_train_4)
    sorted_imp_col_List = [col_List[i] for i in indices_list]
    sorted_importance = [importance_list[i] for i in indices_list]
    
    ## Here I am saving the features and their corresponding feature importance sorted as a dataframe to a csv.
    
    variable_importance_df = pd.DataFrame(data={'features':sorted_imp_col_List,'importance':sorted_importance})
    variable_importance_df.to_csv(investigation_directory + "variable_importance_df.csv")

   

    
if __name__ == '__main__':
    
    train_data_path = 'C:\\Users\\Dell 3450\\Desktop\\INDEED_ASSIGNMENT\\train_features.csv'
    target_data_path = 'C:\\Users\\Dell 3450\\Desktop\\INDEED_ASSIGNMENT\\train_salaries.csv'
    test_data_path  = 'C:\\Users\\Dell 3450\\Desktop\\INDEED_ASSIGNMENT\\test_features.csv'
    
    lgb_model_path = 'C:\\Users\\Dell 3450\\Desktop\\INDEED_ASSIGNMENT\\lgb_model_stack\\'
    xgb_model_path = 'C:\\Users\\Dell 3450\\Desktop\\INDEED_ASSIGNMENT\\xgb_model_stack\\'
    cb_model_path = 'C:\\Users\\Dell 3450\\Desktop\\INDEED_ASSIGNMENT\\cb_model_stack\\'

    ## Dosome EDA and Plot VAriable importance of features as well.
    eda_And_Feature_Selection()

    ## Train and Save the models.
    step_train_models(train_data_path, lgb_model_path, xgb_model_path, cb_model_path)
    
    #ensemble modelling and predicting the final test salaries
    print('Commencing Stacking and Prediction Stage.\n')
    step_stack_ensemble_and_predict(train_data_path, lgb_model_path, xgb_model_path, cb_model_path,test_data_path)
    
    


    


