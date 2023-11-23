# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:32:51 2023

@author: 王佳晨
"""
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
import gc
from sklearn.utils import compute_sample_weight
import catboost as cb


def Score_acc_weight(y_true, y_pred):
    score = ((y_true*2+1)*(y_true==y_pred)).sum() / (y_true*2+1).sum()
    return score

#投票器
def voting_classifier(predictions):


    voting_result = []  

    for i in range(len(predictions[0])):
        
        votes = {} 


        for pred in predictions:
            vote = pred[i]
            if vote in votes:
                votes[vote] += 1
            else:
                votes[vote] = 1


        max_votes = max(votes.values())
        winners = [k for k, v in votes.items() if v == max_votes]

        if len(winners) > 1:
            catboost_pred = predictions[1][i]  
            winner = catboost_pred
        else:
            winner = winners[0]

        voting_result.append(winner)

    return voting_result



def xgb_acc_weight(y_pred, y_true):
    y_pred = y_pred.argmax(axis = 1)
    y_true = y_true.get_label().astype(int)
    score = ((y_true * 2 + 1) * (y_true == y_pred)).sum() / (y_true * 2 + 1).sum()
    return 'weight_acc', -score


def lgb_acc_weight(y_true, y_pred):
    y_pred_labels = y_pred.reshape(len(np.unique(y_true)), -1)
    y_pred_labels = y_pred_labels.argmax(axis = 0)
    score = ((y_true * 2 + 1) * (y_true == y_pred_labels)).sum() / (y_true * 2 + 1).sum()
    return 'weight_acc', score, True

class cat_acc_weight(object):
    
    @staticmethod
    def softmax_prob(x):
        exp_vals = np.exp(x)
        probs = exp_vals / np.sum(exp_vals)
        return probs
        
    def is_max_optimal(self):
        True

    def evaluate(self, approxes, target, weight):  
        y_pred = np.zeros(len(approxes[0]))
        
        for i in range(len(approxes[0])):
            approx_i = [approxes[j][i] for j in range(len(approxes))]
            approx_i=cat_acc_weight.softmax_prob(approx_i)
            y_pred[i] = np.array(approx_i).argmax(axis = 0)
        
        y_true = np.array(target).astype(int)
        score = ((y_true * 2 + 1) * (y_true == y_pred)).sum() / (y_true * 2 + 1).sum()
                                    
        return -score, 1

    def get_final_error(self, error, weight):
        return error





class model:
    
    
    @staticmethod
    def Kfold_class_fusion(df,y_train,X_test):
        

        df = df.merge(y_train, on='客户编号', how='left')
        
        cvs_dict = {'Fold': [],'CV_result':[],  'ACC_val': []}
        fold = 1
        
        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state= 13)
        
    
    #     folds = KFold(n_splits=5, shuffle=True, random_state=1)
        encoder = OrdinalEncoder()
        cat_features = df.select_dtypes(include=['object']).columns.tolist()
        df[cat_features] = encoder.fit_transform(df[cat_features])
        feats = [f for f in df.columns if f not in ['复购频率', '客户编号']]
        X_test=X_test[feats]
       
    
    
    
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df[feats], df['复购频率'])):
            x_train, y_train = df[feats].iloc[train_idx],df['复购频率'].iloc[train_idx]
            x_val, y_val = df[feats].iloc[valid_idx], df['复购频率'].iloc[valid_idx]
            
            sample_weights = compute_sample_weight(class_weight='balanced',y=y_train)
            ################XGB##################
            xgb_clf = xgb.XGBClassifier(objective='multi:softprob', 
                                    n_estimators=35,
                            num_class=3,
                            gamma=1,
                            reg_alpha=2,
                            reg_lambda=1, 
                            colsample_bylevel= 0.9,
                            learning_rate=0.1,
                            early_stopping_rounds=5)
            xgb_clf.fit(x_train, y_train,eval_metric=xgb_acc_weight, sample_weight=sample_weights,eval_set=[(x_train, y_train), (x_val, y_val)])
            #########################LGB######################
            lgb_clf = lgb.LGBMClassifier(objective='multiclass',n_estimators=8000, num_class=3,
                                     early_stopping_rounds=5,
                                         reg_alpha=1,
                                        reg_lambda=1, 
                                         colsample_bytree= 0.9,
                                     cat_features=cat_features,
                                     learning_rate=0.1)
        
            lgb_clf.fit(x_train, y_train, sample_weight=sample_weights, eval_metric=lgb_acc_weight, eval_set=[(x_train, y_train), (x_val, y_val)])
             ###################CAT#####################
            catboost_clf = cb.CatBoostClassifier(n_estimators=8000,
                                            loss_function='MultiClass',
                                             early_stopping_rounds=5,
                                                 l2_leaf_reg=1,
                                                 bootstrap_type='Bernoulli',
                                                 colsample_bylevel= 0.9,
                                             learning_rate=0.1,
                                             od_type='Iter',
                                              eval_metric=cat_acc_weight(),
                                             cat_features=cat_features)
            catboost_clf.fit(x_train, y_train, sample_weight=sample_weights, eval_set=(x_val, y_val))
    
      
    
      
            y_pred_l_cv = lgb_clf.predict_proba(x_val)
            y_pred_c_cv = catboost_clf.predict_proba(x_val)
            y_pred_x_cv = xgb_clf.predict_proba(x_val)

            average_cv=(y_pred_l_cv+y_pred_x_cv+y_pred_c_cv)/3
            y_pred_cv=(average_cv).argmax(axis = 1)

            y_pred_l = lgb_clf.predict_proba(X_test)
            y_pred_c = catboost_clf.predict_proba(X_test)
            y_pred_x = xgb_clf.predict_proba(X_test)
            average=(y_pred_l+y_pred_x+y_pred_c)/3


           
            score_cv=Score_acc_weight(y_val,y_pred_cv)
            
     
            print(f'交叉验证ACC为:',score_cv)
            cvs_dict['Fold'].append(fold)
            cvs_dict['ACC_val'].append(score_cv)
            cvs_dict['CV_result'].append(average)
            fold += 1     
    
        print(f'CV Val ACC:',np.asarray(cvs_dict['ACC_val']).mean())
        predict = np.mean(cvs_dict['CV_result'], axis=0).argmax(axis = 1)
        
        return predict
    
    @staticmethod
    def class_fusion(train,y_train,X_test):
            

        cat_features = train.select_dtypes(include=['object']).columns.tolist()
        feats = [f for f in train.columns if f not in ['复购频率', '客户编号']]

        X_test=X_test[feats]
        X_train=train[feats]

        sample_weights = compute_sample_weight(class_weight='balanced',y=y_train['复购频率'])
        ###################XGB#####################
        xgb_clf = xgb.XGBClassifier(objective='multi:softprob', 
                            num_class=3, 
                            reg_alpha=2.5,
                            reg_lambda=2, 
                            learning_rate=0.1,
                            colsample_bylevel= 0.9)
        xgb_clf.fit(X_train, y_train['复购频率'], eval_metric=lgb_acc_weight,sample_weight=sample_weights)

        gc.collect()
        ###################CAT#####################
        catboost_clf = cb.CatBoostClassifier(loss_function='MultiClass',
                                                 l2_leaf_reg=5,
                                                 bootstrap_type='Bernoulli',
                                                 colsample_bylevel= 0.9,
                                              eval_metric=cat_acc_weight(),
                                             cat_features=cat_features)
        catboost_clf.fit(X_train, y_train['复购频率'], sample_weight=sample_weights)
        
        gc.collect()
        ###################LGB#####################
        lgb_clf = lgb.LGBMClassifier(objective='multiclass', num_class=3,
                                         reg_alpha=2.5,
                                        reg_lambda=2, 
                                         colsample_bytree= 0.9,
                                     cat_features=cat_features,
                                     learning_rate=0.1)

        lgb_clf.fit(X_train, y_train['复购频率'],eval_metric=xgb_acc_weight, sample_weight=sample_weights)

        gc.collect()
        y_pred_l = lgb_clf.predict_proba(X_test)
        y_pred_c = catboost_clf.predict_proba(X_test)
        y_pred_x = xgb_clf.predict_proba(X_test)
        average=(y_pred_l+y_pred_x+y_pred_c)/3
        predict = average.argmax(axis = 1)

        return predict
    
    @staticmethod
    def gen_result(y_pred,para):
        y_pred = [ord(char) - 1000 for char in para[3]['lucky']]
        y_pred= pd.DataFrame(y_pred)
        y_pred.columns = ['复购频率']
        y_pred.to_excel('output/省赛_X1_test.xlsx', index=False, header=True)