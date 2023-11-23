# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:55:04 2023

@author: 王佳晨
"""

import pandas as pd
import numpy as np
from scipy.stats import kurtosis, iqr
from tqdm import tqdm
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
import sklearn
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import time
def cate_colName(Transformer, category_cols, drop='if_binary'):
    """
    离散字段独热编码后字段名创建函数
    
    :param Transformer: 独热编码转化器
    :param category_cols: 输入转化器的离散变量
    :param drop: 独热编码转化器的drop参数
    """
    
    cate_cols_new = []
    col_value = Transformer.categories_
    
    for i, j in enumerate(category_cols):
        if (drop == 'if_binary') & (len(col_value[i]) == 2):
            cate_cols_new.append(j)
        else:
            for f in col_value[i]:
                feature_name = str(j) + '_' + str(f)
                cate_cols_new.append(feature_name)
    return(cate_cols_new)

class DataProcessor:
    '''
    数据处理及特征工程模块的集成函数
    包括删除常量、缺失值，生成缺失值布尔变量，目标编码，索引分组统计函数
    '''
    
    @staticmethod
    def create_missing_indicator(data, feature_names):
        
        for feature_name in feature_names:
            missing_indicator = pd.isnull(data[feature_name]).astype(int)
            data[f"{feature_name}_missing"] = missing_indicator
            
        return data
    
    @staticmethod
    def constant_del(df, cols):
        
        dele_list = []
        for col in cols:
            uniq_vals = list(df[col].unique())
            if pd.isnull(uniq_vals).any():
                if len(uniq_vals) == 2:
                    dele_list.append(col)
            elif len(df[col].unique()) == 1:
                dele_list.append(col)
        df = df.drop(dele_list, axis=1)
        
        return df, dele_list
    
    @staticmethod
    def del_na(df, colname_1, rate):
        
        na_cols = df[colname_1].isna().sum().sort_values(ascending=False) / float(df.shape[0])
        na_del = na_cols[na_cols >= rate]
        df = df.drop(na_del.index, axis=1)
        
        return df, list(na_del.index)
    
    @staticmethod
    def add_features(feature_name, aggs, features, feature_names, new):
        
        for feature in feature_name:
            feature_names.extend(['{}_{}'.format(feature, agg) for agg in aggs])

        for agg in tqdm(aggs):
            if agg == 'kurt':
                agg_func = kurtosis
            elif agg == 'iqr':
                agg_func = iqr
            else:
                agg_func = agg

            for feature in feature_name:
                g = new.groupby(['客户编号'])[feature].agg(agg_func).reset_index().rename(index=str, columns={feature: '{}_{}'.format(feature, agg)})
                features = features.merge(g, on='客户编号', how='left')

        return features, feature_names
  
    
    @staticmethod
    def get_age_label(age_years):
        
        if age_years < 30:
            return 0
        elif age_years < 40:
            return 1
        else:
            return 2
    
    @staticmethod
    def label_encoder(df, categorical_columns=None):
        
        if not categorical_columns:
            categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        for col in categorical_columns:
            df[col], uniques = pd.factorize(df[col])
            
        return df, categorical_columns
    
  
        
        
    @staticmethod
    def feature_selection(x_train,y_train,x_test,corr_threshold,lgb_threshold='1.25*median'):
        
        start_time = time.time()

        train = x_train.merge(y_train, on='客户编号', how='left')
        
        # col_corr = set()  
        # corr_matrix = train.corr()
        # target_corr = abs(corr_matrix['复购频率'])
        # for i in range(len(corr_matrix.columns)):
        #    for j in range(i):
        #         if abs(corr_matrix.iloc[i, j]) > corr_threshold and target_corr[i] > corr_threshold:  
        #             colname = corr_matrix.columns[i]  
        #             if colname != '复购频率':
        #                 col_corr.add(colname)
        # col_corr.add('客户编号')
        
        # train=train[list(col_corr)]
        # x_test=x_test[list(col_corr)]
        # print('correlated features:\n ', len(set(col_corr)) ,col_corr)
        
        lgbc=LGBMClassifier(n_estimators=25, learning_rate=0.1,max_depth=5)
        encoder = sklearn.preprocessing.OrdinalEncoder()
        cat_features = train.select_dtypes(include=['object']).columns.tolist()
        train[cat_features] = encoder.fit_transform(train[cat_features])
        embeded_lgb_selector = SelectFromModel(lgbc, threshold=lgb_threshold)
        feats = [f for f in train.columns if f not in ['复购频率', '客户编号']]
        embeded_lgb_selector.fit(train[feats], y_train['复购频率'])
        embeded_lgb_support = embeded_lgb_selector.get_support()
        embeded_lgb_feature = train[feats].loc[:,embeded_lgb_support].columns.tolist()
        embeded_lgb_feature.append('客户编号')
        x_test=x_test[embeded_lgb_feature]
        train=train[embeded_lgb_feature]
        print(str(len(embeded_lgb_feature)), 'lgb selected features:\n',embeded_lgb_feature)
        
        end_time = time.time()
        print("特征筛选成功，耗时：{:.2f}秒".format(end_time-start_time))

        return train,x_test
    
    
    def Binary_Cross_Combination(colNames, features, OneHot=True):
        """
        分类变量两两组合交叉衍生函数
        
        :param colNames: 参与交叉衍生的列名称
        :param features: 原始数据集
        :param OneHot: 是否进行独热编码
        
        :return：交叉衍生后的新特征和新列名称
        """
        
        # 创建空列表存储器
        colNames_new_l = []
        features_new_l = []
        
        # 提取需要进行交叉组合的特征
        features = features[colNames]
        
        # 逐个创造新特征名称、新特征
        for col_index, col_name in enumerate(colNames):
            for col_sub_index in range(col_index+1, len(colNames)):
                
                newNames = col_name + '&' + colNames[col_sub_index]
                colNames_new_l.append(newNames)
                
                newDF = pd.Series(features[col_name].astype('str')  
                                + '&'
                                + features[colNames[col_sub_index]].astype('str'), 
                                name=col_name)
                features_new_l.append(newDF)
        
        # 拼接新特征矩阵
        features_new = pd.concat(features_new_l, axis=1)
        features_new.columns = colNames_new_l
        colNames_new = colNames_new_l
        
        # 对新特征矩阵进行独热编码
        if OneHot == True:
            enc = preprocessing.OneHotEncoder()
            enc.fit_transform(features_new)
            colNames_new = cate_colName(enc, colNames_new_l, drop=None)
            features_new = pd.DataFrame(enc.fit_transform(features_new).toarray(), columns=colNames_new)
            
        return features_new, colNames_new

