# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:31:09 2023

@author: 王佳晨
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OrdinalEncoder
import sklearn
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.stats import iqr
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.impute import SimpleImputer
from DataProcessor import DataProcessor 
import time

def count_feature(all_df):
    
    real_df = all_df
    cust_id=real_df['客户编号']
    real_df = real_df.drop('客户编号',axis=1)
    all_df = all_df.drop('客户编号',axis=1)
    feature_list = []
    
    _feature = all_df.copy()
    feats = [f for f in real_df.columns if f not in ['客户编号']]
    for c in tqdm(feats):
        count_dict = real_df[c].value_counts().to_dict()
        _feature[c] = all_df[c].map(count_dict)
    _feature = _feature.add_prefix('concat_count_')
    feature_list.append(_feature)
    
    
    concat_feature = pd.concat(feature_list, axis=1)
    concat_feature = pd.concat([concat_feature, cust_id], axis=1)
    return concat_feature


def round_feature(all_df):
    def my_round(val, digit=0):
        p = 10 ** digit
        return (val * p * 2 + 1) // 2 / p

    cust_id=all_df['客户编号']
    all_df = all_df.drop('客户编号',axis=1)
    feature_list = []
    _feature = all_df.copy()
    feats = all_df.select_dtypes(include=['float64']).columns.tolist()
    for digit in [1,3,5]:
        _feature = all_df[feats]
        for c in tqdm(feats):
            _feature[c] = my_round(_feature[c], digit)
        _feature = _feature.add_prefix(f'round{digit}_')
        feature_list.append(_feature)

    concat_feature = pd.concat(feature_list, axis=1)
    concat_feature = pd.concat([all_df,concat_feature], axis=1)
    concat_feature = pd.concat([cust_id,concat_feature], axis=1)

    return concat_feature



class Pipeline:
    
    
    '''
    X1、X2、X3批量处理的通道
    '''
    

    @staticmethod
    def X1_pipeline(X1_train_path,X1_test_path,y_train_path,if_pseudo):
            
        start_time = time.time()
        
        X1_train = pd.read_csv(X1_train_path)
        X1_test = pd.read_csv(X1_test_path)
        y_train = pd.read_csv(y_train_path)
        
    
         #合并数据
        df = pd.concat([X1_train, X1_test])
         #删除常量
        df,dele_list = DataProcessor.constant_del(df, list(df.columns))
         #删除缺失值
        df,na_del = DataProcessor.del_na(df,list(df.columns),rate=0.995)
        nan_feature=['A1','A11','A9','A15']
         #缺失值布尔变量
        df=DataProcessor.create_missing_indicator(df, nan_feature)
         
         # 邮政编码
        df["A4"] =  df["A4"].astype(str)
        zipcode_city_map = {
         "110000": "北京市",
         "120000": "天津市",
         "130000": "河北省",
         "140000": "山西省",
         "150000": "内蒙古自治区",
         "210000": "辽宁省",
         "220000": "吉林省",
         "230000": "黑龙江省",
         "310000": "上海市",
         "320000": "江苏省",
         "330000": "浙江省",
         "340000": "安徽省",
         "350000": "福建省",
         "360000": "江西省",
         "370000": "山东省",
         "410000": "河南省",
         "420000": "湖北省",
         "430000": "湖南省",
         "440000": "广东省",
         "450000": "广西壮族自治区",
         "460000": "海南省",
         "510000": "四川省",
         "520000": "贵州省",
         "530000": "云南省",
         "540000": "西藏自治区",
         "500000": "重庆市",
         "610000": "陕西省",
         "620000": "甘肃省",
         "630000": "青海省",
         "640000": "宁夏回族自治区",
         "650000": "新疆维吾尔自治区",
         "810000": "香港特别行政区",
         "820000": "澳门特别行政区",
         "830000": "台湾省"
         }
        df["A4"] =  df["A4"].map(zipcode_city_map)
         # 创建是否发达列
        developed_cities = ["北京市", "上海市", "广东省", "江苏省", "浙江省", "天津市"]
        df["是否发达"] =  df["A4"].apply(lambda x: 1 if x in developed_cities else 0)
         # 创建是否落后列
        underdeveloped_cities = ["内蒙古自治区", "吉林省", "黑龙江省", "福建省", "江西省", "广西壮族自治区","云南省", "贵州省"]
        df["是否落后"] =  df["A4"].apply(lambda x: 1 if x in underdeveloped_cities else 0)
         #创建是否重点客源
        key_cities = ["四川省", "重庆市", "河北省"]
        df["是否重点"] =  df["A4"].apply(lambda x: 1 if x in underdeveloped_cities else 0)
         # 创建“北”列
        north_cities = ["北京市", "天津市", "河北省", "山西省", "内蒙古自治区", "辽宁省", "吉林省", "黑龙江省"]
        df["北"] =  df["A4"].apply(lambda x: 1 if x in north_cities else 0)
         # 创建“东”列
        east_cities = ["上海市", "江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省"]
        df["东"] =  df["A4"].apply(lambda x: 1 if x in east_cities else 0)
         # 创建“南”列
        south_cities = ["广东省", "广西壮族自治区", "海南省", "香港特别行政区", "澳门特别行政区", "台湾省", "湖南省", "湖北省", "河南省", "海南省", "云南省", "贵州省", "四川省", "重庆市", "西藏自治区"]
        df["南"] =  df["A4"].apply(lambda x: 1 if x in south_cities else 0)
      
         
        df["A17"] =  df["A17"].str[:-5].astype(float)
            
        #交叉组合
        obj_list=['A1','A3','A4','A7','A8','A12','A13','A14','A15','是否发达','是否落后','是否重点','北','东','南']
        cross,objcross_list=DataProcessor.Binary_Cross_Combination(obj_list, df, OneHot=False)

        df=pd.concat([df,cross],axis=1)

        df, le_encoded_cols = DataProcessor.label_encoder(df, None)
        
        

         #拆分数据
        mask = df['客户编号'] >= 15428
        X1_test = df[mask]
        X1_train = df[~mask]
        

        end_time = time.time()
        print("表X1处理成功，耗时：{:.2f}秒".format(end_time-start_time))

        return X1_train,X1_test 
    
     
     
    @staticmethod
    def X2_pipeline(X1_train,X1_test,X2_train_path,X2_test_path,if_pseudo):
        

        start_time = time.time()

        #合并数据
        X1 = pd.concat([X1_train,X1_test])
        X2_test = pd.read_csv(X2_test_path)
        X2_train = pd.read_csv(X2_train_path)
        
        df = pd.concat([X2_train, X2_test])
        df=df.drop(['B11','B12'], axis=1)
        df["B1"] =  df["B1"].str[:-5].astype(float)
        df["B8"] =  df["B8"].str[:-5].astype(float)
        df['day_delta']=df['B8']-df['B1']
        df['interest_delta2']=df["B2"]-df["B7"]
        df['interest_delta3']=df["B2"]-df["B9"]
        df['interest_delta11']=df["B5"]-df["B13"]
        df['interest_delta12']=df["B5"]-df["B14"]
        df['interest_delta13']=df["B5"]-df["B17"]
        df['interest_delta14']=df["B7"]-df["B9"]
        df['interest_delta26']=df["B13"]-df["B14"]
        df['interest_delta27']=df["B13"]-df["B17"]
        df['interest_delta28']=df["B14"]-df["B17"]

        #连续变量分组统计
        num_features = df.select_dtypes(include=['float']).columns.tolist()
        obj_features = ['B2','B3','B4','B5','B6','B7','B9','B13','B14','B15','B16','B17','B19']
        
        cross,objcross_list=DataProcessor.Binary_Cross_Combination(obj_features, df, OneHot=False)
        df=pd.concat([df,cross],axis=1)
        
        obj_features=obj_features+objcross_list
        
        new_feature_cat=[]
        new_feature_num=[]
        aggs_cat = ['count','nunique']
        aggs_num = ['mean', 'max', 'min', 'std', 'sum','median']
        X2,new_feature=DataProcessor.add_features(num_features , aggs_num, X1,new_feature_num, df)
        # X2 = round_feature(X2)
        X2,new_feature=DataProcessor.add_features(num_features+obj_features , aggs_cat, X2 ,new_feature_cat, df)

        # X = X2[new_feature].values.astype('float64')
        # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # X = imp.fit_transform(X)
        # svd = TruncatedSVD(n_components=10)
        # X_svd = svd.fit_transform(X)
        # X_scaled = StandardScaler().fit_transform(X_svd)
        # tsne = TSNE(n_components=2, init='pca', random_state=1001, perplexity=30, method='barnes_hut', n_iter=1000, verbose=1)
        # feats_tsne = tsne.fit_transform(X)
        # feats_tsne = pd.DataFrame(feats_tsne, columns=['tsne1_num', 'tsne2_num'])
        # feats_tsne['客户编号'] = pd.concat([X1_train[['客户编号']], X1_test[['客户编号']]], ignore_index=True)['客户编号'].values
        
        # X2= pd.merge(X2, feats_tsne, on='客户编号', how='left')
    
    
        #类型变量计数
        
        df = pd.get_dummies(df, columns=obj_features, dummy_na = True)
        agg_list = {     
                          'B17_0.7361137793828938':["sum"],
                          'B17_-1.474297574500177':["sum"],
                          'B17_-0.7740344837630655':["sum"],
                          'B17_-2.048498534265193':["sum"],
                          'B17_-0.6121216641451875':["sum"],
                          'B13_-0.2892409274290722':["sum"],
                            'B13_3.349408795592556':["sum"],
                            'B13_3.5790406753025112':["sum"],      
                            # 'B13_3.649746873208761':["sum"],
                            # 'B13_3.69148989047196':["sum"],                  
                            # 'B13_3.711520055887052':["sum"],
                            # 'B13_3.721382600447696':["sum"],
                            # 'B13_3.727546094945962':["sum"],
                            # 'B13_3.732017369377032':["sum"],
                            # 'B13_3.735335074072154':["sum"],
                            # 'B13_3.737165531834981':["sum"],
                            # 'B13_3.7382809670342025':["sum"],
                            # 'B13_3.7392009627327054':["sum"],
                            # 'B13_3.740130492065392':["sum"],
                            # 'B13_3.740912250068265':["sum"],
                            # 'B13_nan':["sum"],
                            'B14_-0.2892633736515787':["sum"],
                            'B14_3.3644994415617764':["sum"],
                            'B14_3.6202399561290433':["sum"],
                            'B14_3.694417386071009':["sum"],
                            'B14_3.7218477801510894':["sum"],
                            # 'B14_3.732669294673472':["sum"],
                            # 'B14_3.7377940207138063':["sum"],
                            # 'B14_3.7401490110988527':["sum"],
                            # 'B14_3.741002337574407':["sum"],
                            # 'B14_3.7412931359822768':["sum"],
                            # 'B14_3.741398014096592':["sum"],
                            # 'B14_3.741426617218676':["sum"],
                            # 'B14_nan':["sum"],
                         'B9_-0.7692734445435109':["sum"],
                         'B9_1.4452605199164534':["sum"],
                         'B9_-1.525624011833305':["sum"],
                         'B9_-1.7526805623126298':["sum"],
                         'B9_0.3026633680711922':["sum"],
                         'B9_-1.2111789054097188':["sum"],
                         'B9_-1.8115591631907937':["sum"],
                         'B7_-0.7692734445435109':["sum"],
                         'B7_1.4452605199164534':["sum"],
                         'B7_-1.525624011833305':["sum"],
                         'B7_-1.7526805623126298':["sum"],
                         'B7_0.3026633680711922':["sum"],
                         'B7_-1.2111789054097188':["sum"],
                         'B7_-1.8115591631907937':["sum"],
                         'B5_-0.2891659486921057':["sum"],
                         'B5_3.6010639353588254':["sum"],
                         'B5_3.5559624729107866':["sum"],
                         'B5_3.2366875808811164':["sum"],
                         'B5_3.6743156871856217':["sum"],
                         'B5_3.480551912706036':["sum"],
                         # 'B5_3.64213848151686':["sum"],
                         # 'B5_3.716090773810675':["sum"],
                         # 'B5_3.7290293273341777':["sum"],
                         # 'B5_3.3700905012433258':["sum"],
                         # 'B5_3.736940192471936':["sum"],
                         # 'B5_3.6976527393420087':["sum"],
                         # 'B5_3.7399425087591567':["sum"],        
                        'B2_-0.7692734445435109':["sum"],
                        'B2_1.4452605199164534':["sum"],
                        'B2_-1.525624011833305':["sum"],
                        'B2_-1.7526805623126298':["sum"],
                        'B2_0.3026633680711922':["sum"],
                        'B2_-1.2111789054097188':["sum"],
                        'B2_-1.8115591631907937':["sum"],
                        'B3_A':["sum"],
                        'B3_B':["sum"],
                        'B3_C':["sum"],
                        'B3_D':["sum"],
                        'B3_E':["sum"],
                        'B3_F':["sum"],
                        'B3_G':["sum"],
                        'B3_nan':["sum"],
                        'B4_t1':["sum"],
                        'B4_t2':["sum"],
                        'B4_nan':["sum"],
                        'B6_M1':["sum"],
                        'B6_M2':["sum"],
                        'B6_nan':["sum"],
                        'B15_A1':["sum"],
                        'B15_A2':["sum"],
                        'B15_nan':["sum"],
                        'B16_X1':["sum"],
                        'B16_X2':["sum"],
                        'B16_X3':["sum"],
                        'B16_X4':["sum"],
                        'B16_nan':["sum"],
                        'B19_X1':["sum"],
                        'B19_X2':["sum"],
                        'B19_X3':["sum"],
                        'B19_X4':["sum"],
                        'B19_nan':["sum"]}
     
        temp= df.groupby('客户编号').agg(agg_list)
        temp.columns = pd.Index([col[0] + "_" + col[1].upper() for col in temp.columns.tolist()])
        cat_feature=temp.columns.tolist()
        X2 = X2.merge(temp, on='客户编号', how='left')

        def convert_decimal_to_int(df):
            for column in tqdm(df.columns):
                if df[column].dtype == 'float64':
                    decimal_places = df[column].apply(lambda x: str(x).split('.')[-1])
                    if all(decimal_places.isin(['0', 'nan'])):
                        df[column] = df[column].fillna(0).astype('int64')

            return df


        X2 = convert_decimal_to_int(X2)


        # X = X2[cat_feature].values.astype('float64')
        # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # X = imp.fit_transform(X)
        # svd = TruncatedSVD(n_components=10)
        # X_svd = svd.fit_transform(X)
        # X_scaled = StandardScaler().fit_transform(X_svd)
        # tsne = TSNE(n_components=2, init='pca', random_state=1001, perplexity=30, method='barnes_hut', n_iter=1000, verbose=1)
        # feats_tsne = tsne.fit_transform(X)
        # feats_tsne = pd.DataFrame(feats_tsne, columns=['tsne1_cat', 'tsne2_cat'])
        # feats_tsne['客户编号'] = pd.concat([X1_train[['客户编号']], X1_test[['客户编号']]], ignore_index=True)['客户编号'].values
        # X2= pd.merge(X2, feats_tsne, on='客户编号', how='left')
        
        mask = X2['客户编号'] >= 15428
        X2_test = X2[mask]
        X2_train = X2[~mask]
    
        
        end_time = time.time()
        print("表X2处理成功，耗时：{:.2f}秒".format(end_time-start_time))


        return X2_train,X2_test
     
        
    @staticmethod
    def X3_pipeline(X2_train,X2_test,X3_train_path,X3_test_path,if_pseudo):

        start_time = time.time()


        #合并数据
        X2 = pd.concat([X2_train,X2_test])
        X3_test = pd.read_csv(X3_test_path)
        X3_train = pd.read_csv(X3_train_path)
        
            
        X3 = pd.concat([X3_train,X3_test])

 
        #拖欠次数
        X3['C1_4']=X3["C1"]/X3["C4"]
        X3['C1_5']=X3["C1"]/X3["C5"]
        X3['C1_48']=X3["C1"]/X3["C48"]
        X3['C1_49']=X3["C1"]/X3["C49"]
        X3['C4_5']=X3["C4"]/X3["C5"]
        X3['C4_48']=X3["C4"]/X3["C48"]
        X3['C4_49']=X3["C4"]/X3["C49"]
        X3['C5_48']=X3["C5"]/X3["C48"]
        X3['C5_49']=X3["C5"]/X3["C49"]

        #拖欠货款较长时间次数
        X3['C9_11']=X3["C9"]/X3["C11"]
        X3['C9_17']=X3["C9"]/X3["C17"]
        X3['C11_17']=X3["C11"]/X3["C17"]

        #货单金额
        X3['C2_6']=X3["C2"]/X3["C6"]
        X3['C2_7']=X3["C2"]/X3["C7"]
        X3['C2_8']=X3["C2"]/X3["C8"]
        X3['C2_30']=X3["C2"]/X3["C30"]
        X3['C6_7']=X3["C6"]/X3["C7"]
        X3['C6_8']=X3["C6"]/X3["C8"]
        X3['C6_30']=X3["C4"]/X3["C30"]
        X3['C7_8']=X3["C7"]/X3["C8"]
        X3['C7_30']=X3["C7"]/X3["C30"]
        X3['C8_30']=X3["C8"]/X3["C30"]

        X3.replace([np.inf, -np.inf], np.nan, inplace=True)

        
        # X3_train= round_feature(X3_train)
        # X3_test = round_feature(X3_test)
    
        # flist = [x for x in X3_train.columns if not x in ['客户编号']]
    
        # TSNE 降维特征
    
        # X = pd.concat([X3_train[flist], X3_test[flist]], ignore_index=True).values.astype('float64')
        # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # X = imp.fit_transform(X)
        # svd = TruncatedSVD(n_components=10)
        # X_svd = svd.fit_transform(X)
        # X_scaled = StandardScaler().fit_transform(X_svd)
        # tsne = TSNE(n_components=2, init='pca', random_state=1001, perplexity=30, method='barnes_hut', n_iter=1000, verbose=1)
        # feats_tsne = tsne.fit_transform(X)
        # feats_tsne = pd.DataFrame(feats_tsne, columns=['tsne1', 'tsne2'])
        # feats_tsne['客户编号'] = pd.concat([X3_train[['客户编号']], X3_test[['客户编号']]], ignore_index=True)['客户编号'].values
        # X3_train = pd.merge(X3_train, feats_tsne, on='客户编号', how='left')
        # X3_test = pd.merge(X3_test, feats_tsne, on='客户编号', how='left')



#         X3 = pd.merge(X2,X3, on='客户编号', how='left')

#         X3_count = count_feature(X3)
    
#         X3 = X3_count.merge(X3, on='客户编号', how='left')

  
         #删除常量
        X3,dele_list = DataProcessor.constant_del(X3, list(X3.columns))
        #删除缺失值
        X3,na_del = DataProcessor.del_na(X3,list(X3.columns),rate=0.99)
    
        mask = X3['客户编号'] >= 15428
        X3_test = X3[mask]
        X3_train = X3[~mask]
        
        end_time = time.time()
        print("表X3处理成功，耗时：{:.2f}秒".format(end_time-start_time))


        return X3_train,X3_test