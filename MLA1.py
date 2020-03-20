import numpy as np
import pandas as pd
import time
import datetime
#-*- coding : utf-8-*-
# coding:unicode_escape

##股票代码列表
stock_df = pd.read_csv("F:/我的大学/港中深第一学期/黑暗森林-219040104/300.csv", header=None)
stock_list = stock_df[0].tolist()
print("The stock selected are from CSI300:")
print(stock_list)
print("")

##原始因子列表
stock_test = pd.read_excel("F:/我的大学/港中深第一学期/黑暗森林-219040104/code/Data weekly/000002.SZ.xlsx", sheet_name=1, header=3, index_col=None)
FactorList = list(stock_test.columns)

##周线日期列表
stock_test.index=stock_test['Date']
DateList = list((stock_test.index).strftime("%Y-%m-%d"))

print("There are 36 factors in our model, including valuation factors,financial factors,market factors and technical factors:")
print(FactorList)
#len(stock_list)
#type(DateList[5])

for s in stock_list:
    df_weekly = pd.read_excel(str("F:/我的大学/港中深第一学期/黑暗森林-219040104/code/Data weekly/"+s+".xlsx"), sheet_name=1, header=3,index_col= None)
    df_weekly['Date'] = df_weekly['Date'].dt.strftime("%Y-%m-%d")
    df_monthly = pd.read_csv(str("F:/我的大学/港中深第一学期/黑暗森林-219040104/code/Data monthly/"+s+".csv"), header=3, index_col=None, encoding='gbk')
    df_monthly = df_monthly.dropna(axis=0, how='any', subset=['yoy_or','yoyprofit'])
    df = pd.merge(df_weekly,df_monthly, how='outer')
    df = df.sort_values(by = ['Date'])
    df = df.fillna(method = 'ffill')
    df = df.dropna(axis = 0 , subset = ['MACD'])
    df['ln_close'] = np.log(df['close'])
    df['ln_ev'] = np.log(df['ev'])
    df['logreturn_b'] = df['ln_close'].diff()
    df['logreturn_f'] = df['logreturn_b'].shift(periods=-1, fill_value=0)
    df['y']=df['logreturn_f'].apply(lambda x:1 if x>0 else 0)
    df = df.drop(['ev', 'close', 'underlyinghisvol_30d'], axis=1)
    df['Code'] = s
    df['Code'] = df['Code'].fillna(method = 'ffill')
    df.to_csv(str("F:/我的大学/港中深第一学期/黑暗森林-219040104/code/Data Pre-process/"+s+".csv"), index=None)

# 按股票分开，存在字典里，免去后续的读取操作
Factor_by_stock={}
for s in stock_list:
    df = pd.read_csv(str("F:/我的大学/港中深第一学期/黑暗森林-219040104/code/Data Pre-process/"+s+".csv"), encoding='utf_8')
    Factor_by_stock[s]=df
    DateList = Factor_by_stock[s]['Date']
    Factor_by_stock[s].index=DateList

#按时间划分数据，形成预处理前最终数据
Factor_by_time = {}
DateList = Factor_by_stock['000001.SZ']['Date']
for date in DateList:
    for s in stock_list:
        if s == '000001.SZ':
            if date >= Factor_by_stock[s].index[0]:
                Factor_by_time[date] = Factor_by_stock[s].loc[[date]]
        else:
            if date >= Factor_by_stock[s].index[0]:

                Factor_by_time[date] = pd.concat([Factor_by_time[date] , Factor_by_stock[s].loc[[date]]], axis=0, sort=False)
    #去掉Date
    Factor_by_time[date] = Factor_by_time[date].fillna(0)
    Factor_by_time[date] = Factor_by_time[date].dropna(how='all', axis=1)
    Factor_by_time[date].to_csv(str("F:/我的大学/港中深第一学期/黑暗森林-219040104/code/Data Final/"+date+".csv"), index=None)

#-- 设置样本
TrainList = DateList[:int(0.7*len(DateList))]
TestList = DateList[int(0.7*len(DateList)):]
for date in TrainList:
    #--load csv
    data_curr_day = Factor_by_time[date]
    # --merge
    if date == TrainList.iloc[0]:# --first day
        Factor_in_sample = data_curr_day
    else:
        Factor_in_sample = Factor_in_sample.append(data_curr_day,sort = False)
    Factor_train = Factor_in_sample
for date in TestList:
    data_curr_day2 = Factor_by_time[date]
    data_curr_day2['Date'] = date
    if date == TestList.iloc[0]:# --first month
        Factor_in_sample2 = data_curr_day2
    else:
        Factor_in_sample2 = Factor_in_sample2.append(data_curr_day2)
    Factor_test = Factor_in_sample2

train_method = 'Reg'
#将样本内集合切分为训练集和测试集
if train_method == 'Clf':
    #设置训练集和验证集
    X_train = Factor_train.loc[:,'pe_ttm':'open']#第一列为pe_ttm因子，最后一列为open因子
    y_train = Factor_train.loc[:,'y']#标签y
    X_test = Factor_test.loc[:,'pe_ttm':'open']
    y_test = Factor_test.loc[:,'ln_close':'Code']
    #test_size:交叉验证集占样本内集合的比例
    #seed:随机数种子，希望每次切分的结果完全相同，从而保证实验结果可重复，如无需回溯对比，可不设置
if train_method == 'Reg':
    #设置训练集和验证集
    X_train = Factor_train.loc[:,'pe_ttm':'open']#第一列为pe_ttm因子，最后一列为open因子
    y_train = Factor_train.loc[:,'logreturn_f']#取logreturn_f作为标签y
    X_test = Factor_test.loc[:,'pe_ttm':'open']
    y_test = Factor_test.loc[:,'ln_close':'Code']
    #test_size:交叉验证集占样本内集合的比例
    #seed:随机数种子，希望每次切分的结果完全相同，从而保证实验结果可重复，如无需回溯对比，可不设置
premethod = 'Scaling to unit length'
from sklearn import preprocessing
if premethod == 'Standardization':
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test_temp = scaler.transform(X_test)
if premethod == 'Rescaling':
    scaler = preprocessing.MinMaxScalerScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test_temp = scaler.transform(X_test)
if premethod == 'Scaling to unit length':
    scaler = preprocessing.MaxAbsScaler(X_train)
    X_train = scaler.transform(X_train)
    X_test_temp = scaler.transform(X_test)
method = 'LinearReg'
if method == 'LinearReg':
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit (X_train,y_train)
    y_pred_test=reg.predict(X_test_temp)
if method == 'Lasso':
    from sklearn.linear_model import Lasso
    from sklearn import metrics
    reg = Lasso()
    reg.fit(X_train, y_train)
    y_pred_test=reg.predict(X_test_temp)
if method == 'Ridge':
    from sklearn.linear_model import Ridge
    from sklearn import metrics
    reg = Ridge()
    reg.fit(X_train, y_train)
    y_pred_test=reg.predict(X_test_temp)
if method == 'LR':
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred_test=clf.predict(X_test_temp)
if method == 'SVM':
    import matplotlib.pyplot as plt
    from sklearn import metrics, model_selection, svm
    clf = svm.SVR()
    clf.fit(X_train, y_train)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
num_folds = 10
seed = 42
scoring = 'roc_auc'
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
if train_method == 'Reg':
    results1 = cross_val_score(reg, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    print("The negative mean squared error is ")
    print(results1)
    results2 = cross_val_score(reg, X_train, y_train, cv=kfold, scoring='r2')
    print("The R2 is ")
    print(results2)
if train_method == 'Clf':
    results = cross_val_score(clf, X_train, y_train, cv=kfold, scoring=scoring)