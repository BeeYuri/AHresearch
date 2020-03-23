import pandas as pd
import numpy as np
import datetime

##股票代码列表
stock_df = pd.read_csv("F:/AHresearch/AHdata tech factor/code1.csv", header=None, encoding='unicode_escape')
stock_list = stock_df[0][1:].tolist()

##原始因子列表
stock_test = pd.read_csv("F:/AHresearch/AHdata tech factor/0038.HK.csv", header=3, index_col = None, encoding='unicode_escape')
FactorList = list(stock_test.columns)

#  日线日期列表
stock_test.index=stock_test['Date']
DateList = stock_test['Date']

#因子列表，设置columns原始顺序
order = ['Date', 'close', 'premiumrate_ah','amt', 'turn', 'pe_ttm', 'pb_mrq', 'ps_ttm',
       'pcf_ocf_ttm', 'dividendyield2', 'ev', 'MA', 'MA.1', 'MA.2', 'open',
       'high', 'volume', 'low', 'pct_chg', 'RSI', 'MACD', 'MACD.1', 'Date',
       'ADTM', 'ATR', 'BBI', 'BBIBOLL', 'BIAS', 'BOLL', 'CCI', 'CDP', 'DMA',
       'DMI_2', 'DPO', 'ENV', 'EXPMA', 'KDJ', 'slowKD', 'MIKE', 'MTM',
       'PRICEOSC', 'PVT', 'RC', 'ROC', 'SAR', 'SI', 'SOBV', 'SRMI', 'STD',
       'TAPI', 'TRIX', 'VHF', 'VMA', 'VMACD', 'VOSC', 'VSTD', 'WVAD',
       'vol_ratio', 'Code', 'HSclose', 'HSrate', 'sign']

#设置AH溢价为昨日收盘/今日开盘
for s in stock_list:
    df = pd.read_excel(str("F:/AHresearch/AHdata/"+s+".xlsx"), header=3) #第一部分因子
    df2 = pd.read_csv(str("F:/AHresearch/AHdata tech factor/"+s+".csv"), header=3) #第二部分因子
    df2['Date'] = pd.to_datetime( df2['Date'],errors = 'coerce')
    df.index = df['Date']
    df2.index = df2['Date']
    df = df.drop(['BIAS'],axis = 1)
    df2 = df2.drop(['2'],axis = 1)
    df = pd.concat([df,df2], axis = 1,join='outer', join_axes=[df.index]) #两部分因子拼合
    df = df.dropna(axis = 0 , subset = ['premiumrate_ah','close','MA.2','MACD.1','amt'])
    df = df.fillna(0)
    df['Code'] = s
    df['Code'] = df['Code'].fillna(method = 'ffill')
    df['HSclose'] = (df['premiumrate_ah']/100+1)*df['close'] #利用原始AH溢价计算A股收盘价
    df['HSrate'] = df['HSclose']/df['HSclose'].shift(1) - 1 #计算A股收益率
    df['sign']=df['HSrate'].apply(lambda x:1 if x>0 else 0) #设置标签，涨为1，跌为0
    df = df.drop(['premiumrate_ah'],axis = 1)
    df['premiumrate_ah'] = df['HSclose'].shift(1)/df['open'] #设置新的AH溢价：昨日A股收盘/今日H股开盘
    df = df[order]
    df = df.drop(['close'],axis = 1)
    df.to_csv(str("F:/AHresearch/Data Pre-process2/"+s+".csv"), index=None) #输出数据

#以股票代码为索引，合成大字典，删去重复列
Factor_by_stock={}
for s in stock_list:
    df = pd.read_csv(str("F:/AHresearch/Data Pre-process2/"+s+".csv"))
    df.drop([len(df)-2],inplace=True)
    DateList = df['Date']
    Factor_by_stock[s]=df
    Factor_by_stock[s].index=DateList
    Factor_by_stock[s] = Factor_by_stock[s].drop(['Date.1','Date.2','Date.3','Code'],axis = 1)
FactorList = Factor_by_stock[s].columns
#设计多级索引:日期部分
DateList = Factor_by_stock['0042.HK']['Date'][1:]
i = 1
DateList_new = pd.DataFrame([np.nan]*np.ones(len(stock_list)))
DateList_new[:] = DateList[0]
while i <= (len(DateList)-1):
    data_temp = pd.DataFrame([np.nan]*np.ones(len(stock_list)))
    data_temp[:] = DateList[i]
    DateList_new = DateList_new.append(data_temp)
    i = i+1

#设计多级索引：股票代码部分
i = 1
stock_df_new = stock_df[0][1:]
stock_list_new = stock_df[0][1:]
while i <= (len(DateList)-1):
    data_temp = stock_df[0][1:]
    stock_list_new = stock_list_new.append(data_temp)
    i = i+1

DateList2 = DateList_new[0].tolist()
stock_list2 = stock_list_new.tolist()

#创建一个包含多级索引的datafame
arrays = [DateList2,stock_list2]
Factor = pd.DataFrame([np.nan]*np.ones((len(DateList2),len(FactorList))),columns = FactorList,index=pd.MultiIndex.from_arrays(arrays,names=('Date', 'Code')))
for s in stock_list:
    Factor_by_stock[s] = Factor_by_stock[s].drop(['Date'],axis = 1)
for s in stock_list:
    # 监测合并进度
    # print(s)
    for date in DateList:
        if date >= Factor_by_stock[s].index[0]:
            Factor.loc[date].loc[s] = Factor_by_stock[s].loc[date]

#设置参数
seed = 42

def normalization(premethod,X_in_sample):
    from sklearn import preprocessing
    if premethod == 'Standardization':
        scaler = preprocessing.StandardScaler().fit(X_in_sample)
    if premethod == 'Rescaling':
        scaler = preprocessing.MinMaxScaler().fit(X_in_sample)
    if premethod == 'Scaling to unit length':
        scaler = preprocessing.MaxAbsScaler().fit(X_in_sample)
    return scaler

#循环训练
import time
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import roc_auc_score
n_stock_select = 10
strategy = pd.DataFrame()
AUC_train = pd.DataFrame()
AUC_test = pd.DataFrame()
time_train = 300
time_test = 10
premethod = 'Standardization'
n_stock_select = 10#设置选股个数为10
for initial_time in range(0,(len(DateList)-(time_train+time_test)),time_test):
    TrainList = DateList[initial_time:initial_time+time_train]
    TestList = DateList[initial_time+time_train:initial_time+time_train+time_test]
    Factor_train = Factor.iloc[initial_time*len(stock_list):(initial_time+time_train)*len(stock_list),:]
    Factor_test =  Factor.iloc[(initial_time+time_train)*len(stock_list):(initial_time+time_train+time_test)*len(stock_list),:]
    Factor_train = Factor_train.drop(['Date'],axis = 1)
    Factor_train = Factor_train.dropna(axis = 0)
    Factor_test = Factor_test.drop(['Date'],axis = 1)
    Factor_test = Factor_test.dropna(axis = 0)
    X_in_sample = Factor.loc[:,'premiumrate_ah':'vol_ratio']
    X_in_sample = X_in_sample.dropna(axis = 0)
    scaler = normalization(premethod,X_in_sample)
    X_train = Factor_train.loc[:,'premiumrate_ah':'vol_ratio']
    y_train = Factor_train.loc[:,'sign']
    X_test = Factor_test.loc[:,'premiumrate_ah':'vol_ratio']
    y_test = Factor_test.loc[:,'HSclose':'sign']
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf=tree.DecisionTreeClassifier()
    clf=clf.fit(X_train,y_train)
    importance =clf.feature_importances_.round(4).tolist()
    y_pred_train=clf.predict_proba(X_train)[:,1]
    AUC_train_curr = pd.DataFrame({'AUC_train':[roc_auc_score(y_train, y_pred_train)]})
    AUC_train =  AUC_train.append(AUC_train_curr )
    y_pred_test=clf.predict_proba(X_test)[:,1]
    AUC_test_curr = pd.DataFrame({'AUC_test':[roc_auc_score(y_test['sign'], y_pred_test)]})
    AUC_test =  AUC_test.append(AUC_test_curr )
    y_pred_test=pd.DataFrame({'result':y_pred_test})
    y_pred_test.index = y_test.index
    #初始化策略的收益和净值。变量strategy 是DataFrame 类型，有两列'return'和'value'，分别表示策略的每月收益和每月净值。'return'列的初始值均为0，'value'列的初始值均为1。
    #--loop for days
    strategy_temp = pd.DataFrame({'return':[0]*(time_test),'value':[1]*(time_test)},index = TestList)
    for date in TestList:
        y_true_curr_day = y_test.loc[date]
        y_pred_curr_day= y_pred_test.loc[date]
        #y_true_curr_month 为当前日的真实收益，y_score_curr_month 为当前日的预测的决策函数值，均为只包含一列的DataFrame 类型。其中，函数iloc 是按索引获取切片，列为i_month-1 是因为列的索引从0 开始。
        #-- sort predicted return,and choose the best 10
        y_pred_curr_day = y_pred_curr_day.sort_values(by = 'result',ascending = False)
        index_select = y_pred_curr_day.iloc[0:n_stock_select].index
    #         if i_day == 45:
    #             y_true_temp = y_true_curr_day.sort_values(ascending = False)
    #             print(y_true_temp)
    #             print(index_select)
        #-- take the average return as the return of the portfolio
        strategy_temp.loc[date,'return'] = np.mean(y_true_curr_day['HSrate'][index_select])
        strategy_temp = strategy_temp.fillna(0)
    strategy = strategy.append(strategy_temp)
strategy['value'] = (strategy['return']+1).cumprod()

#画出266次训练的AUC图
import matplotlib.pyplot as plt
# AUC_train.reset_index(drop = True)
# AUC_test.reset_index(drop = True)
# plt.plot(range(266),AUC_train, label='AUC_train')
# plt.plot(range(266),AUC_test, label='AUC_test')
# plt.legend()
# plt.show()

#输出因子重要性
Factor_importance = pd.DataFrame({'factor':FactorList[1:-3],'importance':importance})
print(Factor_importance)

#确定最终回测区间
FinalTestList = DateList['2009-03-26':'2020-01-07']
FinalTestList = pd.to_datetime( FinalTestList ,errors = 'coerce')
strategy.index = FinalTestList

#计算胜率
def WinRate(return_list,basereturn_list):
    excess_return = return_list - basereturn_list
    return np.sum([excess_return>0])  / len(excess_return)

# 对上证指数切片
SH000001_df = pd.read_excel("C:/Users/Administrator/Desktop/000001_SH.xlsx", header=3)
SH000001_df['Date'] = pd.to_datetime( SH000001_df['Date'],errors = 'coerce')
SH000001_df['pct_chg'] = SH000001_df['pct_chg']/100
SH000001_df.index = SH000001_df['Date']
SH000001_df = SH000001_df.drop(['Date'], axis = 1)
SH000001_df = SH000001_df.reindex(FinalTestList)
strategy['000001_SH_rate'] = SH000001_df
strategy['000001_SH_rate'] = strategy['000001_SH_rate'].fillna(0)
strategy['000001_SH_value'] = (strategy['000001_SH_rate']+1).cumprod()

# 对沪深300指数切片
SH000300_df = pd.read_excel("C:/Users/Administrator/Desktop/000300_SH.xlsx")
SH000300_df['Date'] = pd.to_datetime( SH000300_df['Date'],errors = 'coerce')
SH000300_df.index = SH000300_df['Date']
SH000300_df = SH000300_df.drop(['Date'], axis = 1)
SH000300_df = SH000300_df.reindex(FinalTestList)
strategy['000300_SH_rate'] = SH000300_df
strategy['000300_SH_rate'] = strategy['000300_SH_rate'].fillna(0)
strategy['000300_SH_value'] = (strategy['000300_SH_rate']+1).cumprod()

# 对中证500指数切片
SH000905_df = pd.read_excel("C:/Users/Administrator/Desktop/000905_SH.xlsx")
SH000905_df['Date'] = pd.to_datetime( SH000905_df['Date'],errors = 'coerce')
SH000905_df.index = SH000905_df['Date']
SH000905_df = SH000905_df.drop(['Date'], axis = 1)
SH000905_df = SH000905_df.reindex(FinalTestList)
strategy['000905_SH_rate'] = SH000905_df
strategy['000905_SH_rate'] = strategy['000905_SH_rate'].fillna(0)
strategy['000905_SH_value'] = (strategy['000905_SH_rate']+1).cumprod()

#对创业板指数切片
SZ399006_df = pd.read_excel("C:/Users/Administrator/Desktop/399006_SZ.xlsx")
SZ399006_df['Date'] = pd.to_datetime( SZ399006_df['Date'],errors = 'coerce')
strategy1 = strategy.reindex(SZ399006_df['Date'])
SZ399006_df.index = SZ399006_df['Date']
SZ399006_df =SZ399006_df.drop(['Date'], axis = 1)
strategy1['399006_SZ_rate'] = SZ399006_df
strategy1['399006_SZ_rate'] = strategy1['399006_SZ_rate'].fillna(0)
strategy1['399006_SZ_value'] = (strategy1['399006_SZ_rate']+1).cumprod()

#输出胜率
print("对上证指数胜率为 %2f."% WinRate(strategy['return'],strategy['000001_SH_rate']))
print("对沪深300指数胜率为 %2f."% WinRate(strategy['return'],strategy['000300_SH_rate']))
print("对中证500指数胜率为 %2f."% WinRate(strategy['return'],strategy['000905_SH_rate']))
print("对创业板指数胜率为 %2f."% WinRate(strategy1['return'],strategy1['399006_SZ_rate']))

#-- plot the value
#-- plot the value
plt.plot(FinalTestList,strategy.loc[FinalTestList,'value'], label='strategy')
plt.plot(FinalTestList,strategy.loc[FinalTestList,'000001_SH_value'], label='000001.SH')
plt.plot(FinalTestList,strategy.loc[FinalTestList,'000300_SH_value'], label='000300.SH')
plt.plot(FinalTestList,strategy.loc[FinalTestList,'000905_SH_value'], label='000905.SH')
plt.plot(FinalTestList,strategy1.loc[FinalTestList,'399006_SZ_value'],label = '399006.SZ')
plt.legend()
plt.show()
#年化收益率
def ann_return(return_list):
    ar = pow((return_list+1).cumprod()[-1],1/5)-1
    return ar
#年化超额收益率
def ann_excess_return(return_list,basereturn_list):
    excess_return = return_list - basereturn_list
    aer  = pow((excess_return+1).cumprod()[-1],1/5)-1
    return aer
#信息比率
def IR(return_list,basereturn_list):
    excess_return = return_list - basereturn_list #计算每期超额收益
    #计算股票及指数年化收益率
    aer  = pow((excess_return+1).cumprod()[-1],1/5)-1
    aer_std = np.std(excess_return)*np.sqrt(365)#计算每期超额收益标准差
    return aer/aer_std
#夏普比率
def SharpeRatio(return_list,basereturn_list):
    excess_return = return_list - basereturn_list #计算每期超额收益
    aer  = pow((excess_return+1).cumprod()[-1],1/5)-1
    return_std = np.std(return_list)*np.sqrt(365)
    return aer / return_std
#最大回撤
def MaxDrawdown(return_list):
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    mdd = ((return_list[j]) - (return_list[i])) / (return_list[j])
    return mdd
# 最大回撤时间及持续长度
def MddHappen(return_list):
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    print('The MaxDrawdown happens at ')
    print(j)
    print('The MaxDrawdown keeps for ')
    print(i - j)
    return 0
#calmar比率
def Calmar(aer,mdd):
    return aer / mdd
#输出结果
print('annual return = %.3f'%ann_return(strategy['return']))
print('annual excess return = %.3f'%ann_excess_return(strategy['return'],strategy['000001_SH_rate']))
print('information ratio = %.3f'%IR(strategy['return'],strategy['000001_SH_rate']))
print('SharpeRatio = %.3f'%SharpeRatio(strategy['return'],strategy['000001_SH_rate']))
print('MaxDrawdown = %.3f'%MaxDrawdown(strategy['value']))
MddHappen(strategy['value'])
print('Calmar = %.3f'%Calmar(ann_excess_return(strategy['return'],strategy['000001_SH_rate']),MaxDrawdown(strategy['value'])))




