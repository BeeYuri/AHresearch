import pandas as pd
import numpy as np
import Model
class utils:
    def __init__(self):
        self
    def parameter(self,n_stock_select,seed):
        self.n_stock_select = n_stock_select
        self.seed = seed
    def struct_strategy(self):
        self.data_new = Model.Model.data.dropna(subset = ['y_pred'])
        self.Date_bt = self.data_new["Date"].unique()
        self.strategy = pd.DataFrame({'return': [0] * (len(self.Date_bt)), 'value': [1] * (len(self.Date_bt))},
                                     index=self.Date_bt)
        for date in self.Date_bt:
            self.data_curr_day = self.data_new[self.data_new['Date'] == date].sort_values(by="y_pred")
            self.index_select = self.data_curr_day.iloc[0:self.n_stock_select].index
            self.strategy.loc[date, 'return'] = np.mean(self.data_curr_day['HSrate'][self.index_select])
        self.strategy['value'] = (self.strategy['return'] + 1).cumprod()
        self.strategy = self.strategy.dropna(subset=['return'])
    def merging_index(self):
        # 上证指数
        self.SH000001_df = pd.read_excel("C:/Users/Administrator/Desktop/000001_SH.xlsx", header=3)
        self.SH000001_df['Date'] = pd.to_datetime(self.SH000001_df['Date'], errors='coerce')
        self.SH000001_df['pct_chg'] = self.SH000001_df['pct_chg'] / 100
        self.SH000001_df.index = self.SH000001_df['Date']
        self.SH000001_df = self.SH000001_df.drop(['Date'], axis=1)
        self.SH000001_df = self.SH000001_df.reindex(self.Date_bt)
        self.strategy['000001_SH_rate'] = self.SH000001_df
        self.strategy['000001_SH_rate'] = self.strategy['000001_SH_rate'].fillna(0)
        self.strategy['000001_SH_value'] = (self.strategy['000001_SH_rate'] + 1).cumprod()
        # 沪深300
        self.SH000300_df = pd.read_excel("C:/Users/Administrator/Desktop/000300_SH.xlsx")
        self.SH000300_df['Date'] = pd.to_datetime(self.SH000300_df['Date'], errors='coerce')
        self.SH000300_df.index = self.SH000300_df['Date']
        self.SH000300_df = self.SH000300_df.drop(['Date'], axis=1)
        self.SH000300_df = self.SH000300_df.reindex(self.Date_bt)
        self.strategy['000300_SH_rate'] = self.SH000300_df
        self.strategy['000300_SH_rate'] = self.strategy['000300_SH_rate'].fillna(0)
        self.strategy['000300_SH_value'] = (self.strategy['000300_SH_rate'] + 1).cumprod()
        # 中证500
        self.SH000905_df = pd.read_excel("C:/Users/Administrator/Desktop/000905_SH.xlsx")
        self.SH000905_df['Date'] = pd.to_datetime(self.SH000905_df['Date'], errors='coerce')
        self.SH000905_df.index = self.SH000905_df['Date']
        self.SH000905_df = self.SH000905_df.drop(['Date'], axis=1)
        self.SH000905_df = self.SH000905_df.reindex(self.Date_bt)
        self.strategy['000905_SH_rate'] = self.SH000905_df
        self.strategy['000905_SH_rate'] = self.strategy['000905_SH_rate'].fillna(0)
        self.strategy['000905_SH_value'] = (self.strategy['000905_SH_rate'] + 1).cumprod()
        # 创业板指
        self.SZ399006_df = pd.read_excel("C:/Users/Administrator/Desktop/399006_SZ.xlsx")
        self.SZ399006_df['Date'] = pd.to_datetime(self.SZ399006_df['Date'], errors='coerce')
        self.strategy1 = self.strategy.reindex(self.SZ399006_df['Date'])
        self.SZ399006_df.index = self.SZ399006_df['Date']
        self.SZ399006_df = self.SZ399006_df.drop(['Date'], axis=1)
        self.SZ399006_df = self.SZ399006_df.loc[self.Date_bt]
        self.strategy1['399006_SZ_rate'] = self.SZ399006_df
        self.strategy1['399006_SZ_rate'] = self.strategy1['399006_SZ_rate'].fillna(0)
        self.strategy1['399006_SZ_value'] = (self.strategy1['399006_SZ_rate'] + 1).cumprod()
    def print_winrate(self):
        def WinRate(return_list, basereturn_list):
            excess_return = return_list - basereturn_list
            return np.sum([excess_return > 0]) / len(excess_return)

        def absolute_winrate(return_list):
            positive = return_list[return_list > 0].count()
            return positive / len(return_list)

        print("对上证指数胜率为 %2f." % WinRate(self.strategy['return'], self.strategy['000001_SH_rate']))
        print("对沪深300指数胜率为 %2f." % WinRate(self.strategy['return'], self.strategy['000300_SH_rate']))
        print("对中证500指数胜率为 %2f." % WinRate(self.strategy['return'], self.strategy['000905_SH_rate']))
        print("对创业板指数胜率为 %2f." % WinRate(self.strategy1['return'], self.strategy1['399006_SZ_rate']))
        print("绝对胜率为 %2f." % absolute_winrate(self.strategy['return']))

    def plot_value(self):
        import matplotlib.pyplot as plt
        plt.plot(self.Date_bt, self.strategy.loc[self.Date_bt, 'value'], label='strategy')
        plt.plot(self.Date_bt, self.strategy.loc[self.Date_bt, '000001_SH_value'], label='000001.SH')
        plt.plot(self.Date_bt, self.strategy.loc[self.Date_bt, '000300_SH_value'], label='000300.SH')
        plt.plot(self.Date_bt, self.strategy.loc[self.Date_bt, '000905_SH_value'], label='000905.SH')
        plt.plot(self.Date_bt, self.strategy1.loc[self.Date_bt, '399006_SZ_value'], label='399006_SZ')
        plt.legend()
        plt.show()

    def print_ann_exces_return(self):
        def ann_excess_return(return_list, basereturn_list):
            excess_return = return_list - basereturn_list
            aer = pow((excess_return + 1).cumprod()[-1], 365 / len(self.Date_bt)) - 1
            return aer
        self.excess_return = self.strategy['return'] - self.strategy['000001_SH_rate']
        self.aer = ann_excess_return(self.strategy['return'], self.strategy['000001_SH_rate'])
        print('annual excess return = %.3f' % self.aer)

    def print_ann_return(self):
        def ann_return(return_list):
            ar = pow((return_list + 1).cumprod()[-1], 365 / len(self.Date_bt)) - 1
            return ar

        print('annual return = %.3f' % ann_return(self.strategy['return']))

    def print_IR(self):
        def IR(return_list, basereturn_list):
            excess_return = return_list - basereturn_list  # 计算每期超额收益
            # 计算股票及指数年化收益率
            aer = pow((excess_return + 1).cumprod()[-1], 365 / len(self.Date_bt)) - 1
            aer_std = np.std(excess_return) * np.sqrt(365)  # 计算每期超额收益标准差
            return aer / aer_std

        print('information ratio = %.3f' % IR(self.strategy['return'], self.strategy['000001_SH_rate']))

    def print_Sharpe_ratio(self):
        def SharpeRatio(return_list, basereturn_list):
            excess_return = return_list - basereturn_list  # 计算每期超额收益
            aer = pow((excess_return + 1).cumprod()[-1], 365 / len(self.Date_bt)) - 1
            return_std = np.std(return_list) * np.sqrt(365)
            return aer / return_std

        print('SharpeRatio = %.3f' % SharpeRatio(self.strategy['return'], self.strategy['000001_SH_rate']))

    def print_MaxDrawdown(self):
        def MaxDrawdown(return_list):
            i = np.argmax(
                (np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
            if i == 0:
                return 0
            j = np.argmax(return_list[:i])  # 开始位置
            mdd = ((return_list[j]) - (return_list[i])) / (return_list[j])
            self.mdd = mdd
            return mdd

        def MddHappen(return_list):
            i = np.argmax(
                (np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
            if i == 0:
                return 0
            j = np.argmax(return_list[:i])  # 开始位置
            print('The MaxDrawdown happens at ')
            print(j)
            print('The MaxDrawdown keeps for ')
            print(i - j)
            return 0

        print('MaxDrawdown = %.3f' % MaxDrawdown(self.strategy['value']))
        MddHappen(self.strategy['value'])

    def print_Calmar(self):
        def Calmar(self):
            return self.aer / self.mdd

        print('Calmar = %.3f' % Calmar(self))
