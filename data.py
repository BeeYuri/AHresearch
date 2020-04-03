import pandas as pd
class data:
    def __init__(self,name):
        self
    def read_data(self):
        self.data = pd.read_pickle("F:/AHresearch/Data Pre-process2/Factor.pkl")
        self.data = self.data.sort_values(by = 'Date')
        self.date = pd.read_csv("F:/AHresearch/DateList.csv")
        self.date = pd.to_datetime(self.date['Date'],errors='coerce')
    def split_data(self):
        self.X_withdate = self.data.loc[:, 'Date':'vol_ratio']
        self.X_withoutdate = self.data.loc[:, 'close':'vol_ratio']
        self.y = self.data.loc[:, ['Date','y']]
    def pre_process(self, pre_method):
        from sklearn import preprocessing
        if pre_method == 'Standardization':
            self.scaler = preprocessing.StandardScaler().fit(self.X_withoutdate)
        if pre_method == 'Rescaling':
            self.scaler = preprocessing.MinMaxScaler().fit(self.X_withoutdate)
        if pre_method == 'Scaling to unit length':
            self.scaler = preprocessing.MaxAbsScaler().fit(self.X_withoutdate)

