import pandas as pd
import numpy as np
import data
class Model:
    # Common base class for all employees
    def __init__(self,name):
        self.name = name
    def read_data(self):
        self.data = data.data.data
    def train(self,time_train,time_test):
        from datetime import timedelta
        self.end_train_time = self.initial_train_time + timedelta(days=time_train - 1)
        self.X_train = data.data.X_withdate[(data.data.X_withdate['Date']>= self.initial_train_time) & (data.data.X_withdate['Date']<= self.end_train_time)]
        self.X_train = self.X_train.drop(columns = ["Date","Code"])
        self.y_train = data.data.y[(data.data.y['Date'] >= self.initial_train_time) & (data.data.y['Date'] <= self.end_train_time)]
        self.y_train = self.y_train.drop(columns="Date")
        self.X_train = data.data.scaler.transform(self.X_train)
        self.clf =self.clf.fit(self.X_train,self.y_train)
        self.initial_train_time = self.initial_train_time + timedelta(days = time_test)
    def test(self,time_train,time_test):
        from datetime import timedelta
        self.end_test_time =self.initial_test_time + timedelta(days=time_test-1)
        print(self.end_test_time)
        self.X_test = data.data.X_withdate[(data.data.X_withdate['Date'] >= self.initial_test_time) & (data.data.X_withdate['Date'] <= self.end_test_time)]
        self.date_temp = self.X_test['Date']
        self.X_test = self.X_test.drop(columns = ["Date","Code"])
        self.y_test = data.data.y[(data.data.y['Date'] >= self.initial_test_time) & (data.data.y['Date'] <= self.end_test_time)]
        self.X_test = data.data.scaler.transform(self.X_test)
        self.y_predict_temp = self.clf.predict_proba(self.X_test)[:,1]
        self.y_predict_temp = pd.DataFrame({'Date':self.date_temp,'result':self.y_predict_temp})
        self.y_predict = self.y_predict.append(self.y_predict_temp)
        self.data['y_pred'] = pd.Series()
        self.initial_test_time = self.initial_train_time + timedelta(days = time_train+1)
    def roll_train(self,train_method,time_train,time_test):
        from datetime import datetime,timedelta
        self.y_predict = pd.DataFrame()
        self.initial_train_time = data.data.date[0]
        # self.initial_train_time = datetime(2008,1,2)
        self.initial_test_time = self.initial_train_time + timedelta(days = time_train+1)
        self.train_method= train_method
        self.end_test_time = self.initial_train_time
        if self.train_method == 'DecisionTree':
            from sklearn import tree
            self.clf = tree.DecisionTreeClassifier()
        if self.train_method =='RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            self.clf = RandomForestClassifier()
        while self.end_test_time <= data.data.date.iloc[-1]:
            self.train(self, time_train,time_test)
            self.test(self,time_train,time_test)
        self.data['y_pred'][(self.data['Date'] >= self.y_predict['Date'].iloc[0]) & (self.data['Date'] <= self.y_predict['Date'].iloc[-1])] =  Model.y_predict['result'].values
    def evaluation(self):
        from sklearn import metrics
        self.report = metrics.classification_report(self.data_new['y'],self.data_new['y_pred'])
        print(self.report)
        self.AUC = metrics.roc_auc_score(self.data_new['y'],self.data_new['y_pred'])
        print("The AUC is ")
        print(self.AUC)



