import Model
pre_method = 'Rescaling'
train_method = 'RandomForest'
time_train = 300
time_test = 10
n_stock_select = 10
seed = 41
model = Model.Model
model.read_data(model)
model.split_data(model)
model.pre_process(model,pre_method)
model.roll_train(model,train_method,time_train,time_test)
model.parameter(model,n_stock_select,seed)
model.struct_strategy(model)
model.merging_index(model)
model.back_test(model)

