import Model
import data
import utils
import log
log = log.log
log.struct_log(log)

pre_method = 'Rescaling'
train_method = 'RandomForest'
time_train = 300
time_test = 10
n_stock_select = 10
seed = 41

data = data.data
data.read_data(data)
data.split_data(data)
data.pre_process(data,pre_method)

model = Model.Model
model.read_data(model)
model.roll_train(model,train_method,time_train,time_test)

utils = utils.utils
utils.parameter(utils,n_stock_select,seed)
utils.struct_strategy(utils)
utils.merging_index(utils)

log.logger.info(utils.strategy)

utils.print_winrate(utils)
utils.plot_value(utils)
utils.print_ann_return(utils)
utils.print_ann_exces_return(utils)
utils.print_IR(utils)
utils.print_Sharpe_ratio(utils)
utils.print_MaxDrawdown(utils)
utils.print_Calmar(utils)

