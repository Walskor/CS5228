import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv('./train.csv', parse_dates=['month'])
test_data = pd.read_csv('./test.csv', parse_dates=['month'])
au_train_data = pd.read_csv('auxiliary-data/comCountTrain.csv')
au_test_data = pd.read_csv('auxiliary-data/comCountTest.csv')

train_data.drop(['block', 'eco_category', 'elevation', 'planning_area'], axis=1, inplace=True)
test_data.drop(['block', 'eco_category', 'elevation', 'planning_area'], axis=1, inplace=True)
au_test_data = au_test_data.loc[:, ["commercialCount_5","marketCount_3","shoppingCount_3","stationCount_2"]]
au_train_data = au_train_data.loc[:, ["commercialCount_5","marketCount_3","shoppingCount_3","stationCount_2"]]

com_train_data = pd.concat([train_data, au_train_data], axis=1, ignore_index=False)
com_test_data = pd.concat([test_data, au_test_data], axis=1, ignore_index=False)

# data preprocessing
# convert flat_type to int
def process_value(value):
    if value.startswith('e'):
        return 6
    elif value.startswith('m'):
        return 7
    else:
        return int(value[0])


com_train_data['flat_type'] = com_train_data['flat_type'].apply(process_value)
com_test_data['flat_type'] = com_test_data['flat_type'].apply(process_value)

# storey_range process
def transfer(x):
    storeys = x.split(' to ')
    return int(int(storeys[0])+int(storeys[1]))/2


com_train_data['storey_range'] = com_train_data['storey_range'].apply(transfer)
com_test_data['storey_range'] = com_test_data['storey_range'].apply(transfer)

# convert time to float
com_train_data['sell_time'] = com_train_data['month'].apply(lambda x: x.year + x.month / 12)
com_train_data.drop('month', axis=1, inplace=True)
com_test_data['sell_time'] = com_test_data['month'].apply(lambda x: x.year + x.month / 12)
com_test_data.drop('month', axis=1, inplace=True)

# Run Model and Predict
# 对分类特征进行编码
categorical_features = ['town', 'street_name', 'flat_model', 'subzone', 'region']
for col in categorical_features:
    lbl = LabelEncoder()
    com_train_data[col] = lbl.fit_transform(com_train_data[col])

# 将数据分为特征和目标变量
X = com_train_data.drop('resale_price', axis=1)
y = com_train_data['resale_price']

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建LightGBM回归器
lgbm = LGBMRegressor()

# 定义要搜索的参数网格
param_grid = {
    'n_estimators': [7000, 9000],
    'learning_rate': [0.1,],
#     'num_leaves': [31, 50],
    'max_depth': [None, 17, 34],
    # 'min_child_samples': [20, 30]
}

# 使用GridSearchCV搜索最佳参数
grid_search = GridSearchCV(lgbm, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数：", grid_search.best_params_)

# 使用最佳参数重新训练模型
best_lgbm = grid_search.best_estimator_
best_lgbm.fit(X_train, y_train)

# 进行预测
y_pred = best_lgbm.predict(X_test)

# 计算并输出均方误差
mse = mean_squared_error(y_test, y_pred)
print("均方误差：", mse)

# 对com_test_data的分类特征进行编码，与训练数据集保持一致
for col in categorical_features:
    lbl = LabelEncoder()
    com_test_data[col] = lbl.fit_transform(com_test_data[col])

# 使用训练好的模型进行预测
test_pred = best_lgbm.predict(com_test_data)

# 为预测结果创建一个新的DataFrame
result_df = pd.DataFrame({'Id': np.arange(len(test_pred)), 'Predicted': test_pred})

# 将结果保存到CSV文件
result_df.to_csv('predictions.csv', index=False)