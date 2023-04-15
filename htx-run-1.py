import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

train_data = pd.read_csv('./train.csv', parse_dates=['month'])
test_data = pd.read_csv('./test.csv', parse_dates=['month'])
au_train_data = pd.read_csv('auxiliary-data/comCountTrain.csv')
au_test_data = pd.read_csv('auxiliary-data/comCountTest.csv')

train_data.drop(['block', 'eco_category', 'elevation', 'planning_area', 'street_name'], axis=1, inplace=True)
test_data.drop(['block', 'eco_category', 'elevation', 'planning_area', 'street_name'], axis=1, inplace=True)
au_test_data = au_test_data.loc[:, ["commercialCount_5","marketCount_3","shoppingCount_3","stationCount_2"]]
au_train_data = au_train_data.loc[:, ["commercialCount_5","marketCount_3","shoppingCount_3","stationCount_2"]]
all_train_data = pd.concat([train_data, au_train_data], axis=1, ignore_index=False)
all_test_data = pd.concat([test_data, au_test_data], axis=1, ignore_index=False)


# convert flat_type to int
def process_value(value):
    if value.startswith('e'):
        return 6
    elif value.startswith('m'):
        return 7
    else:
        return int(value[0])


all_train_data['flat_type'] = all_train_data['flat_type'].apply(process_value)
all_test_data['flat_type'] = all_test_data['flat_type'].apply(process_value)

# storey_range process
def transfer(x):
    storeys = x.split(' to ')
    return int(int(storeys[0])+int(storeys[1]))/2


all_train_data['storey_range'] = all_train_data['storey_range'].apply(transfer)
all_test_data['storey_range'] = all_test_data['storey_range'].apply(transfer)

# convert time to float
all_train_data['sell_time'] = all_train_data['month'].apply(lambda x: x.year + x.month / 12)
all_train_data.drop('month', axis=1, inplace=True)
all_test_data['sell_time'] = all_test_data['month'].apply(lambda x: x.year + x.month / 12)
all_test_data.drop('month', axis=1, inplace=True)

# # add age data
# age_train_data = pd.read_csv('auxiliary-data/WithAge.csv')
# age_train_data = age_train_data.loc[:, ["0-14", "15-29", "30-59", "60+"]]
# all_train_data = pd.concat([all_train_data, age_train_data], axis=1, ignore_index=False)

# age_test_data = pd.read_csv("auxiliary-data/TestWithAge.csv")
# age_test_data = age_test_data.loc[:, ["0-14", "15-29", "30-59", "60+"]]
# all_test_data = pd.concat([all_test_data, age_test_data], axis=1, ignore_index=False)

# add school data, distance = 3
school_train_data = pd.read_csv('auxiliary-data/schoolCountTrain_1.csv')
school_train_data = school_train_data.loc[:, ["primaryCount", "secondaryCount"]]
# school_train_data.rename(columns={'primaryCount': 'primaryCount_3', 'secondaryCount': 'secondaryCount_3'}, inplace=True)
all_train_data = pd.concat([all_train_data, school_train_data], axis=1, ignore_index=False)

school_test_data = pd.read_csv('auxiliary-data/schoolCountTest_1.csv')
school_test_data = school_test_data.loc[:, ['primaryCount', 'secondaryCount']]
all_test_data = pd.concat([all_test_data, school_test_data], axis=1, ignore_index=False)


# use all_train_data and all_test_data

all_train_data = all_train_data.dropna()

# 对分类特征进行编码
categorical_features = ['town', 'flat_model', 'subzone', 'region']
for col in categorical_features:
    lbl = LabelEncoder()
    all_train_data[col] = lbl.fit_transform(all_train_data[col])
    all_test_data[col] = lbl.fit_transform(all_test_data[col])


def evaluate(model, train, test):
    # 将数据分为特征和目标变量
    X = train.drop('resale_price', axis=1)
    y = train['resale_price']

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 定义要搜索的参数网格
    tree_param_grid = {
        'n_estimators': [10000],
        'learning_rate': [0.1],
        #     'num_leaves': [31, 50],
        # 'max_depth': [None, 15, 17, 19],
        'max_depth': [None],
        # 'min_child_samples': [20, 30, 50]
    }

    linear_param_grid = {
        'fit_intercept': [True, False],
    }

    catboost_param_grid = {
        'iterations': [500, 5000],
        'depth': [10, 20, 30],
        'learning_rate': [0.1],
        'loss_function': ['RMSE']
    }

    xbg_param_grid = {
        'xgb__n_estimators': [100, 200, 500],
        'xgb__max_depth': [4, 6, 8],
        'xgb__learning_rate': [0.01, 0.05, 0.1]
    }
    preprocessing_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('xgb', model)
    ])

    # 使用GridSearchCV搜索最佳参数
    # grid_search = GridSearchCV(model, tree_param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
    # grid_search = GridSearchCV(model, linear_param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
    grid_search = GridSearchCV(model, catboost_param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=1)
    # grid_search = GridSearchCV(estimator=pipeline, param_grid=xbg_param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)

    grid_search.fit(X_train, y_train)
    # 输出最佳参数
    print("最佳参数：", grid_search.best_params_)

    # 使用最佳参数重新训练模型
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # 进行预测
    y_pred = best_model.predict(X_test)

    # 计算并输出均方误差
    mse = mean_squared_error(y_test, y_pred)
    print("均方误差：", mse)

    # 使用训练好的模型进行预测
    test_pred = best_model.predict(test)

    # 为预测结果创建一个新的DataFrame
    result_df = pd.DataFrame({'Id': np.arange(len(test_pred)), 'Predicted': test_pred})

    # 将结果保存到CSV文件
    result_df.to_csv('predictions.csv', index=False)


# LightGBM
# lgbm = LGBMRegressor()
# evaluate(lgbm, all_train_data, all_test_data)

# Linear Regression
# regressor = LinearRegression()
# evaluate(regressor, all_train_data, all_test_data)

# CatBoost
catboost_regressor = CatBoostRegressor()
evaluate(catboost_regressor, all_train_data, all_test_data)

# XGBRegressor
# xgb_regressor = XGBRegressor()
# evaluate(xgb_regressor, all_train_data, all_test_data)
