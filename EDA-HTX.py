import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine, Unit


train_data = pd.read_csv('train.csv', parse_dates=['month'])
test_data = pd.read_csv('test.csv', parse_dates=['month'])

# eda_data = train_data.loc[:, ["month", "town", "block", "street_name", "region","floor_area_sqm", "resale_price"]]
eda_data = train_data
eda_data["year"] = eda_data["month"].dt.strftime('%Y-%m')
# eda_data_pre = eda_data.assign(price_pre_sqm=eda_data['resale_price'] / eda_data['floor_area_sqm'])
eda_data = eda_data.assign(price_pre_sqm=eda_data['resale_price'] / eda_data['floor_area_sqm'])


#
# do some EDA
#

# cat_cols = ["town", "block", "street_name", "region"]
# for col in cat_cols:
#     # compute and display mean
#     # group_data = eda_data.groupby(col)["resale_price"].mean()
#     # group_data.plot(kind='bar', stacked=True, figsize=(640, 480))
#     # plt.show()
#     # compute and display Median, quartiles, outliers
#     # groups = eda_data_pre.groupby(col)
#     # fig, ax = plt.subplots(figsize=(10, 6))
#     # index = 0
#     # for name, group in groups:
#     #     ax.boxplot(group['resale_price'], positions=[index], vert=False, widths=0.5)
#     #     index += 1
#     # ax.set_xlabel('resale_price')
#     # ax.set_ylabel(col)
#     # ax.set_xticks(range(len(groups)))
#     # ax.set_xticklabels(groups.groups.keys())
#     # plt.show()
#     group_data = eda_data_pre.groupby(col)["resale_price"]

# group_data = eda_data_pre.groupby(['region', 'year'])["resale_price"]
# group_data
# fig, ax = plt.subplots(figsize=(10, 6))
# index = 0
# for name, group in group_data:
#     if ((name[0] != 'ang mo kio') | (index>5)):
#         break;
# #     print(name)
# #     print(group)
#     index += 1
#     q1 = group.quantile(0.25)
#     q3 = group.quantile(0.75)
#     iqr = q3 - q1
#     upper_limit = q3 + 1.5 * iqr
#     lower_limit = q1 - 1.5 * iqr
#     group2 = group[(group >= lower_limit) & (group <= upper_limit)]
#     value_counts = group2.value_counts()
#     print(value_counts)
#     plt.bar(value_counts.index, value_counts.values)
#     plt.tight_layout()
#     plt.show()

# fig, ax = plt.subplots(figsize=(10, 6))
# index = 0
# for name, group in group_data:
# #     if ((name[0] != 'ang mo kio') | (index>5)):
#     print(name)
#     if (name[0] != 'central region'):
#         break;
# #     print(group)
#     q1 = group.quantile(0.25)
#     q3 = group.quantile(0.75)
#     iqr = q3 - q1
#     upper_limit = q3 + 1.5 * iqr
#     lower_limit = q1 - 1.5 * iqr
#     group2 = group[(group >= lower_limit) & (group <= upper_limit)]
#     print(len(group))
#     print(len(group2))

#     variance = np.var(group)
#     mean = np.mean(group)
#     median = np.median(group)
#     print("The variance of the data is:", variance)
#     print("The mean of the data is:", mean)
#     print("The median of the data is:", median)
#     variance2 = np.var(group2)
#     mean2 = np.mean(group2)
#     median2 = np.median(group2)
#     print("The variance of the data after removing the discrete values is:", variance2)
#     print("The mean of the data after removing the discrete values is:", mean2)
#     print("The median of the data after removing the discrete values is:", median2)
#     ax.boxplot(group2, positions=[index], vert=False)
#     index += 1
#
# plt.show()

# 计算一公里以内的commerical centres；markets hawker centres；shopping malls；train stations数量
# 距离可配置
# data_place是有resale_price的数据
# data_ref是commerical centres/markets hawker centres/shopping malls/train stations数据
# dis是距离参数，单位为km
def computeNumInDis(data_place, data_ref, dis):
    count = 0
    for _, centre in data_ref.iterrows():
        distance = haversine((data_place['latitude'], data_place['longitude']), (centre['latitude'], centre['longitude']),
                             unit=Unit.KILOMETERS)
        if distance <= dis:
            count += 1
    return count


# distance_threshold = 1
#
# commerical_data = pd.read_csv("auxiliary-data/sg-commerical-centres.csv")
# commerical_data = commerical_data.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
# eda_data["commercialCount"] = eda_data.apply(lambda data_place : computeNumInDis(data_place, commerical_data, distance_threshold), axis=1)
# print("commercialCount")
# market_data = pd.read_csv("auxiliary-data/sg-gov-markets-hawker-centres.csv")
# market_data = market_data.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
# eda_data["marketCount"] = eda_data.apply(lambda data_place : computeNumInDis(data_place, market_data, distance_threshold), axis=1)
# print("marketCount")
# shopping_data = pd.read_csv("auxiliary-data/sg-shopping-malls.csv")
# shopping_data = shopping_data.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
# eda_data["shoppingCount"] = eda_data.apply(lambda data_place : computeNumInDis(data_place, shopping_data, distance_threshold), axis=1)
# print("shoppingCount")
# train_station_data = pd.read_csv("auxiliary-data/sg-train-stations.csv")
# train_station_data = train_station_data.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
# eda_data["stationCount"] = eda_data.apply(lambda data_place : computeNumInDis(data_place, train_station_data, distance_threshold), axis=1)
# print("stationCount")
# eda_data.to_csv("auxiliary-data/comCount.csv")

#
# do some EDA based on comCount.csv
#

# load comCount.csv and check
# com_data = pd.read_csv('auxiliary-data/comCount.csv', parse_dates=['month'])

# 1km for commercialCount is not a good distance_threshold, replace with 5 and recompute
# commercial_distance_threshold = 5
# commerical_data = pd.read_csv("auxiliary-data/sg-commerical-centres.csv")
# commerical_data = commerical_data.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
# com_data["commercialCount_5"] = com_data.apply(lambda data_place : computeNumInDis(data_place, commerical_data, commercial_distance_threshold), axis=1)
# com_data.to_csv("auxiliary-data/comCount2.csv")

# com_data = pd.read_csv('auxiliary-data/comCount2.csv', parse_dates=['month'])
# Try some other values
# market_distance_threshold = 3
# market_data = pd.read_csv("auxiliary-data/sg-gov-markets-hawker-centres.csv")
# market_data = market_data.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
# com_data["marketCount_3"] = com_data.apply(lambda data_place : computeNumInDis(data_place, market_data, market_distance_threshold), axis=1)
# print("marketCount")
# shopping_distance_threshold = 3
# shopping_data = pd.read_csv("auxiliary-data/sg-shopping-malls.csv")
# shopping_data = shopping_data.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
# com_data["shoppingCount_3"] = com_data.apply(lambda data_place : computeNumInDis(data_place, shopping_data, shopping_distance_threshold), axis=1)
# print("shoppingCount")
# station_distance_threshold = 2
# train_station_data = pd.read_csv("auxiliary-data/sg-train-stations.csv")
# train_station_data = train_station_data.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
# com_data["stationCount"] = com_data.apply(lambda data_place : computeNumInDis(data_place, train_station_data, station_distance_threshold), axis=1)
# print("stationCount_2")
# com_data.to_csv("auxiliary-data/comCount3.csv")

#
# use the new data, generate final csv
#
# com_data = pd.read_csv('auxiliary-data/comCount3.csv', parse_dates=['month'])
# final_data = com_data.loc[:, ["resale_price","price_pre_sqm","commercialCount_5","marketCount_3","shoppingCount_3","stationCount"]]
# final_data = final_data.rename(columns={'stationCount': 'stationCount_2'})
# final_data.to_csv("auxiliary-data/comCountTrain.csv")


#
# generate the test data
#
test_eda_data = test_data.loc[:, ['latitude','longitude']]

commercial_distance_threshold = 5
commerical_data = pd.read_csv("auxiliary-data/sg-commerical-centres.csv")
commerical_data = commerical_data.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
test_eda_data["commercialCount_5"] = test_eda_data.apply(lambda data_place : computeNumInDis(data_place, commerical_data, commercial_distance_threshold), axis=1)
market_distance_threshold = 3
market_data = pd.read_csv("auxiliary-data/sg-gov-markets-hawker-centres.csv")
market_data = market_data.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
test_eda_data["marketCount_3"] = test_eda_data.apply(lambda data_place : computeNumInDis(data_place, market_data, market_distance_threshold), axis=1)
print("marketCount")
shopping_distance_threshold = 3
shopping_data = pd.read_csv("auxiliary-data/sg-shopping-malls.csv")
shopping_data = shopping_data.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
test_eda_data["shoppingCount_3"] = test_eda_data.apply(lambda data_place : computeNumInDis(data_place, shopping_data, shopping_distance_threshold), axis=1)
print("shoppingCount")
station_distance_threshold = 2
train_station_data = pd.read_csv("auxiliary-data/sg-train-stations.csv")
train_station_data = train_station_data.rename(columns={'lat': 'latitude', 'lng': 'longitude'})
test_eda_data["stationCount_2"] = test_eda_data.apply(lambda data_place : computeNumInDis(data_place, train_station_data, station_distance_threshold), axis=1)
print("stationCount")
test_eda_data.to_csv("auxiliary-data/comCountTest.csv")