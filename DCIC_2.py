import os, codecs
import pandas as pd
import numpy as np

PATH = '../input/'
# 共享单车轨迹数据
bike_track = pd.concat([
    pd.read_csv(PATH + 'gxdc_gj20201221.csv'),
    pd.read_csv(PATH + 'gxdc_gj20201222.csv'),
    pd.read_csv(PATH + 'gxdc_gj20201223.csv'),
    pd.read_csv(PATH + 'gxdc_gj20201224.csv'),
    pd.read_csv(PATH + 'gxdc_gj20201225.csv')

])

# 按照单车ID和时间进行排序
bike_track = bike_track.sort_values(['BICYCLE_ID', 'LOCATING_TIME'])
import folium
m = folium.Map(location=[24.482426, 118.157606], zoom_start=12)
my_PolyLine=folium.PolyLine(locations=bike_track[bike_track['BICYCLE_ID'] == '000152773681a23a7f2d9af8e8902703'][['LATITUDE', 'LONGITUDE']].values,weight=5)
m.add_children(my_PolyLine)
def bike_fence_format(s):
    s = s.replace('[', '').replace(']', '').split(',')
    s = np.array(s).astype(float).reshape(5, -1)
    return s

# 共享单车停车点位（电子围栏）数据
bike_fence = pd.read_csv(PATH + 'gxdc_tcd.csv')
bike_fence['FENCE_LOC'] = bike_fence['FENCE_LOC'].apply(bike_fence_format)
import folium
m = folium.Map(location=[24.482426, 118.157606], zoom_start=12)

for data in bike_fence['FENCE_LOC'].values[:100]:
    folium.Marker(
        data[0, ::-1]
    ).add_to(m)
m
# 共享单车订单数据
bike_order = pd.read_csv(PATH + 'gxdc_dd.csv')
bike_order = bike_order.sort_values(['BICYCLE_ID', 'UPDATE_TIME'])
import folium
m = folium.Map(location=[24.482426, 118.157606], zoom_start=12)
my_PolyLine=folium.PolyLine(locations=bike_order[bike_order['BICYCLE_ID'] == '0000ff105fd5f9099b866bccd157dc50'][['LATITUDE', 'LONGITUDE']].values,weight=5)
m.add_children(my_PolyLine)
# 轨道站点进站客流数据
rail_inflow = pd.read_excel(PATH + 'gdzdtjsj_jzkl.csv')
rail_inflow = rail_inflow.drop(0)

# 轨道站点出站客流数据
rail_outflow = pd.read_excel(PATH + 'gdzdtjsj_czkl.csv')
rail_outflow = rail_outflow.drop(0)

# 轨道站点闸机设备编码数据
rail_device = pd.read_excel(PATH + 'gdzdkltj_zjbh.csv')
rail_device.columns = [
    'LINE_NO', 'STATION_NO', 'STATION_NAME',
    'A_IN_MANCHINE', 'A_OUT_MANCHINE',
    'B_IN_MANCHINE', 'B_OUT_MANCHINE'
]
rail_device = rail_device.drop(0)
# 得出停车点 LATITUDE 范围
bike_fence['MIN_LATITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.min(x[:, 1]))
bike_fence['MAX_LATITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.max(x[:, 1]))

# 得到停车点 LONGITUDE 范围
bike_fence['MIN_LONGITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.min(x[:, 0]))
bike_fence['MAX_LONGITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.max(x[:, 0]))

from geopy.distance import geodesic
# 根据停车点 范围 计算具体的面积
bike_fence['FENCE_AREA'] = bike_fence.apply(lambda x: geodesic(
    (x['MIN_LATITUDE'], x['MIN_LONGITUDE']), (x['MAX_LATITUDE'], x['MAX_LONGITUDE'])
).meters, axis=1)

# 根据停车点 计算中心经纬度
bike_fence['FENCE_CENTER'] = bike_fence['FENCE_LOC'].apply(
    lambda x: np.mean(x[:-1, ::-1], 0)
)
import geohash
bike_order['geohash'] = bike_order.apply(
    lambda x: geohash.encode(x['LATITUDE'], x['LONGITUDE'], precision=6),
axis=1)

bike_fence['geohash'] = bike_fence['FENCE_CENTER'].apply(
    lambda x: geohash.encode(x[0], x[1], precision=6)
)
bike_order['UPDATE_TIME'] = pd.to_datetime(bike_order['UPDATE_TIME'])
bike_order['DAY'] = bike_order['UPDATE_TIME'].dt.day.astype(object)
bike_order['DAY'] = bike_order['DAY'].apply(str)

bike_order['HOUR'] = bike_order['UPDATE_TIME'].dt.hour.astype(object)
bike_order['HOUR'] = bike_order['HOUR'].apply(str)
bike_order['HOUR'] = bike_order['HOUR'].str.pad(width=2,side='left',fillchar='0')

# 日期和时间进行拼接
bike_order['DAY_HOUR'] = bike_order['DAY'] + bike_order['HOUR']
bike_inflow = pd.pivot_table(bike_order[bike_order['LOCK_STATUS'] == 1],
                   values='LOCK_STATUS', index=['geohash'],
                    columns=['DAY_HOUR'], aggfunc='count', fill_value=0
)

bike_outflow = pd.pivot_table(bike_order[bike_order['LOCK_STATUS'] == 0],
                   values='LOCK_STATUS', index=['geohash'],
                    columns=['DAY_HOUR'], aggfunc='count', fill_value=0
)
bike_inflow.loc['wsk593'].plot()
bike_outflow.loc['wsk593'].plot()
plt.xticks(list(range(bike_inflow.shape[1])), bike_inflow.columns, rotation=40)
plt.legend(['入流量', '出流量'])
bike_inflow.loc['wsk52r'].plot()
bike_outflow.loc['wsk52r'].plot()
plt.xticks(list(range(bike_inflow.shape[1])), bike_inflow.columns, rotation=40)
plt.legend(['入流量', '出流量'], prop = prop)
bike_inflow = pd.pivot_table(bike_order[bike_order['LOCK_STATUS'] == 1],
                   values='LOCK_STATUS', index=['geohash'],
                    columns=['DAY'], aggfunc='count', fill_value=0
)

bike_outflow = pd.pivot_table(bike_order[bike_order['LOCK_STATUS'] == 0],
                   values='LOCK_STATUS', index=['geohash'],
                    columns=['DAY'], aggfunc='count', fill_value=0
)
bike_remain = (bike_inflow - bike_outflow).fillna(0)

# 存在骑走的车数量 大于 进来的车数量
bike_remain[bike_remain < 0] = 0

# 按照天求平均
bike_remain = bike_remain.sum(1)
# 总共有993条街
bike_fence['STREET'] = bike_fence['FENCE_ID'].apply(lambda x: x.split('_')[0])

# 留存车辆 / 街道停车位总面积，计算得到密度
bike_density = bike_fence.groupby(['STREET'])['geohash'].unique().apply(
    lambda hs: np.sum([bike_remain[x] for x in hs])
) / bike_fence.groupby(['STREET'])['FENCE_AREA'].sum()

# 按照密度倒序
bike_density = bike_density.sort_values(ascending=False).reset_index()
from sklearn.neighbors import NearestNeighbors

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
knn = NearestNeighbors(metric = "haversine", n_jobs=-1, algorithm='brute')
knn.fit(np.stack(bike_fence['FENCE_CENTER'].values))
# 需要11s左右
dist, index = knn.kneighbors(bike_order[['LATITUDE','LONGITUDE']].values[:20000], n_neighbors=1)
import hnswlib
import numpy as np

p = hnswlib.Index(space='l2', dim=2)
p.init_index(max_elements=300000, ef_construction=1000, M=32)
p.set_ef(1024)
p.set_num_threads(14)

p.add_items(np.stack(bike_fence['FENCE_CENTER'].values))
index, dist = p.knn_query(bike_order[['LATITUDE','LONGITUDE']].values[:], k=1)
bike_order['fence'] = bike_fence.iloc[index.flatten()]['FENCE_ID'].values

bike_inflow = pd.pivot_table(bike_order[bike_order['LOCK_STATUS'] == 1],
                   values='LOCK_STATUS', index=['fence'],
                    columns=['DAY'], aggfunc='count', fill_value=0
)

bike_outflow = pd.pivot_table(bike_order[bike_order['LOCK_STATUS'] == 0],
                   values='LOCK_STATUS', index=['fence'],
                    columns=['DAY'], aggfunc='count', fill_value=0
)

bike_remain = (bike_inflow - bike_outflow).fillna(0)
bike_remain[bike_remain < 0] = 0
bike_remain = bike_remain.sum(1)
bike_density = bike_remain / bike_fence.set_index('FENCE_ID')['FENCE_AREA']
bike_density = bike_density.sort_values(ascending=False).reset_index()
bike_density = bike_density.fillna(0)
bike_order['UPDATE_TIME'] = pd.to_datetime(bike_order['UPDATE_TIME'])
bike_order['DAY'] = bike_order['UPDATE_TIME'].dt.day.astype(object)
bike_order['DAY'] = bike_order['DAY'].apply(str)

bike_order['HOUR'] = bike_order['UPDATE_TIME'].dt.hour.astype(object)
bike_order['HOUR'] = bike_order['HOUR'].apply(str)
bike_order['HOUR'] = bike_order['HOUR'].str.pad(width=2,side='left',fillchar='0')

bike_order['DAY_HOUR'] = bike_order['DAY'] + bike_order['HOUR']
import folium
from folium import plugins
from folium.plugins import HeatMap

map_hooray = folium.Map(location=[24.482426, 118.157606], zoom_start=14)
HeatMap(bike_order.loc[(bike_order['DAY_HOUR'] == '2106') & (bike_order['LOCK_STATUS'] == 1),
                   ['LATITUDE', 'LONGITUDE']]).add_to(map_hooray)

for data in bike_fence['FENCE_LOC'].values[::10]:
    folium.Marker(
        data[0, ::-1]
    ).add_to(map_hooray)

map_hooray
import geohash
bike_order['geohash'] = bike_order.apply(lambda x:
                        geohash.encode(x['LATITUDE'], x['LONGITUDE'], precision=9), axis=1)

from geopy.distance import geodesic

bike_fence['MIN_LATITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.min(x[:, 1]))
bike_fence['MAX_LATITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.max(x[:, 1]))

bike_fence['MIN_LONGITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.min(x[:, 0]))
bike_fence['MAX_LONGITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.max(x[:, 0]))

bike_fence['FENCE_AREA'] = bike_fence.apply(lambda x: geodesic(
    (x['MIN_LATITUDE'], x['MIN_LONGITUDE']), (x['MAX_LATITUDE'], x['MAX_LONGITUDE'])
).meters, axis=1)

bike_fence['FENCE_CENTER'] = bike_fence['FENCE_LOC'].apply(
    lambda x: np.mean(x[:-1, ::-1], 0)
)

import geohash
bike_order['geohash'] = bike_order.apply(
    lambda x: geohash.encode(x['LATITUDE'], x['LONGITUDE'], precision=7),
axis=1)

bike_fence['geohash'] = bike_fence['FENCE_CENTER'].apply(
    lambda x: geohash.encode(x[0], x[1], precision=7)
)

bike_inflow = pd.pivot_table(bike_order[bike_order['LOCK_STATUS'] == 1],
                   values='LOCK_STATUS', index=['geohash'],
                    columns=['DAY'], aggfunc='count', fill_value=0
)

bike_outflow = pd.pivot_table(bike_order[bike_order['LOCK_STATUS'] == 0],
                   values='LOCK_STATUS', index=['geohash'],
                    columns=['DAY'], aggfunc='count', fill_value=0
)

bike_remain = (bike_inflow - bike_outflow).fillna(0)
bike_remain[bike_remain < 0] = 0
bike_remain = bike_remain.sum(1)
bike_fence['DENSITY'] = bike_fence['geohash'].map(bike_remain).fillna(0)
from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(metric = "haversine", n_jobs=-1, algorithm='brute')
knn.fit(np.stack(bike_fence['FENCE_CENTER'].values))
def fence_recommend1(fence_id):
    fence_center = bike_fence.loc[bike_fence['FENCE_ID']==fence_id, 'FENCE_CENTER'].values[0]
    # 具体街道方向
    fence_dir = fence_id.split('_')[1]
    # 根据距离计算最近的20个待选位置
    dist, index = knn.kneighbors([fence_center], n_neighbors=20)

    # 对每个待选位置进行筛选
    for idx in index[0]:
        # 剔除已经有很多车的
        if bike_fence.iloc[idx]['DENSITY'] > 10:
            continue
        # 剔除需要过街的
        if fence_dir not in bike_fence.iloc[idx]['FENCE_ID']:
            continue
        return bike_fence.iloc[idx]['FENCE_ID']

    return None
def fence_recommend2(la, lo):
    dist, index = knn.kneighbors([[la, lo]], n_neighbors=20)
    for idx in index[0][1:]:
        if bike_fence.iloc[idx]['DENSITY'] > 10:
            continue
        return bike_fence.iloc[idx]['FENCE_ID']

    return None



